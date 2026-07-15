#  ═══════════════════════════════════════════════════════════════════════════════
# src/equations/acoustic2d/acoustic2d.jl
# 声波方程唯一入口
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

"""
    acoustic2d(vp, rho, dh, dt, nt, f0; kwargs...) -> NamedTuple

2D 声波模拟，一步到位。

# 参数
- `vp, rho`: [nx × nz] Float32 速度模型（不需要 vs）
- `dh`: 网格间距 (m)
- `dt`: 时间步长 (s)
- `nt`: 时间步数
- `f0`: 震源主频 (Hz)（用于默认 Ricker 子波与频散检查）

# 关键字参数
- `sx, sz`: 震源坐标向量（网格索引）
- `rx, rz`: 接收器坐标向量（网格索引）
- `nbc`: 吸收边界层数 (default: 50)，有效吸收区为 nbc + fd_order÷2
- `fd_order`: 有限差分阶数 (default: 8)
- `snap_interval`: 快照间隔，0=不保存 (default: 0)
- `boundary`: 吸收边界类型 `:habc`（默认）或 `:sponge`（Cerjan）
- `v_ref`: HABC 参考速度 (default: 自动取 minimum(vp[vp.>0])，sponge 时忽略)
- `sponge_factor`: Cerjan sponge 衰减系数 (default: 0.015，HABC 时忽略)
- `wavelet`: 自定义震源子波（长度 nt 的向量；default: nothing → Ricker(f0)）
- `scale_source_by_dt`: 注入前将子波乘以 dt，使震源振幅在改变 dt 时保持物理一致
  （default: false，保持旧版振幅约定；与 deepwave 对比振幅时建议 true）
- `use_cuda_graph`: HABC 且无快照时将整个时间步录制为 CUDA Graph，每步仅 1 次
  graph launch（default: true；capture 失败自动回退融合循环，结果不变）
- `verbose`: 是否打印进度日志 (default: true)

# 返回（NamedTuple）
- `seis_p`:  [n_rec × nt] 压力记录
- `seis_vx`: [n_rec × nt] 水平速度记录
- `seis_vz`: [n_rec × nt] 垂直速度记录
- `snaps`:   压力场快照序列
- `stats`:   `(kernel_time_s=...,)` GPU 主循环耗时（warmup 后）

# 场的交错位置（对比外部软件时注意半格偏移）
- p:  (i, j)；vx: (i−1/2, j)；vz: (i, j+1/2)
"""
function acoustic2d(
    vp::Matrix{Float32}, rho::Matrix{Float32},
    dh::Float32, dt::Float32, nt::Int, f0::Float32;
    sx::Vector{Int},
    sz::Vector{Int},
    rx::AbstractVector{Int},
    rz::AbstractVector{Int},
    nbc::Int=50,
    fd_order::Int=8,
    snap_interval::Int=0,
    boundary::Symbol=:habc,
    v_ref::Float32=Float32(minimum(x for x in vp if x > 0.0f0)),
    sponge_factor::Float32=0.015f0,
    wavelet::Union{Nothing,AbstractVector{<:Real}}=nothing,
    scale_source_by_dt::Bool=false,
    use_cuda_graph::Bool=true,
    verbose::Bool=true,
)
    boundary in (:habc, :sponge) ||
        throw(ArgumentError("boundary must be :habc or :sponge, got $boundary"))
    nx, nz = size(vp)

    # ── 参数校验 ──
    _check_geometry(nx, nz, sx, sz, rx, rz)
    _check_numerics(maximum(vp), minimum(x for x in vp if x > 0.0f0),
        dh, dt, f0, fd_order)

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_acoustic_medium(vp, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = AcousticWavefield(nx, nz, medium.pad)

    # ── 吸收边界 ──
    if boundary == :habc
        bc = init_habc(nx, nz, medium.pad, dt, dh, v_ref)
    else  # :sponge
        bc = init_sponge(nx, nz, medium.pad, nbc; factor=sponge_factor)
    end

    # ── 震源：默认 Ricker，可传自定义子波；多源按行展开 ──
    wavelet_data = isnothing(wavelet) ? ricker_wavelet(f0, dt, nt) : Float32.(wavelet)
    length(wavelet_data) == nt ||
        throw(ArgumentError("wavelet 长度 $(length(wavelet_data)) ≠ nt=$nt"))
    wavelet_matrix = repeat(reshape(wavelet_data, 1, nt), length(sx), 1)
    # 可选：按 dt 缩放震源，使振幅在改变 dt 时物理一致（对齐 deepwave 的做法）
    scale_source_by_dt && (wavelet_matrix .*= dt)
    source = init_source(medium.pad, Int32.(sx), Int32.(sz), wavelet_matrix)

    # ── 接收器 ──
    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :p)

    # ── 分配地震记录 ──
    num_receivers = length(receiver.rx)
    seis_p = CUDA.zeros(Float32, num_receivers, nt)
    seis_vx = CUDA.zeros(Float32, num_receivers, nt)
    seis_vz = CUDA.zeros(Float32, num_receivers, nt)

    # ── 分配快照 ──
    if snap_interval > 0
        num_snaps = nt ÷ snap_interval
        snaps = Vector{Matrix{Float32}}(undef, num_snaps)
    else
        snaps = Vector{Matrix{Float32}}()
    end

    # ── 预计算 ──
    pad = medium.pad
    inner_nx = medium.nx - fd_order
    inner_nz = medium.nz - fd_order

    # HABC 标量参数提取（循环外一次性完成，sponge 时未使用）
    nx_i = Int32(medium.nx)
    nz_i = Int32(medium.nz)
    if boundary == :habc
        # 与弹性路径统一：作用区宽度取 bc.nbc (= pad-1 = nbc+M-1)，
        # 使一阶 Higdon 作用区与权重 ramp（跨度 pad-1）完全对齐
        nbc_i = Int32(bc.nbc)
        qx = Float32(bc.qx)
        qz = Float32(bc.qz)
        qt_x = Float32(bc.qt_x)
        qt_z = Float32(bc.qt_z)
        qxt = Float32(bc.qxt)
    else
        nbc_i = Int32(nbc)
        qx = qz = qt_x = qt_z = qxt = 0.0f0
    end

    # ── Warmup ──
    verbose && @info "Warming up kernels..."
    use_graph = (boundary === :habc) && snap_interval == 0 && use_cuda_graph
    if use_graph
        _acoustic2d_loop_graph!(W, medium, source, receiver, bc,
            a_static, dt, 1, pad,
            nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_p, seis_vx, seis_vz)
    elseif boundary === :habc
        _acoustic2d_loop_fused!(W, medium, source, receiver, bc,
            a_static, dt, 1, pad,
            nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_p, seis_vx, seis_vz, snaps, 0)
    else
        _acoustic2d_loop!(W, medium, source, receiver, bc,
            a_static, dt, 1, inner_nx, inner_nz, pad,
            nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_p, seis_vx, seis_vz, snaps, 0, boundary)
    end
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_p, 0.0f0)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    verbose && @info "Starting acoustic2d simulation... (nx=$nx, nz=$nz, nt=$nt, boundary=$boundary)"
    elapsed = CUDA.@elapsed begin
        if use_graph
            _acoustic2d_loop_graph!(W, medium, source, receiver, bc,
                a_static, dt, nt, pad,
                nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
                seis_p, seis_vx, seis_vz)
        elseif boundary === :habc
            _acoustic2d_loop_fused!(W, medium, source, receiver, bc,
                a_static, dt, nt, pad,
                nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
                seis_p, seis_vx, seis_vz, snaps, snap_interval)
        else
            _acoustic2d_loop!(W, medium, source, receiver, bc,
                a_static, dt, nt, inner_nx, inner_nz, pad,
                nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
                seis_p, seis_vx, seis_vz, snaps, snap_interval, boundary)
        end
    end
    verbose && @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return (seis_p=Array(seis_p),
        seis_vx=Array(seis_vx),
        seis_vz=Array(seis_vz),
        snaps=snaps,
        stats=(kernel_time_s=Float64(elapsed),))
end

"""
声波内部时间步循环。boundary 取 :habc 或 :sponge，决定每步更新后用哪种吸收。
"""
function _acoustic2d_loop!(W, M, S, R, B,
    a_static, dt, nt, inner_nx, inner_nz, pad,
    nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_p, seis_vx, seis_vz, snaps, snap_interval, boundary::Symbol)
    snap_idx = 1

    for it in 1:nt
        # ── A. Velocity phase ──
        if boundary === :habc
            backup_single_field!(W.vx_old, W.vx, nbc, nx, nz)
            backup_single_field!(W.vz_old, W.vz, nbc, nx, nz)
            update_velocity_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_habc_single_field!(W.vx, W.vx_old, B.w_vx,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            apply_habc_single_field!(W.vz, W.vz_old, B.w_vz,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        else  # :sponge
            update_velocity_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_sponge!(W.vx, B, nx, nz)
            apply_sponge!(W.vz, B, nx, nz)
        end

        # ── B. Pressure phase ──
        if boundary === :habc
            backup_single_field!(W.p_old, W.p, nbc, nx, nz)
            update_pressure_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_habc_single_field!(W.p, W.p_old, B.w_tau,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        else  # :sponge
            update_pressure_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_sponge!(W.p, B, nx, nz)
        end

        # ── C. Source injection（注入压力场）──
        inject_source!(W.p, S, it, dt)

        # ── D. Record receivers ──
        record_receivers!(seis_p, W.p, R, it)
        record_receivers!(seis_vx, W.vx, R, it)
        record_receivers!(seis_vz, W.vz, R, it)

        # ── E. Snapshots（保存压力场）──
        if snap_interval > 0 && it % snap_interval == 0
            snaps[snap_idx] = Array(@view W.p[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end
