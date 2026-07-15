# src/equations/elastic2d/elastic2d.jl

using CUDA
using StaticArrays

"""
    elastic2d(vp, vs, rho, dh, dt, nt, f0; kwargs...) -> NamedTuple

2D 弹性波模拟，一步到位。

# 参数
- `vp, vs, rho`: [nx × nz] Float32 速度模型
- `dh`: 网格间距 (m)
- `dt`: 时间步长 (s)
- `nt`: 时间步数
- `f0`: 震源主频 (Hz)（用于默认 Ricker 子波与频散检查）

# 关键字参数
- `sx, sz`: 震源坐标向量（网格索引；爆炸源，注入 τxx+τzz）
- `rx, rz`: 接收器坐标向量（网格索引）
- `nbc`: 吸收边界层数 (default: 50)，有效吸收区为 nbc + fd_order÷2
- `fd_order`: 有限差分阶数 (default: 8)
- `snap_interval`: 快照间隔，0=不保存 (default: 0)
- `boundary`: 吸收边界类型 `:habc`（默认）或 `:sponge`（Cerjan）
- `v_ref`: HABC 参考速度 (default: 自动取 minimum(vp[vp.>0])，sponge 时忽略)
  注意：velocity-stress 格式中 P/S 共存于同一组场，无法按波型分设参考速度，
  S 波吸收相对 P 波略差是单一 v_ref 的固有折中
- `sponge_factor`: Cerjan sponge 衰减系数 (default: 0.015，HABC 时忽略)
- `wavelet`: 自定义震源子波（长度 nt 的向量；default: nothing → Ricker(f0)）
- `scale_source_by_dt`: 注入前将子波乘以 dt，使震源振幅在改变 dt 时保持物理一致
  （default: false，保持旧版振幅约定；与 deepwave 对比振幅时建议 true）
- `use_cuda_graph`: HABC 且无快照时将整个时间步录制为 CUDA Graph，每步仅 1 次
  graph launch（default: true；capture 失败自动回退融合循环，结果不变）
- `verbose`: 是否打印进度日志 (default: true)

# 返回（NamedTuple）
- `seis_vx`: [n_rec × nt] 水平速度记录
- `seis_vz`: [n_rec × nt] 垂直速度记录
- `snaps`:   vz 快照序列
- `stats`:   `(kernel_time_s=...,)` GPU 主循环耗时（warmup 后）

# 场的交错位置（对比外部软件时注意半格偏移）
- τxx/τzz: (i, j)；vx: (i−1/2, j)；vz: (i, j+1/2)；τxz: (i−1/2, j+1/2)
"""
function elastic2d(
    vp::Matrix{Float32}, vs::Matrix{Float32}, rho::Matrix{Float32},
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

    # ── 参数校验（弹性的频散由最短 S 波长控制）──
    _check_geometry(nx, nz, sx, sz, rx, rz)
    vmin_disp = any(x -> x > 0.0f0, vs) ? minimum(x for x in vs if x > 0.0f0) :
                minimum(x for x in vp if x > 0.0f0)
    _check_numerics(maximum(vp), vmin_disp, dh, dt, f0, fd_order)

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_medium(vp, vs, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = Wavefield(nx, nz, medium.pad)

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
    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :vz)

    # ── 分配地震记录 ──
    num_receivers = length(receiver.rx)
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
    nx_pad = Int32(medium.nx)
    nz_pad = Int32(medium.nz)

    # ── Warmup: 1 步触发 JIT ──
    verbose && @info "Warming up kernels..."
    use_graph = (boundary === :habc) && snap_interval == 0 && use_cuda_graph
    if use_graph
        _elastic2d_loop_graph!(W, medium, source, receiver, bc,
            a_static, dt, 1, pad, seis_vx, seis_vz)
    elseif boundary === :habc
        _elastic2d_loop_fused!(W, medium, source, receiver, bc,
            a_static, dt, 1, pad, seis_vx, seis_vz, snaps, 0)
    else
        _elastic2d_loop!(W, medium, source, receiver, bc,
            a_static, dt, 1, inner_nx, inner_nz, pad,
            seis_vx, seis_vz, snaps, 0, boundary, nx_pad, nz_pad)
    end
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    verbose && @info "Starting elastic2d simulation... (nx=$nx, nz=$nz, nt=$nt, boundary=$boundary)"
    elapsed = CUDA.@elapsed begin
        if use_graph
            _elastic2d_loop_graph!(W, medium, source, receiver, bc,
                a_static, dt, nt, pad, seis_vx, seis_vz)
        elseif boundary === :habc
            _elastic2d_loop_fused!(W, medium, source, receiver, bc,
                a_static, dt, nt, pad, seis_vx, seis_vz, snaps, snap_interval)
        else
            _elastic2d_loop!(W, medium, source, receiver, bc,
                a_static, dt, nt, inner_nx, inner_nz, pad,
                seis_vx, seis_vz, snaps, snap_interval, boundary, nx_pad, nz_pad)
        end
    end
    verbose && @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return (seis_vx=Array(seis_vx),
        seis_vz=Array(seis_vz),
        snaps=snaps,
        stats=(kernel_time_s=Float64(elapsed),))
end

"""
内部时间步循环，不导出。boundary ∈ {:habc, :sponge}。
"""
function _elastic2d_loop!(W, M, S, R, B,
    a_static, dt, nt, inner_nx, inner_nz, pad,
    seis_vx, seis_vz, snaps, snap_interval,
    boundary::Symbol, nx_pad::Int32, nz_pad::Int32)
    snap_idx = 1

    for it in 1:nt
        # A. Velocity phase
        if boundary === :habc
            backup_velocity!(W, B, M)
            update_velocity!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_habc_velocity!(W, B, M)
        else  # :sponge
            update_velocity!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_sponge!(W.vx, B, nx_pad, nz_pad)
            apply_sponge!(W.vz, B, nx_pad, nz_pad)
        end

        # B. Stress phase
        if boundary === :habc
            backup_stress!(W, B, M)
            update_stress!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_habc_stress!(W, B, M)
        else  # :sponge
            update_stress!(W, M, a_static, dt, inner_nx, inner_nz)
            apply_sponge!(W.txx, B, nx_pad, nz_pad)
            apply_sponge!(W.tzz, B, nx_pad, nz_pad)
            apply_sponge!(W.txz, B, nx_pad, nz_pad)
        end

        # C. Source injection (弹性波：注入 txx + tzz)
        inject_source!(W.txx, S, it, dt)
        inject_source!(W.tzz, S, it, dt)

        # D. Record receivers
        record_receivers!(seis_vx, W.vx, R, it)
        record_receivers!(seis_vz, W.vz, R, it)

        # E. Snapshots
        if snap_interval > 0 && it % snap_interval == 0
            snaps[snap_idx] = Array(@view W.vz[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end