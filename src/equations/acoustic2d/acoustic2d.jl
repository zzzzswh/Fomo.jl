#  ═══════════════════════════════════════════════════════════════════════════════
# [NEW] src/equations/acoustic2d/acoustic2d.jl
# 声波方程唯一入口
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

"""
    acoustic2d(vp, rho, dh, dt, nt, f0; kwargs...) -> (seis_vx, seis_vz, snaps)

2D 声波模拟，一步到位。

# 参数
- `vp, rho`: [nx × nz] Float32 速度模型（不需要 vs）
- `dh`: 网格间距 (m)
- `dt`: 时间步长 (s)
- `nt`: 时间步数
- `f0`: 震源主频 (Hz)

# 关键字参数
- `sx, sz`: 震源坐标向量
- `rx, rz`: 接收器坐标向量
- `nbc`: 吸收边界层数 (default: 50)
- `fd_order`: 有限差分阶数 (default: 8)
- `snap_interval`: 快照间隔，0=不保存 (default: 0)
- `v_ref`: HABC 参考速度 (default: 自动取 minimum(vp[vp.>0]))
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
    v_ref::Float32=minimum(vp[vp.>0.0f0])
)
    nx, nz = size(vp)

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_acoustic_medium(vp, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = AcousticWavefield(nx, nz, medium.pad)

    # ── HABC 边界条件 ──
    habc = init_habc(nx, nz, medium.pad, dt, dh, v_ref)

    # ── 震源 ──
    wavelet_data = ricker_wavelet(f0, dt, nt)
    wavelet_matrix = reshape(wavelet_data, 1, nt)
    source = init_source(medium.pad, Int32.(sx), Int32.(sz), wavelet_matrix)

    # ── 接收器 ──
    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :p)

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

    # HABC 标量参数提取（循环外一次性完成）
    nbc_i = Int32(habc.nbc)
    nx_i = Int32(medium.nx)
    nz_i = Int32(medium.nz)
    qx = Float32(habc.qx)
    qz = Float32(habc.qz)
    qt_x = Float32(habc.qt_x)
    qt_z = Float32(habc.qt_z)
    qxt = Float32(habc.qxt)

    # ── Warmup ──
    @info "Warming up kernels..."
    _acoustic2d_loop!(W, medium, source, receiver, habc,
        a_static, dt, 1, inner_nx, inner_nz, pad,
        nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
        seis_vx, seis_vz, snaps, 0)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    @info "Starting acoustic2d simulation... (nx=$nx, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _acoustic2d_loop!(W, medium, source, receiver, habc,
            a_static, dt, nt, inner_nx, inner_nz, pad,
            nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_vx, seis_vz, snaps, snap_interval)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_vx), Array(seis_vz), snaps
end

"""
声波内部时间步循环。
"""
function _acoustic2d_loop!(W, M, S, R, H,
    a_static, dt, nt, inner_nx, inner_nz, pad,
    nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_vx, seis_vz, snaps, snap_interval)
    snap_idx = 1

    for it in 1:nt
        # ── A. Velocity phase ──
        # backup vx, vz
        backup_single_field!(W.vx_old, W.vx, nbc, nx, nz)
        backup_single_field!(W.vz_old, W.vz, nbc, nx, nz)
        # update
        update_velocity_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
        # HABC
        apply_habc_single_field!(W.vx, W.vx_old, H.w_vx,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        apply_habc_single_field!(W.vz, W.vz_old, H.w_vz,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # ── B. Pressure phase ──
        # backup p
        backup_single_field!(W.p_old, W.p, nbc, nx, nz)
        # update
        update_pressure_acoustic!(W, M, a_static, dt, inner_nx, inner_nz)
        # HABC
        apply_habc_single_field!(W.p, W.p_old, H.w_tau,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # ── C. Source injection（注入压力场）──
        inject_source!(W.p, S, it, dt)

        # ── D. Record receivers ──
        record_receivers!(seis_vx, W.vx, R, it)
        record_receivers!(seis_vz, W.vz, R, it)

        # ── E. Snapshots（保存压力场）──
        if snap_interval > 0 && it % snap_interval == 0
            snaps[snap_idx] = Array(@view W.p[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end
