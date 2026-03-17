# ═══════════════════════════════════════════════════════════════════════════════
# src/equations/acoustic3d/acoustic3d.jl
# 3D声波方程唯一入口（从2D版本扩展）
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

"""
    acoustic3d(vp, rho, dh, dt, nt, f0; kwargs...) -> (seis_vx, seis_vy, seis_vz, snaps)

3D 声波模拟，一步到位。

# 参数
- `vp, rho`: [nx × ny × nz] Float32 速度模型
- `dh`: 网格间距 (m)
- `dt`: 时间步长 (s)
- `nt`: 时间步数
- `f0`: 震源主频 (Hz)

# 关键字参数
- `sx, sy, sz`: 震源坐标向量
- `rx, ry, rz`: 接收器坐标向量
- `nbc`: 吸收边界层数 (default: 50)
- `fd_order`: 有限差分阶数 (default: 8)
- `snap_interval`: 快照间隔，0=不保存 (default: 0)
- `snap_plane`: 快照切片方向 :xy, :xz, :yz (default: :xz)
- `snap_index`: 快照切片索引 (default: ny÷2)
- `v_ref`: HABC 参考速度 (default: 自动取 minimum(vp[vp.>0]))
"""
function acoustic3d(
    vp::Array{Float32,3}, rho::Array{Float32,3},
    dh::Float32, dt::Float32, nt::Int, f0::Float32;
    sx::Vector{Int},
    sy::Vector{Int},
    sz::Vector{Int},
    rx::AbstractVector{Int},
    ry::AbstractVector{Int},
    rz::AbstractVector{Int},
    nbc::Int=50,
    fd_order::Int=8,
    snap_interval::Int=0,
    snap_plane::Symbol=:xz,
    snap_index::Int=0,
    v_ref::Float32=minimum(vp[vp.>0.0f0])
)
    nx, ny, nz = size(vp)

    # 默认快照切片位置
    if snap_index == 0
        snap_index = ny ÷ 2
    end

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_acoustic_medium_3d(vp, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = AcousticWavefield3D(nx, ny, nz, medium.pad)

    # ── HABC 边界条件 ──
    habc = init_habc_3d(nx, ny, nz, medium.pad, dt, dh, v_ref)

    # ── 震源 ──
    wavelet_data = ricker_wavelet(f0, dt, nt)
    wavelet_matrix = reshape(wavelet_data, 1, nt)
    source = init_source_3d(medium.pad, Int32.(sx), Int32.(sy), Int32.(sz), wavelet_matrix)

    # ── 接收器 ──
    receiver = init_receiver_3d(medium.pad, Int32.(collect(rx)),
        Int32.(collect(ry)), Int32.(collect(rz)), :p)

    # ── 分配地震记录 ──
    num_receivers = length(receiver.rx)
    seis_vx = CUDA.zeros(Float32, num_receivers, nt)
    seis_vy = CUDA.zeros(Float32, num_receivers, nt)
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
    inner_ny = medium.ny - fd_order
    inner_nz = medium.nz - fd_order

    # HABC 标量参数提取
    nbc_i = Int32(habc.nbc)
    nx_i = Int32(medium.nx)
    ny_i = Int32(medium.ny)
    nz_i = Int32(medium.nz)
    qx = Float32(habc.qx); qy = Float32(habc.qy); qz = Float32(habc.qz)
    qt_x = Float32(habc.qt_x); qt_y = Float32(habc.qt_y); qt_z = Float32(habc.qt_z)
    qxt = Float32(habc.qxt)

    # ── Warmup ──
    @info "Warming up 3D acoustic kernels..."
    _acoustic3d_loop!(W, medium, source, receiver, habc,
        a_static, dt, 1, inner_nx, inner_ny, inner_nz, pad,
        nbc_i, nx_i, ny_i, nz_i, qx, qy, qz, qt_x, qt_y, qt_z, qxt,
        seis_vx, seis_vy, seis_vz, snaps, 0, snap_plane, snap_index)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vy, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    @info "Starting acoustic3d simulation... (nx=$nx, ny=$ny, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _acoustic3d_loop!(W, medium, source, receiver, habc,
            a_static, dt, nt, inner_nx, inner_ny, inner_nz, pad,
            nbc_i, nx_i, ny_i, nz_i, qx, qy, qz, qt_x, qt_y, qt_z, qxt,
            seis_vx, seis_vy, seis_vz, snaps, snap_interval, snap_plane, snap_index)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_vx), Array(seis_vy), Array(seis_vz), snaps
end

"""
3D声波内部时间步循环。
"""
function _acoustic3d_loop!(W, M, S, R, H,
    a_static, dt, nt, inner_nx, inner_ny, inner_nz, pad,
    nbc, nx, ny, nz, qx, qy, qz, qt_x, qt_y, qt_z, qxt,
    seis_vx, seis_vy, seis_vz, snaps, snap_interval,
    snap_plane, snap_index)
    snap_idx = 1

    for it in 1:nt
        # ── A. Velocity phase ──
        backup_single_field_3d!(W.vx_old, W.vx, nbc, nx, ny, nz)
        backup_single_field_3d!(W.vy_old, W.vy, nbc, nx, ny, nz)
        backup_single_field_3d!(W.vz_old, W.vz, nbc, nx, ny, nz)

        update_velocity_acoustic3d!(W, M, a_static, dt, inner_nx, inner_ny, inner_nz)

        apply_habc_single_field_3d!(W.vx, W.vx_old, H.w_vx,
            qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
        apply_habc_single_field_3d!(W.vy, W.vy_old, H.w_vy,
            qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
        apply_habc_single_field_3d!(W.vz, W.vz_old, H.w_vz,
            qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)

        # ── B. Pressure phase ──
        backup_single_field_3d!(W.p_old, W.p, nbc, nx, ny, nz)

        update_pressure_acoustic3d!(W, M, a_static, dt, inner_nx, inner_ny, inner_nz)

        apply_habc_single_field_3d!(W.p, W.p_old, H.w_tau,
            qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)

        # ── C. Source injection（注入压力场）──
        inject_source_3d!(W.p, S, it, dt)

        # ── D. Record receivers ──
        record_receivers_3d!(seis_vx, W.vx, R, it)
        record_receivers_3d!(seis_vy, W.vy, R, it)
        record_receivers_3d!(seis_vz, W.vz, R, it)

        # ── E. Snapshots（保存压力场2D切片）──
        if snap_interval > 0 && it % snap_interval == 0
            si = snap_index + pad
            if snap_plane == :xz
                snaps[snap_idx] = Array(@view W.p[pad+1:end-pad, si, pad+1:end-pad])
            elseif snap_plane == :xy
                snaps[snap_idx] = Array(@view W.p[pad+1:end-pad, pad+1:end-pad, si])
            elseif snap_plane == :yz
                snaps[snap_idx] = Array(@view W.p[si, pad+1:end-pad, pad+1:end-pad])
            end
            snap_idx += 1
        end
    end
end
