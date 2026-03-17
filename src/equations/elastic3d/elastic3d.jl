# ═══════════════════════════════════════════════════════════════════════════════
# src/equations/elastic3d/elastic3d.jl
# 3D弹性波方程唯一入口（从2D版本扩展）
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

"""
    elastic3d(vp, vs, rho, dh, dt, nt, f0; kwargs...) -> (seis_vx, seis_vy, seis_vz, snaps)

3D 弹性波模拟，一步到位。

# 参数
- `vp, vs, rho`: [nx × ny × nz] Float32 速度模型
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
function elastic3d(
    vp::Array{Float32,3}, vs::Array{Float32,3}, rho::Array{Float32,3},
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

    if snap_index == 0
        snap_index = ny ÷ 2
    end

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_medium_3d(vp, vs, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = Wavefield3D(nx, ny, nz, medium.pad)

    # ── HABC 边界条件 ──
    habc = init_habc_3d(nx, ny, nz, medium.pad, dt, dh, v_ref)

    # ── 震源 ──
    wavelet_data = ricker_wavelet(f0, dt, nt)
    wavelet_matrix = reshape(wavelet_data, 1, nt)
    source = init_source_3d(medium.pad, Int32.(sx), Int32.(sy), Int32.(sz), wavelet_matrix)

    # ── 接收器 ──
    receiver = init_receiver_3d(medium.pad, Int32.(collect(rx)),
        Int32.(collect(ry)), Int32.(collect(rz)), :vz)

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

    # ── Warmup ──
    @info "Warming up 3D elastic kernels..."
    _elastic3d_loop!(W, medium, source, receiver, habc,
        a_static, dt, 1, inner_nx, inner_ny, inner_nz, pad,
        seis_vx, seis_vy, seis_vz, snaps, 0, snap_plane, snap_index)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vy, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    @info "Starting elastic3d simulation... (nx=$nx, ny=$ny, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _elastic3d_loop!(W, medium, source, receiver, habc,
            a_static, dt, nt, inner_nx, inner_ny, inner_nz, pad,
            seis_vx, seis_vy, seis_vz, snaps, snap_interval, snap_plane, snap_index)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_vx), Array(seis_vy), Array(seis_vz), snaps
end

"""
3D弹性波内部时间步循环。
"""
function _elastic3d_loop!(W, M, S, R, H,
    a_static, dt, nt, inner_nx, inner_ny, inner_nz, pad,
    seis_vx, seis_vy, seis_vz, snaps, snap_interval,
    snap_plane, snap_index)
    snap_idx = 1

    for it in 1:nt
        # A. Velocity phase
        backup_velocity_3d!(W, H, M)
        update_velocity_3d!(W, M, a_static, dt, inner_nx, inner_ny, inner_nz)
        apply_habc_velocity_3d!(W, H, M)

        # B. Stress phase
        backup_stress_3d!(W, H, M)
        update_stress_3d!(W, M, a_static, dt, inner_nx, inner_ny, inner_nz)
        apply_habc_stress_3d!(W, H, M)

        # C. Source injection (弹性波：注入 txx + tyy + tzz 爆炸源)
        inject_source_3d!(W.txx, S, it, dt)
        inject_source_3d!(W.tyy, S, it, dt)
        inject_source_3d!(W.tzz, S, it, dt)

        # D. Record receivers
        record_receivers_3d!(seis_vx, W.vx, R, it)
        record_receivers_3d!(seis_vy, W.vy, R, it)
        record_receivers_3d!(seis_vz, W.vz, R, it)

        # E. Snapshots（保存 vz 的2D切片）
        if snap_interval > 0 && it % snap_interval == 0
            si = snap_index + pad
            if snap_plane == :xz
                snaps[snap_idx] = Array(@view W.vz[pad+1:end-pad, si, pad+1:end-pad])
            elseif snap_plane == :xy
                snaps[snap_idx] = Array(@view W.vz[pad+1:end-pad, pad+1:end-pad, si])
            elseif snap_plane == :yz
                snaps[snap_idx] = Array(@view W.vz[si, pad+1:end-pad, pad+1:end-pad])
            end
            snap_idx += 1
        end
    end
end
