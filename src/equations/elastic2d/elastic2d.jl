# src/equations/elastic2d/elastic2d.jl

using CUDA
using StaticArrays

"""
    elastic2d(vp, vs, rho, dh, dt, nt, f0; kwargs...) -> (seis_vx, seis_vz, snaps)

2D 弹性波模拟，一步到位。

# 参数
- `vp, vs, rho`: [nx × nz] Float32 速度模型
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
    v_ref::Float32=minimum(vp[vp.>0.0f0])
)
    nx, nz = size(vp)

    # ── 有限差分系数 ──
    a_static = get_fd_coefficients(fd_order)

    # ── 介质初始化 ──
    medium = init_medium(vp, vs, rho, dh, nbc, fd_order)

    # ── 波场初始化 ──
    W = Wavefield(nx, nz, medium.pad)

    # ── HABC 边界条件 ──
    habc = init_habc(nx, nz, medium.pad, dt, dh, v_ref)

    # ── 震源 ──
    wavelet_data = ricker_wavelet(f0, dt, nt)
    wavelet_matrix = reshape(wavelet_data, 1, nt)
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

    # ── Warmup: 1 步触发 JIT ──
    @info "Warming up kernels..."
    _elastic2d_loop!(W, medium, source, receiver, habc,
        a_static, dt, 1, inner_nx, inner_nz, pad,
        seis_vx, seis_vz, snaps, 0)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    @info "Starting elastic2d simulation... (nx=$nx, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _elastic2d_loop!(W, medium, source, receiver, habc,
            a_static, dt, nt, inner_nx, inner_nz, pad,
            seis_vx, seis_vz, snaps, snap_interval)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_vx), Array(seis_vz), snaps
end

"""
内部时间步循环，不导出。
"""
function _elastic2d_loop!(W, M, S, R, H,
    a_static, dt, nt, inner_nx, inner_nz, pad,
    seis_vx, seis_vz, snaps, snap_interval)
    snap_idx = 1

    for it in 1:nt
        # A. Velocity phase
        backup_velocity!(W, H, M)
        update_velocity!(W, M, a_static, dt, inner_nx, inner_nz)
        apply_habc_velocity!(W, H, M)

        # B. Stress phase
        backup_stress!(W, H, M)
        update_stress!(W, M, a_static, dt, inner_nx, inner_nz)
        apply_habc_stress!(W, H, M)

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