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
- `boundary`: 吸收边界类型 `:habc`（默认）或 `:sponge`（Cerjan）
- `v_ref`: HABC 参考速度 (default: 自动取 minimum(vp[vp.>0])，sponge 时忽略)
- `sponge_factor`: Cerjan sponge 衰减系数 (default: 0.015，HABC 时忽略)
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
    v_ref::Float32=minimum(vp[vp.>0.0f0]),
    sponge_factor::Float32=0.015f0,
)
    boundary in (:habc, :sponge) ||
        throw(ArgumentError("boundary must be :habc or :sponge, got $boundary"))
    nx, nz = size(vp)

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
    nx_pad = Int32(medium.nx)
    nz_pad = Int32(medium.nz)

    # ── Warmup: 1 步触发 JIT ──
    @info "Warming up kernels..."
    _elastic2d_loop!(W, medium, source, receiver, bc,
        a_static, dt, 1, inner_nx, inner_nz, pad,
        seis_vx, seis_vz, snaps, 0, boundary, nx_pad, nz_pad)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    @info "Starting elastic2d simulation... (nx=$nx, nz=$nz, nt=$nt, boundary=$boundary)"
    elapsed = CUDA.@elapsed begin
        _elastic2d_loop!(W, medium, source, receiver, bc,
            a_static, dt, nt, inner_nx, inner_nz, pad,
            seis_vx, seis_vz, snaps, snap_interval, boundary, nx_pad, nz_pad)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_vx), Array(seis_vz), snaps
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