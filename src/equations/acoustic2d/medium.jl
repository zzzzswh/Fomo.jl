# src/equations/acoustic2d/medium.jl

using CUDA

"""
    AcousticMedium{T}

声波介质：只需体积模量 κ 和浮力。
"""
struct AcousticMedium{T}
    nx::Int
    nz::Int
    dh::Float32
    x_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    kappa::T        # 体积模量 κ = ρ·vp²
    buoy_vx::T      # 1/ρ at vx positions
    buoy_vz::T      # 1/ρ at vz positions
end

"""
    init_acoustic_medium(vp, rho, dh, nbc, fd_order)

声波介质初始化。真空区域（vp=0, rho=0）自动处理。
"""
function init_acoustic_medium(vp::Matrix, rho::Matrix,
    dh::Real, nbc::Int, fd_order::Int)
    M = fd_order ÷ 2
    pad = nbc + M

    nx_inner, nz_inner = size(vp)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    vp_pad = _pad_array(vp, pad)
    rho_pad = _pad_array(rho, pad)

    # 体积模量 κ = ρ * vp²（真空区域自动为 0）
    kappa = Float32.(rho_pad .* vp_pad .^ 2)

    # 浮力：复用共享函数
    buoy_vx, buoy_vz = _compute_staggered_buoyancy(Float32.(rho_pad))

    return AcousticMedium(
        nx, nz, Float32(dh), x_max, z_max, M, pad,
        to_device(kappa),
        to_device(buoy_vx),
        to_device(buoy_vz)
    )
end