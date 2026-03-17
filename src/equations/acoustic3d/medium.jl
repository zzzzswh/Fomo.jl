# src/equations/acoustic3d/medium.jl
#
# 3D声波介质：从2D扩展，增加 ny, y_max, buoy_vy

using CUDA

"""
    AcousticMedium3D{T}

3D声波介质：体积模量 κ 和三个方向的浮力。
"""
struct AcousticMedium3D{T}
    nx::Int
    ny::Int
    nz::Int
    dh::Float32
    x_max::Float32
    y_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    kappa::T        # 体积模量 κ = ρ·vp²
    buoy_vx::T      # 1/ρ at vx positions
    buoy_vy::T      # 1/ρ at vy positions
    buoy_vz::T      # 1/ρ at vz positions
end

"""
    init_acoustic_medium_3d(vp, rho, dh, nbc, fd_order)

3D声波介质初始化。
"""
function init_acoustic_medium_3d(vp::Array{Float32,3}, rho::Array{Float32,3},
    dh::Real, nbc::Int, fd_order::Int)
    M = fd_order ÷ 2
    pad = nbc + M

    nx_inner, ny_inner, nz_inner = size(vp)
    nx = nx_inner + 2pad
    ny = ny_inner + 2pad
    nz = nz_inner + 2pad

    x_max = Float32((nx_inner - 1) * dh)
    y_max = Float32((ny_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    vp_pad  = _pad_array_3d(vp, pad)
    rho_pad = _pad_array_3d(rho, pad)

    # 体积模量 κ = ρ * vp²
    kappa = Float32.(rho_pad .* vp_pad .^ 2)

    # 浮力
    buoy_vx, buoy_vy, buoy_vz = _compute_staggered_buoyancy_3d(Float32.(rho_pad))

    return AcousticMedium3D(
        nx, ny, nz, Float32(dh), x_max, y_max, z_max, M, pad,
        to_device(kappa),
        to_device(buoy_vx),
        to_device(buoy_vy),
        to_device(buoy_vz)
    )
end
