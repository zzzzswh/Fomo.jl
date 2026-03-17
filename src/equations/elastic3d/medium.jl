# src/equations/elastic3d/medium.jl
#
# 3D弹性波介质（从2D扩展）：
#   - 增加 ny, y_max, buoy_vy
#   - 应力分量从 txx, tzz, txz 变为 txx, tyy, tzz, txy, txz, tyz
#   - mu 插值位置从 mu_txz 变为 mu_txy, mu_txz, mu_tyz

using CUDA

struct Medium3D{T}
    nx::Int
    ny::Int
    nz::Int
    dh::Float32
    x_max::Float32
    y_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    lam::T
    mu_txx::T       # mu 在正应力位置（算术平均 = 原始值）
    mu_txy::T       # mu 在 txy 位置的调和平均
    mu_txz::T       # mu 在 txz 位置的调和平均
    mu_tyz::T       # mu 在 tyz 位置的调和平均
    buoy_vx::T
    buoy_vy::T
    buoy_vz::T
    lam_2mu::T
end

function init_medium_3d(vp::Array{Float32,3}, vs::Array{Float32,3},
    rho::Array{Float32,3}, dh::Real, nbc::Int, fd_order::Int)
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
    vs_pad  = _pad_array_3d(vs, pad)
    rho_pad = _pad_array_3d(rho, pad)

    # 浮力
    buoy_vx, buoy_vy, buoy_vz = _compute_staggered_buoyancy_3d(Float32.(rho_pad))

    # Lamé 参数 + mu 插值
    lam, mu_txx, mu_txy, mu_txz, mu_tyz, lam_2mu = _compute_elastic_params_3d(
        Float32.(vp_pad), Float32.(vs_pad), Float32.(rho_pad))

    return Medium3D(
        nx, ny, nz, Float32(dh), x_max, y_max, z_max, M, pad,
        to_device(lam), to_device(mu_txx),
        to_device(mu_txy), to_device(mu_txz), to_device(mu_tyz),
        to_device(buoy_vx), to_device(buoy_vy), to_device(buoy_vz),
        to_device(lam_2mu)
    )
end

"""
3D弹性波参数计算：lam, mu_txx, mu_txy, mu_txz, mu_tyz, lam_2mu

mu在剪切应力位置需要调和平均：
  - mu_txy: 在 (i,j) 和 (i+1,j+1) 之间，xy平面4点调和平均
  - mu_txz: 在 (i,k) 和 (i+1,k+1) 之间，xz平面4点调和平均
  - mu_tyz: 在 (j,k) 和 (j+1,k+1) 之间，yz平面4点调和平均
"""
function _compute_elastic_params_3d(vp, vs, rho)
    nx, ny, nz = size(vp)

    mu      = Float32.(rho .* vs .^ 2)
    lam     = Float32.(rho .* vp .^ 2 .- 2.0f0 .* mu)
    lam_2mu = Float32.(lam .+ 2.0f0 .* mu)

    # mu_txy: xy平面调和平均 (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    mu_txy = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny-1, i in 1:nx-1
        m1 = mu[i, j, k]; m2 = mu[i+1, j, k]
        m3 = mu[i, j+1, k]; m4 = mu[i+1, j+1, k]
        if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
            mu_txy[i, j, k] = 0.0f0
        else
            mu_txy[i, j, k] = 4.0f0 / (1.0f0/m1 + 1.0f0/m2 + 1.0f0/m3 + 1.0f0/m4)
        end
    end
    mu_txy[nx, :, :] .= mu_txy[nx-1, :, :]
    mu_txy[:, ny, :] .= mu_txy[:, ny-1, :]

    # mu_txz: xz平面调和平均 (i,k), (i+1,k), (i,k+1), (i+1,k+1)
    mu_txz = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz-1, j in 1:ny, i in 1:nx-1
        m1 = mu[i, j, k]; m2 = mu[i+1, j, k]
        m3 = mu[i, j, k+1]; m4 = mu[i+1, j, k+1]
        if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
            mu_txz[i, j, k] = 0.0f0
        else
            mu_txz[i, j, k] = 4.0f0 / (1.0f0/m1 + 1.0f0/m2 + 1.0f0/m3 + 1.0f0/m4)
        end
    end
    mu_txz[nx, :, :] .= mu_txz[nx-1, :, :]
    mu_txz[:, :, nz] .= mu_txz[:, :, nz-1]

    # mu_tyz: yz平面调和平均 (j,k), (j+1,k), (j,k+1), (j+1,k+1)
    mu_tyz = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz-1, j in 1:ny-1, i in 1:nx
        m1 = mu[i, j, k]; m2 = mu[i, j+1, k]
        m3 = mu[i, j, k+1]; m4 = mu[i, j+1, k+1]
        if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
            mu_tyz[i, j, k] = 0.0f0
        else
            mu_tyz[i, j, k] = 4.0f0 / (1.0f0/m1 + 1.0f0/m2 + 1.0f0/m3 + 1.0f0/m4)
        end
    end
    mu_tyz[:, ny, :] .= mu_tyz[:, ny-1, :]
    mu_tyz[:, :, nz] .= mu_tyz[:, :, nz-1]

    return lam, mu, mu_txy, mu_txz, mu_tyz, lam_2mu
end
