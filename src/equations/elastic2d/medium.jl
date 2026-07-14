# src/equations/elastic2d/medium.jl

struct Medium{T}
    nx::Int
    nz::Int
    dh::Float32
    x_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    lam::T
    mu_txx::T
    mu_txz::T
    buoy_vx::T
    buoy_vz::T
    lam_2mu::T
end

function init_medium(vp::Matrix, vs::Matrix, rho::Matrix,
    dh::Real, nbc::Int, fd_order::Int)
    M = fd_order ÷ 2
    pad = nbc + M

    nx_inner, nz_inner = size(vp)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    vp_pad = _pad_array(vp, pad)       # ← 调用共享版本
    vs_pad = _pad_array(vs, pad)
    rho_pad = _pad_array(rho, pad)

    # 浮力：调用共享函数
    buoy_vx, buoy_vz = _compute_staggered_buoyancy(Float32.(rho_pad))

    # 弹性波独有：Lamé 参数 + mu 插值
    lam, mu_txx, mu_txz, lam_2mu = _compute_elastic_params(
        Float32.(vp_pad), Float32.(vs_pad), Float32.(rho_pad))

    return Medium(
        nx, nz, Float32(dh), x_max, z_max, M, pad,
        to_device(lam), to_device(mu_txx), to_device(mu_txz),
        to_device(buoy_vx), to_device(buoy_vz), to_device(lam_2mu)
    )
end

"""
弹性波独有的参数计算：lam, mu_txx, mu_txz, lam_2mu
"""
function _compute_elastic_params(vp, vs, rho)
    nx, nz = size(vp)

    mu = Float32.(rho .* vs .^ 2)
    lam = Float32.(rho .* vp .^ 2 .- 2.0f0 .* mu)
    lam_2mu = Float32.(lam .+ 2.0f0 .* mu)

    # mu 在 txz 位置 (i-1/2, j+1/2) 的调和平均
    # （txz 位置由模板确定：dvxdz = vx[j+1]-vx[j]、dvzdx = vz[i]-vz[i-1] 均落在 (i-1/2, j+1/2)）
    mu_txz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1, i in 2:nx
        m1, m2, m3, m4 = mu[i-1, j], mu[i, j], mu[i-1, j+1], mu[i, j+1]
        if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
            mu_txz[i, j] = 0.0f0
        else
            mu_txz[i, j] = 4.0f0 / (1.0f0 / m1 + 1.0f0 / m2 + 1.0f0 / m3 + 1.0f0 / m4)
        end
    end
    mu_txz[1, :] .= mu_txz[2, :]
    mu_txz[:, nz] .= mu_txz[:, nz-1]

    return lam, mu, mu_txz, lam_2mu
end