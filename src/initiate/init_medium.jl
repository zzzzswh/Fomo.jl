# src/initiate/init_medium.jl
"""
    init_medium(vp, vs, rho, dh, nbc, fd_order)

Initialize Medium with material properties (双语说明/Bilingual):
Initialize Medium with material properties. 使用物理参数初始化介质。
"""
function init_medium(vp::Matrix, vs::Matrix, rho::Matrix,
    dh::Real, nbc::Int, fd_order::Int)

    M = fd_order ÷ 2
    pad = nbc + M

    # 直接使用原始输入的维度，不再进行转置
    nx_inner, nz_inner = size(vp)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    # 直接对外扩函数传入原始矩阵
    vp_pad = _pad_array(vp, pad)
    vs_pad = _pad_array(vs, pad)
    rho_pad = _pad_array(rho, pad)

    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = _compute_staggered_params_optimized(vp_pad, vs_pad, rho_pad)

    return Medium(
        nx, nz, Float32(dh), x_max, z_max,
        M, pad,
        to_device(lam),
        to_device(mu_txx),
        to_device(mu_txz),
        to_device(buoy_vx),
        to_device(buoy_vz),
        to_device(lam_2mu)
    )
end

function _pad_array(data::Matrix, pad::Int)
    nx, nz = size(data)
    result = zeros(Float32, nx + 2 * pad, nz + 2 * pad)
    result[pad+1:pad+nx, pad+1:pad+nz] .= Float32.(data)

    for i in 1:pad
        result[i, :] .= result[pad+1, :]
        result[end-i+1, :] .= result[end-pad, :]
    end
    for j in 1:pad
        result[:, j] .= result[:, pad+1]
        result[:, end-j+1] .= result[:, end-pad]
    end
    return result
end

function _compute_staggered_params_optimized(vp, vs, rho)
    nx, nz = size(vp)

    mu = Float32.(rho .* vs .^ 2)
    lam = Float32.(rho .* vp .^ 2 .- 2.0f0 .* mu)
    lam_2mu = Float32.(lam .+ 2.0f0 .* mu)

    buoy_vx = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz
        for i in 1:nx-1
            rho1, rho2 = rho[i, j], rho[i+1, j]
            if rho1 == 0.0f0 && rho2 == 0.0f0
                buoy_vx[i, j] = 0.0f0
            elseif rho1 == 0.0f0
                buoy_vx[i, j] = 2.0f0 / rho2
            elseif rho2 == 0.0f0
                buoy_vx[i, j] = 2.0f0 / rho1
            else
                buoy_vx[i, j] = 2.0f0 / (rho1 + rho2)
            end
        end
    end
    buoy_vx[nx, :] .= buoy_vx[nx-1, :]

    buoy_vz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx
            rho1, rho2 = rho[i, j], rho[i, j+1]
            if rho1 == 0.0f0 && rho2 == 0.0f0
                buoy_vz[i, j] = 0.0f0
            elseif rho1 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / rho2
            elseif rho2 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / rho1
            else
                buoy_vz[i, j] = 2.0f0 / (rho1 + rho2)
            end
        end
    end
    buoy_vz[:, nz] .= buoy_vz[:, nz-1]

    mu_txz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx-1
            m1, m2, m3, m4 = mu[i, j], mu[i+1, j], mu[i, j+1], mu[i+1, j+1]
            if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
                mu_txz[i, j] = 0.0f0
            else
                mu_txz[i, j] = 4.0f0 / (1.0f0 / m1 + 1.0f0 / m2 + 1.0f0 / m3 + 1.0f0 / m4)
            end
        end
    end
    mu_txz[nx, :] .= mu_txz[nx-1, :]
    mu_txz[:, nz] .= mu_txz[:, nz-1]

    return lam, mu, mu_txz, buoy_vx, buoy_vz, lam_2mu
end


