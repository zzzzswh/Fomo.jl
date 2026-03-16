# src/utils/pad_array.jl

"""
    _pad_array(data, pad) -> Matrix{Float32}

对 2D 数组进行边界扩展（边界值外推）。所有方程的介质初始化共用。
"""
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

"""
    _compute_staggered_buoyancy(rho) -> (buoy_vx, buoy_vz)

计算交错网格浮力（1/ρ），vx 和 vz 位置使用相邻点调和平均。
真空区域（ρ=0）自动处理。所有方程共用。
"""
function _compute_staggered_buoyancy(rho::Matrix{Float32})
    nx, nz = size(rho)

    buoy_vx = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz, i in 1:nx-1
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
    buoy_vx[nx, :] .= buoy_vx[nx-1, :]

    buoy_vz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1, i in 1:nx
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
    buoy_vz[:, nz] .= buoy_vz[:, nz-1]

    return buoy_vx, buoy_vz
end