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

    # vx 位于 (i-1/2, j)（由差分模板 dpdx = p[i]-p[i-1] 确定）：
    # 用 rho[i-1,j] 与 rho[i,j] 的调和平均
    buoy_vx = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz, i in 2:nx
        rho1, rho2 = rho[i-1, j], rho[i, j]
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
    buoy_vx[1, :] .= buoy_vx[2, :]

    # vz 位于 (i, j+1/2)（由 dpdz = p[j+1]-p[j] 确定）：rho[i,j] 与 rho[i,j+1]（原实现即正确）
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