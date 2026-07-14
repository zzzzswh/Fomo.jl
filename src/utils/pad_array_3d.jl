# src/utils/pad_array_3d.jl

"""
    _pad_array_3d(data, pad) -> Array{Float32,3}

对 3D 数组进行边界扩展（边界值外推）。所有3D方程的介质初始化共用。
"""
function _pad_array_3d(data::Array{<:Real,3}, pad::Int)
    nx, ny, nz = size(data)
    result = zeros(Float32, nx + 2pad, ny + 2pad, nz + 2pad)
    result[pad+1:pad+nx, pad+1:pad+ny, pad+1:pad+nz] .= Float32.(data)

    # x 方向外推
    for i in 1:pad
        result[i, :, :] .= result[pad+1, :, :]
        result[end-i+1, :, :] .= result[end-pad, :, :]
    end
    # y 方向外推
    for j in 1:pad
        result[:, j, :] .= result[:, pad+1, :]
        result[:, end-j+1, :] .= result[:, end-pad, :]
    end
    # z 方向外推
    for k in 1:pad
        result[:, :, k] .= result[:, :, pad+1]
        result[:, :, end-k+1] .= result[:, :, end-pad]
    end
    return result
end

"""
    _compute_staggered_buoyancy_3d(rho) -> (buoy_vx, buoy_vy, buoy_vz)

计算3D交错网格浮力（1/ρ），vx, vy, vz 位置使用相邻点调和平均。
真空区域（ρ=0）自动处理。所有3D方程共用。
"""
function _compute_staggered_buoyancy_3d(rho::Array{Float32,3})
    nx, ny, nz = size(rho)

    # buoy_vx: vx 位于 (i-1/2, j, k)（由 dpdx = p[i]-p[i-1] 确定），x方向交错 (i-1, i)
    buoy_vx = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        rho1, rho2 = rho[i-1, j, k], rho[i, j, k]
        if rho1 == 0.0f0 && rho2 == 0.0f0
            buoy_vx[i, j, k] = 0.0f0
        elseif rho1 == 0.0f0
            buoy_vx[i, j, k] = 2.0f0 / rho2
        elseif rho2 == 0.0f0
            buoy_vx[i, j, k] = 2.0f0 / rho1
        else
            buoy_vx[i, j, k] = 2.0f0 / (rho1 + rho2)
        end
    end
    buoy_vx[1, :, :] .= buoy_vx[2, :, :]

    # buoy_vy: vy 位于 (i, j-1/2, k)（由 dpdy = p[j]-p[j-1] 确定），y方向交错 (j-1, j)
    buoy_vy = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        rho1, rho2 = rho[i, j-1, k], rho[i, j, k]
        if rho1 == 0.0f0 && rho2 == 0.0f0
            buoy_vy[i, j, k] = 0.0f0
        elseif rho1 == 0.0f0
            buoy_vy[i, j, k] = 2.0f0 / rho2
        elseif rho2 == 0.0f0
            buoy_vy[i, j, k] = 2.0f0 / rho1
        else
            buoy_vy[i, j, k] = 2.0f0 / (rho1 + rho2)
        end
    end
    buoy_vy[:, 1, :] .= buoy_vy[:, 2, :]

    # buoy_vz: vz 位于 (i, j, k+1/2)（由 dpdz = p[k+1]-p[k] 确定），z方向交错 (k, k+1)（原实现即正确）
    buoy_vz = zeros(Float32, nx, ny, nz)
    @inbounds for k in 1:nz-1, j in 1:ny, i in 1:nx
        rho1, rho2 = rho[i, j, k], rho[i, j, k+1]
        if rho1 == 0.0f0 && rho2 == 0.0f0
            buoy_vz[i, j, k] = 0.0f0
        elseif rho1 == 0.0f0
            buoy_vz[i, j, k] = 2.0f0 / rho2
        elseif rho2 == 0.0f0
            buoy_vz[i, j, k] = 2.0f0 / rho1
        else
            buoy_vz[i, j, k] = 2.0f0 / (rho1 + rho2)
        end
    end
    buoy_vz[:, :, nz] .= buoy_vz[:, :, nz-1]

    return buoy_vx, buoy_vy, buoy_vz
end
