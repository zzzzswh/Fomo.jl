# src/equations/elastic3d/update_velocity.jl
#
# 3D弹性波速度更新（从2D扩展）
#
# 2D方程：
#   vx += b_x * (∂txx/∂x + ∂txz/∂z)
#   vz += b_z * (∂txz/∂x + ∂tzz/∂z)
#
# 3D方程（增加 y 方向和 vy 分量）：
#   vx += b_x * (∂txx/∂x + ∂txy/∂y + ∂txz/∂z)
#   vy += b_y * (∂txy/∂x + ∂tyy/∂y + ∂tyz/∂z)
#   vz += b_z * (∂txz/∂x + ∂tyz/∂y + ∂tzz/∂z)

using CUDA
using StaticArrays

function _update_velocity_3d_cuda!(
    vx, vy, vz,
    txx, tyy, tzz, txy, txz, tyz,
    bx, by, bz,
    a::SVector{N,Float32},
    dtdh::Float32, M::Int32,
    inner_nx::Int32, inner_ny::Int32, inner_nz::Int32
) where {N}
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if ix <= inner_nx && iy <= inner_ny && iz <= inner_nz
        i = ix + M
        j = iy + M
        k = iz + M

        # ── vx 所需的导数 ──
        dtxxdx = 0.0f0
        dtxydy = 0.0f0
        dtxzdz = 0.0f0

        # ── vy 所需的导数 ──
        dtxydx = 0.0f0
        dtyydy = 0.0f0
        dtyzdz = 0.0f0

        # ── vz 所需的导数 ──
        dtxzdx = 0.0f0
        dtyzdy = 0.0f0
        dtzzdz = 0.0f0

        @inbounds for l in 1:N
            c = a[l]

            # ── 正应力对各方向的导数（forward stencil）──
            # ∂txx/∂x (与2D同)
            dtxxdx += c * (txx[i+l-1, j, k] - txx[i-l, j, k])
            # ∂tyy/∂y (新增)
            dtyydy += c * (tyy[i, j+l-1, k] - tyy[i, j-l, k])
            # ∂tzz/∂z (与2D同)
            dtzzdz += c * (tzz[i, j, k+l] - tzz[i, j, k-l+1])

            # ── 剪切应力对各方向的导数 ──
            # ∂txy/∂y：txy 位于 (i-1/2, j-1/2, k)，txy[i,j+1]-txy[i,j] 中心恰在 vx 的 y=j
            dtxydy += c * (txy[i, j+l, k] - txy[i, j-l+1, k])
            # ∂txy/∂x：txy[i+1,j]-txy[i,j] 中心在 x=i，恰为 vy 的 x 位置
            dtxydx += c * (txy[i+l, j, k] - txy[i-l+1, j, k])

            # ∂txz/∂z (txz在(i+1/2, k+1/2)位置，与2D同)
            dtxzdz += c * (txz[i, j, k+l-1] - txz[i, j, k-l])
            # ∂txz/∂x (与2D同)
            dtxzdx += c * (txz[i+l, j, k] - txz[i-l+1, j, k])

            # ∂tyz/∂z (tyz在(j+1/2, k+1/2)位置)
            dtyzdz += c * (tyz[i, j, k+l-1] - tyz[i, j, k-l])
            # ∂tyz/∂y
            dtyzdy += c * (tyz[i, j+l, k] - tyz[i, j-l+1, k])
        end

        @inbounds begin
            vx[i, j, k] += bx[i, j, k] * dtdh * (dtxxdx + dtxydy + dtxzdz)
            vy[i, j, k] += by[i, j, k] * dtdh * (dtxydx + dtyydy + dtyzdz)
            vz[i, j, k] += bz[i, j, k] * dtdh * (dtxzdx + dtyzdy + dtzzdz)
        end
    end
    return nothing
end

function update_velocity_3d!(W, M_med, a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_ny::Int, inner_nz::Int) where {N}
    dtdh = Float32(dt / M_med.dh)
    M32  = Int32(M_med.M)
    inx  = Int32(inner_nx)
    iny  = Int32(inner_ny)
    inz  = Int32(inner_nz)

    threads = (8, 8, 4)
    blocks  = (cld(inner_nx, 8), cld(inner_ny, 8), cld(inner_nz, 4))

    @cuda threads=threads blocks=blocks _update_velocity_3d_cuda!(
        W.vx, W.vy, W.vz,
        W.txx, W.tyy, W.tzz, W.txy, W.txz, W.tyz,
        M_med.buoy_vx, M_med.buoy_vy, M_med.buoy_vz,
        a_static, dtdh, M32, inx, iny, inz)
    return nothing
end
