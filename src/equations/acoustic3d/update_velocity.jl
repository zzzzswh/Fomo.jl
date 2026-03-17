# ═══════════════════════════════════════════════════════════════════════════════
# src/equations/acoustic3d/update_velocity.jl
#
# 3D声波速度更新（从2D扩展）：
#   vx += (1/ρ) · dt/dh · ∂p/∂x
#   vy += (1/ρ) · dt/dh · ∂p/∂y
#   vz += (1/ρ) · dt/dh · ∂p/∂z
#
# 差分模板与2D完全一致，只是增加了 y 方向
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

function _update_velocity_acoustic3d_cuda!(
    vx, vy, vz, p,
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

        dpdx = 0.0f0
        dpdy = 0.0f0
        dpdz = 0.0f0

        @inbounds for l in 1:N
            c = a[l]
            # ∂p/∂x：与2D dtxxdx 同模板
            dpdx += c * (p[i+l-1, j, k] - p[i-l, j, k])
            # ∂p/∂y：新增 y 方向，模板与 ∂p/∂x 类似
            dpdy += c * (p[i, j+l-1, k] - p[i, j-l, k])
            # ∂p/∂z：与2D dtzzdz 同模板
            dpdz += c * (p[i, j, k+l] - p[i, j, k-l+1])
        end

        @inbounds begin
            vx[i, j, k] += bx[i, j, k] * dtdh * dpdx
            vy[i, j, k] += by[i, j, k] * dtdh * dpdy
            vz[i, j, k] += bz[i, j, k] * dtdh * dpdz
        end
    end
    return nothing
end

function update_velocity_acoustic3d!(W::AcousticWavefield3D, M_med::AcousticMedium3D,
    a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_ny::Int, inner_nz::Int) where {N}
    dtdh = Float32(dt / M_med.dh)
    M32  = Int32(M_med.M)
    inx  = Int32(inner_nx)
    iny  = Int32(inner_ny)
    inz  = Int32(inner_nz)

    threads = (8, 8, 4)
    blocks  = (cld(inner_nx, 8), cld(inner_ny, 8), cld(inner_nz, 4))

    @cuda threads=threads blocks=blocks _update_velocity_acoustic3d_cuda!(
        W.vx, W.vy, W.vz, W.p,
        M_med.buoy_vx, M_med.buoy_vy, M_med.buoy_vz,
        a_static, dtdh, M32, inx, iny, inz)
    return nothing
end
