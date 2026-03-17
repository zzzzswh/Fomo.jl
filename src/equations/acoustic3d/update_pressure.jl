# ═══════════════════════════════════════════════════════════════════════════════
# src/equations/acoustic3d/update_pressure.jl
#
# 3D声波压力更新（从2D扩展）：
#   p += κ · dt/dh · (∂vx/∂x + ∂vy/∂y + ∂vz/∂z)
#
# 差分模板与2D完全一致，增加了 ∂vy/∂y 项
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

function _update_pressure_acoustic3d_cuda!(
    p, vx, vy, vz,
    kappa,
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

        dvxdx = 0.0f0
        dvydy = 0.0f0
        dvzdz = 0.0f0

        @inbounds for l in 1:N
            c = a[l]
            # ∂vx/∂x：与2D同模板
            dvxdx += c * (vx[i+l, j, k] - vx[i-l+1, j, k])
            # ∂vy/∂y：新增 y 方向
            dvydy += c * (vy[i, j+l, k] - vy[i, j-l+1, k])
            # ∂vz/∂z：与2D同模板
            dvzdz += c * (vz[i, j, k+l-1] - vz[i, j, k-l])
        end

        @inbounds begin
            p[i, j, k] += kappa[i, j, k] * dtdh * (dvxdx + dvydy + dvzdz)
        end
    end
    return nothing
end

function update_pressure_acoustic3d!(W::AcousticWavefield3D, M_med::AcousticMedium3D,
    a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_ny::Int, inner_nz::Int) where {N}
    dtdh = Float32(dt / M_med.dh)
    M32  = Int32(M_med.M)
    inx  = Int32(inner_nx)
    iny  = Int32(inner_ny)
    inz  = Int32(inner_nz)

    threads = (8, 8, 4)
    blocks  = (cld(inner_nx, 8), cld(inner_ny, 8), cld(inner_nz, 4))

    @cuda threads=threads blocks=blocks _update_pressure_acoustic3d_cuda!(
        W.p, W.vx, W.vy, W.vz,
        M_med.kappa,
        a_static, dtdh, M32, inx, iny, inz)
    return nothing
end
