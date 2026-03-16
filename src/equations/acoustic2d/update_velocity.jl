# ═══════════════════════════════════════════════════════════════════════════════
# [NEW] src/equations/acoustic2d/update_velocity.jl
#
# 声波速度更新：
#   vx += (1/ρ) · dt/dh · ∂p/∂x
#   vz += (1/ρ) · dt/dh · ∂p/∂z
#
# 差分模板与弹性波完全一致（p 在 txx/tzz 同一网格位置）
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

function _update_velocity_acoustic_cuda!(
    vx, vz, p,
    bx, bz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32,
    inner_nx::Int32, inner_nz::Int32
) where {N}
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= inner_nx && iy <= inner_nz
        i = ix + M
        j = iy + M

        dpdx = 0.0f0
        dpdz = 0.0f0

        @inbounds for l in 1:N
            c = a[l]
            dpdx += c * (p[i+l-1, j] - p[i-l, j])     # 与弹性波 dtxxdx 同模板
            dpdz += c * (p[i, j+l] - p[i, j-l+1])      # 与弹性波 dtzzdz 同模板
        end

        @inbounds begin
            vx[i, j] += bx[i, j] * dtx * dpdx
            vz[i, j] += bz[i, j] * dtz * dpdz
        end
    end
    return nothing
end

function update_velocity_acoustic!(W::AcousticWavefield, M_med::AcousticMedium,
    a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_nz::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    M32 = Int32(M_med.M)
    inx = Int32(inner_nx)
    inz = Int32(inner_nz)

    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads = threads blocks = blocks _update_velocity_acoustic_cuda!(
        W.vx, W.vz, W.p,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, inx, inz
    )
    return nothing
end