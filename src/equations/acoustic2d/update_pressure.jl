# ═══════════════════════════════════════════════════════════════════════════════
# [NEW] src/equations/acoustic2d/update_pressure.jl
#
# 声波压力更新：
#   p += κ · dt/dh · (∂vx/∂x + ∂vz/∂z)
#
# 差分模板与弹性波 stress 中的 dvxdx, dvzdz 完全一致
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

function _update_pressure_acoustic_cuda!(
    p, vx, vz,
    kappa,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32,
    inner_nx::Int32, inner_nz::Int32
) where {N}
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= inner_nx && iy <= inner_nz
        i = ix + M
        j = iy + M

        dvxdx = 0.0f0
        dvzdz = 0.0f0

        @inbounds for l in 1:N
            c = a[l]
            dvxdx += c * (vx[i+l, j] - vx[i-l+1, j])   # 与弹性波同模板
            dvzdz += c * (vz[i, j+l-1] - vz[i, j-l])      # 与弹性波同模板
        end

        @inbounds begin
            p[i, j] += kappa[i, j] * (dvxdx * dtx + dvzdz * dtz)
        end
    end
    return nothing
end

function update_pressure_acoustic!(W::AcousticWavefield, M_med::AcousticMedium,
    a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_nz::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    M32 = Int32(M_med.M)
    inx = Int32(inner_nx)
    inz = Int32(inner_nz)

    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads = threads blocks = blocks _update_pressure_acoustic_cuda!(
        W.p, W.vx, W.vz,
        M_med.kappa,
        a_static, dtx, dtz, M32, inx, inz
    )
    return nothing
end