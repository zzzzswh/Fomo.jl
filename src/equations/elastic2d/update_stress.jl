# src/equations/elastic2d/update_stress.jl

using CUDA
using StaticArrays

# ==============================================================================
# 手写 CUDA kernel —— 消除 @parallel 宏的堆分配
# ==============================================================================

function _update_stress_cuda!(
    txx, tzz, txz, vx, vz,
    lam, lam_2mu, mu_txz,
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
        dvxdz = 0.0f0
        dvzdx = 0.0f0

        @inbounds for l in 1:N
            c = a[l]
            dvxdx += c * (vx[i+l, j] - vx[i-l+1, j])
            dvzdz += c * (vz[i, j+l-1] - vz[i, j-l])
            dvxdz += c * (vx[i, j+l] - vx[i, j-l+1])
            dvzdx += c * (vz[i+l-1, j] - vz[i-l, j])
        end

        @inbounds begin
            l_val = lam[i, j]
            l2m_val = lam_2mu[i, j]

            txx[i, j] += l2m_val * dvxdx * dtx + l_val * dvzdz * dtz
            tzz[i, j] += l_val * dvxdx * dtx + l2m_val * dvzdz * dtz
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
    return nothing
end

function update_stress!(W, M_med, a_static::SVector{N,Float32}, dt, inner_nx::Int, inner_nz::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    M32 = Int32(M_med.M)
    inx = Int32(inner_nx)
    inz = Int32(inner_nz)

    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads = threads blocks = blocks _update_stress_cuda!(
        W.txx, W.tzz, W.txz, W.vx, W.vz,
        M_med.lam, M_med.lam_2mu, M_med.mu_txz,
        a_static, dtx, dtz, M32, inx, inz
    )
    return nothing
end