# src/equations/elastic3d/update_stress.jl
#
# 3D弹性波应力更新（从2D扩展）
#
# 2D方程：
#   txx += (λ+2μ)·∂vx/∂x + λ·∂vz/∂z
#   tzz += λ·∂vx/∂x + (λ+2μ)·∂vz/∂z
#   txz += μ·(∂vx/∂z + ∂vz/∂x)
#
# 3D方程（增加 y 方向和 tyy, txy, tyz）：
#   txx += (λ+2μ)·∂vx/∂x + λ·∂vy/∂y + λ·∂vz/∂z
#   tyy += λ·∂vx/∂x + (λ+2μ)·∂vy/∂y + λ·∂vz/∂z
#   tzz += λ·∂vx/∂x + λ·∂vy/∂y + (λ+2μ)·∂vz/∂z
#   txy += μ_txy·(∂vx/∂y + ∂vy/∂x)
#   txz += μ_txz·(∂vx/∂z + ∂vz/∂x)
#   tyz += μ_tyz·(∂vy/∂z + ∂vz/∂y)

using CUDA
using StaticArrays

function _update_stress_3d_cuda!(
    txx, tyy, tzz, txy, txz, tyz,
    vx, vy, vz,
    lam, lam_2mu, mu_txy, mu_txz, mu_tyz,
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

        # ── 正应力所需的速度导数 ──
        dvxdx = 0.0f0
        dvydy = 0.0f0
        dvzdz = 0.0f0

        # ── 剪切应力所需的速度导数 ──
        dvxdy = 0.0f0
        dvydx = 0.0f0
        dvxdz = 0.0f0
        dvzdx = 0.0f0
        dvydz = 0.0f0
        dvzdy = 0.0f0

        @inbounds for l in 1:N
            c = a[l]

            # ── 正应力位置的导数（与2D dvxdx, dvzdz 同模板）──
            dvxdx += c * (vx[i+l, j, k]   - vx[i-l+1, j, k])
            dvydy += c * (vy[i, j+l, k]   - vy[i, j-l+1, k])
            dvzdz += c * (vz[i, j, k+l-1] - vz[i, j, k-l])

            # ── txy 位置 (i-1/2, j-1/2, k) 的导数 ──
            # ∂vx/∂y：vx 位于 (i-1/2, j)，vx[i,j]-vx[i,j-1] 中心在 (i-1/2, j-1/2)
            dvxdy += c * (vx[i, j+l-1, k] - vx[i, j-l, k])
            # ∂vy/∂x：vy 位于 (i, j-1/2)，vy[i,j]-vy[i-1,j] 中心在 (i-1/2, j-1/2)
            dvydx += c * (vy[i+l-1, j, k] - vy[i-l, j, k])

            # ── txz 位置的导数（与2D dvxdz, dvzdx 同模板）──
            dvxdz += c * (vx[i, j, k+l]   - vx[i, j, k-l+1])
            dvzdx += c * (vz[i+l-1, j, k] - vz[i-l, j, k])

            # ── tyz 位置的导数 ──
            # ∂vy/∂z (vy在(i,j+1/2)，对z用forward stencil)
            dvydz += c * (vy[i, j, k+l]   - vy[i, j, k-l+1])
            # ∂vz/∂y (vz在(i,k+1/2)，对y用backward stencil)
            dvzdy += c * (vz[i, j+l-1, k] - vz[i, j-l, k])
        end

        @inbounds begin
            l_val   = lam[i, j, k]
            l2m_val = lam_2mu[i, j, k]

            div_dtdh = dtdh  # 统一系数

            # 正应力更新
            txx[i, j, k] += (l2m_val * dvxdx + l_val * dvydy + l_val * dvzdz) * div_dtdh
            tyy[i, j, k] += (l_val * dvxdx + l2m_val * dvydy + l_val * dvzdz) * div_dtdh
            tzz[i, j, k] += (l_val * dvxdx + l_val * dvydy + l2m_val * dvzdz) * div_dtdh

            # 剪切应力更新
            txy[i, j, k] += mu_txy[i, j, k] * (dvxdy + dvydx) * div_dtdh
            txz[i, j, k] += mu_txz[i, j, k] * (dvxdz + dvzdx) * div_dtdh
            tyz[i, j, k] += mu_tyz[i, j, k] * (dvydz + dvzdy) * div_dtdh
        end
    end
    return nothing
end

function update_stress_3d!(W, M_med, a_static::SVector{N,Float32}, dt,
    inner_nx::Int, inner_ny::Int, inner_nz::Int) where {N}
    dtdh = Float32(dt / M_med.dh)
    M32  = Int32(M_med.M)
    inx  = Int32(inner_nx)
    iny  = Int32(inner_ny)
    inz  = Int32(inner_nz)

    threads = (8, 8, 4)
    blocks  = (cld(inner_nx, 8), cld(inner_ny, 8), cld(inner_nz, 4))

    @cuda threads=threads blocks=blocks _update_stress_3d_cuda!(
        W.txx, W.tyy, W.tzz, W.txy, W.txz, W.tyz,
        W.vx, W.vy, W.vz,
        M_med.lam, M_med.lam_2mu, M_med.mu_txy, M_med.mu_txz, M_med.mu_tyz,
        a_static, dtdh, M32, inx, iny, inz)
    return nothing
end
