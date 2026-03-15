# src/kernels/elastic2d/update_stress.jl 
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using StaticArrays

"""
    update_stress_kernel!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz, a, dtx, dtz, M)

底层应力更新核函数 (Kernel)。
使用 @parallel_indices 且无任何 if 分支，通过索引平移映射到内部网格。
"""
@parallel_indices (ix, iy) function update_stress_kernel!(
    txx, tzz, txz, vx, vz,
    lam, lam_2mu, mu_txz,
    a, dtx, dtz, M
)
    # 加上偏移量 M 直接映射到物理网格的内部区域
    i = ix + M
    j = iy + M

    dvxdx = 0.0f0
    dvzdz = 0.0f0
    dvxdz = 0.0f0
    dvzdx = 0.0f0

    # a 是 SVector，循环在编译期会被完全铺平
    for l in 1:length(a)
        c = a[l]
        dvxdx += c * (vx[i+l, j] - vx[i-l+1, j])
        dvzdz += c * (vz[i, j+l-1] - vz[i, j-l])
        dvxdz += c * (vx[i, j+l] - vx[i, j-l+1])
        dvzdx += c * (vz[i+l-1, j] - vz[i-l, j])
    end

    l_val = lam[i, j]
    l2m_val = lam_2mu[i, j]

    txx[i, j] += l2m_val * dvxdx * dtx + l_val * dvzdz * dtz
    tzz[i, j] += l_val * dvxdx * dtx + l2m_val * dvzdz * dtz
    txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)

    return nothing
end

function update_stress!(W, M_med, a_static::SVector{M,Float32}, dt, inner_nx::Int, inner_nz::Int) where {M}
    # 此时，编译器通过 where {M} 明确知道了阶数 M！
    # 没有任何运行时开销，直接全速启动 GPU 内核。

    # 动态计算 dtx 和 dtz (直接从传入的 dt 和 M_med.dh 计算)
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)

    @parallel (1:inner_nx, 1:inner_nz) update_stress_kernel!(
        W.txx, W.tzz, W.txz, W.vx, W.vz,
        M_med.lam, M_med.lam_2mu, M_med.mu_txz,
        a_static, dtx, dtz, M
    )

    return nothing
end