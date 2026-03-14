using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using StaticArrays

"""
    update_velocity_kernel!(vx, vz, txx, tzz, txz, bx, bz, a, dtx, dtz, M)

Velocity update kernel function (双语说明/Bilingual).
Lower-level velocity update kernel function. 底层速度更新核函数。
Using @parallel_indices without any if branches, mapping to internal grid via index shifting.
使用 @parallel_indices 且无任何 if 分支，通过索引平移映射到内部网格。
"""
@parallel_indices (ix, iy) function update_velocity_kernel!(
    vx, vz, txx, tzz, txz,
    bx, bz,
    a, dtx, dtz, M
)
    i = ix + M
    j = iy + M

    dtxxdx = 0.0f0
    dtxzdz = 0.0f0
    dtxzdx = 0.0f0
    dtzzdz = 0.0f0

    for l in 1:length(a)
        c = a[l]
        dtxxdx += c * (txx[i+l-1, j] - txx[i-l, j])
        dtxzdz += c * (txz[i, j+l-1] - txz[i, j-l])
        dtxzdx += c * (txz[i+l, j] - txz[i-l+1, j])
        dtzzdz += c * (tzz[i, j+l] - tzz[i, j-l+1])
    end

    bx_ij = bx[i, j]
    bz_ij = bz[i, j]

    vx[i, j] += bx_ij * (dtx * dtxxdx + dtz * dtxzdz)
    vz[i, j] += bz_ij * (dtx * dtxzdx + dtz * dtzzdz)

    return nothing
end

"""
    update_velocity!(W, M_med, a_static::SVector{M, Float32}, p, inner_nx::Int, inner_nz::Int) where {M}

Velocity update API (双语说明/Bilingual).
Top-level velocity update API. 顶层速度更新 API。
Note: The input a_static must already be of SVector type, inner_nx/nz pre-calculated outside the loop.
注意：传入的 a_static 必须已经是 SVector 类型，inner_nx/nz 在循环外预先计算。
"""
function update_velocity!(W, M_med, a_static::SVector{M,Float32}, p, inner_nx::Int, inner_nz::Int) where {M}
    @parallel (1:inner_nx, 1:inner_nz) update_velocity_kernel!(
        W.vx, W.vz, W.txx, W.tzz, W.txz,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, p.dtx, p.dtz, M
    )

    return nothing
end