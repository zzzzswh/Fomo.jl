# src/equations/elastic3d/wavefield.jl
#
# 3D弹性波波场（从2D扩展）：
#   速度分量: vx, vy, vz （2D只有 vx, vz）
#   应力分量: txx, tyy, tzz, txy, txz, tyz （2D只有 txx, tzz, txz）

using CUDA

mutable struct Wavefield3D{T}
    vx::T
    vy::T
    vz::T
    txx::T
    tyy::T
    tzz::T
    txy::T
    txz::T
    tyz::T
    vx_old::T
    vy_old::T
    vz_old::T
    txx_old::T
    tyy_old::T
    tzz_old::T
    txy_old::T
    txz_old::T
    tyz_old::T
end

function Wavefield3D(nx::Int, ny::Int, nz::Int, pad::Int)
    nx += 2pad; ny += 2pad; nz += 2pad
    z() = CUDA.zeros(Float32, nx, ny, nz)
    return Wavefield3D(
        z(), z(), z(),           # vx, vy, vz
        z(), z(), z(),           # txx, tyy, tzz
        z(), z(), z(),           # txy, txz, tyz
        z(), z(), z(),           # vx_old, vy_old, vz_old
        z(), z(), z(),           # txx_old, tyy_old, tzz_old
        z(), z(), z()            # txy_old, txz_old, tyz_old
    )
end

function reset!(W::Wavefield3D)
    for f in (W.vx, W.vy, W.vz,
              W.txx, W.tyy, W.tzz, W.txy, W.txz, W.tyz,
              W.vx_old, W.vy_old, W.vz_old,
              W.txx_old, W.tyy_old, W.tzz_old, W.txy_old, W.txz_old, W.tyz_old)
        fill!(f, 0.0f0)
    end
end
