# src/equations/acoustic3d/wavefield.jl
#
# 3D声波波场：从2D扩展，增加 vy, vy_old

using CUDA

mutable struct AcousticWavefield3D{T}
    vx::T
    vy::T
    vz::T
    p::T
    vx_old::T
    vy_old::T
    vz_old::T
    p_old::T
end

function AcousticWavefield3D(nx::Int, ny::Int, nz::Int, pad::Int)
    nx += 2pad
    ny += 2pad
    nz += 2pad
    return AcousticWavefield3D(
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz),
        CUDA.zeros(Float32, nx, ny, nz)
    )
end

function reset!(W::AcousticWavefield3D)
    for f in (W.vx, W.vy, W.vz, W.p, W.vx_old, W.vy_old, W.vz_old, W.p_old)
        fill!(f, 0.0f0)
    end
end
