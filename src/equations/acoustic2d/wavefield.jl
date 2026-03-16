# src/equations/acoustic2d/wavefield.jl

using CUDA

mutable struct AcousticWavefield{T}
    vx::T
    vz::T
    p::T
    vx_old::T
    vz_old::T
    p_old::T
end

function AcousticWavefield(nx::Int, nz::Int, pad::Int)
    nx += 2 * pad
    nz += 2 * pad
    return AcousticWavefield(
        CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz)
    )
end

function reset!(W::AcousticWavefield)
    for f in (W.vx, W.vz, W.p, W.vx_old, W.vz_old, W.p_old)
        fill!(f, 0.0f0)
    end
end
