# src/equations/elastic2d/wavefield.jl

using CUDA

mutable struct Wavefield{T}
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

function Wavefield(nx::Int, nz::Int, pad::Int)
    nx += 2 * pad
    nz += 2 * pad
    return Wavefield(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )
end

function reset!(W::Wavefield)
    for f in (W.vx, W.vz, W.txx, W.tzz, W.txz,
        W.vx_old, W.vz_old, W.txx_old, W.tzz_old, W.txz_old)
        fill!(f, 0.0f0)
    end
end