# ==============================================================================
# Wavefield - Parametric over array type
# ==============================================================================

using CUDA

"""
    Wavefield{T}

Velocity and stress components. Works for both CPU and GPU.
`T` is the array type (Array{Float32,2} or CuArray{Float32,2}).
"""
mutable struct Wavefield{T<:AbstractMatrix{Float32}}
    # Current time step
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T

    # Previous time step (for HABC)
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

"""
    Wavefield(nx, nz)

Create zero-initialized wavefield on GPU.
"""
function Wavefield(nx::Int, nz::Int, pad::Int)
    nx += 2 * pad
    nz += 2 * pad
    return Wavefield(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )
end
