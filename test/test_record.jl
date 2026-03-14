using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots
using Printf
using StaticArrays
using CUDA



@init_parallel_stencil(CUDA, Float32, 2)

# Include the package source
include("../src/Fomo_gpu.jl")

# ==============================================================================
# 1. Setup Simulation Parameters
# ==============================================================================
nx = 200
nz = 200
dh = 10.0f0
nt = 1000
dt = 0.001f0
fd_order = 4
nbc = 50

# Create SimParams
p = SimParams(dt, nt, dh, fd_order)

# ==============================================================================
# 2. Setup Medium (Two Layers)
# ==============================================================================
vp = 2000.0f0 .* ones(Float32, nx, nz)
vs = 1000.0f0 .* ones(Float32, nx, nz)
rho = 2000.0f0 .* ones(Float32, nx, nz)

# Second layer (bottom half)
vp[100:end, :] .= 3000.0f0
vs[100:end, :] .= 1500.0f0
rho[100:end, :] .= 2500.0f0

println("Initializing ...")

M_med = init_medium(vp, vs, rho, dh, nbc, fd_order)

H = init_habc(M_med.nx, M_med.nz, nbc, M_med.pad, p.dt, dh, dh, 2000.0f0)