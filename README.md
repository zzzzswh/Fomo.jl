# Fomo.jl

**GPU-accelerated seismic wave equation forward modeling in Julia**

<p align="center">
  <img src="wavefield.gif" alt="Elastic wavefield simulation" width="600">
</p>

Fomo is a CUDA-based 2D seismic forward modeling package that solves acoustic and elastic wave equations on staggered grids using high-order finite differences.

## Features

- **Acoustic & Elastic 2D** — Full support for both acoustic (pressure-velocity) and elastic (stress-velocity) formulations
- **Staggered Grid Finite Differences** — Up to 10th-order spatial accuracy for minimal numerical dispersion
- **Hybrid Absorbing Boundary (HABC)** — Efficient boundary condition based on the method by Liu Yang, combining one-way wave equations with sponge damping
- **Vacuum Free Surface** — Realistic free-surface modeling using the vacuum method (zero velocity/density at the surface)
- **GPU Acceleration** — Built on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for high-performance computation on NVIDIA GPUs
- **Built-in Visualization** — Shot record plotting and wavefield animation export out of the box

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

## Quick Start

### Acoustic 2D

```julia
using CUDA
using Fomo

# Grid parameters
nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# Velocity model
vp  = fill(2500.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# Vacuum free surface (top row = 0)
vp[:, 1]  .= 0.0f0
rho[:, 1] .= 0.0f0

# Source and receivers
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# Forward modeling
vx, vz, snaps = acoustic2d(vp, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# Visualization
plot_shot(trace_norm(vz, dims=2), "acoustic_vz.png")
plot_wavefield_video(snaps, 50, "acoustic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

### Elastic 2D

```julia
using CUDA
using Fomo

nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# Elastic model: vp, vs, rho
vp  = fill(2500.0f0, nx, nz)
vs  = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# Vacuum free surface
vp[:, 1]  .= 0.0f0
vs[:, 1]  .= 0.0f0
rho[:, 1] .= 0.0f0

# Source and receivers
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# Forward modeling
vx, vz, snaps = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# Visualization
plot_shot(trace_norm(vz, dims=2), "elastic_vz.png")
plot_wavefield_video(snaps, 50, "elastic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

### Example Output: Shot Record

<p align="center">
  <img src="vz.png" alt="Shot record (Vz component)" width="600">
</p>

## API Reference

### Forward Modeling

| Function | Description |
|---|---|
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)` | Acoustic wave equation forward modeling |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)` | Elastic wave equation forward modeling |

**Common keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `sx, sz` | — | Source positions (grid indices) |
| `rx, rz` | — | Receiver positions (grid indices) |
| `nbc` | `50` | Number of absorbing boundary grid points |
| `fd_order` | `8` | Finite difference order (2, 4, 6, 8, or 10) |
| `snap_interval` | `0` | Snapshot interval (0 = no snapshots) |

**Returns:** `(vx_record, vz_record, snapshots)`

### Utilities

| Function | Description |
|---|---|
| `ricker_wavelet(f0, dt, nt)` | Generate a Ricker wavelet source |
| `trace_norm(data; dims)` | Trace-by-trace normalization |
| `plot_shot(data, filename)` | Save a shot record figure |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)` | Export wavefield animation as video |

## Method

Fomo implements the velocity-stress formulation on a staggered grid (Virieux, 1986). Key numerical aspects:

- **Spatial discretization** — Standard staggered grid with up to 10th-order FD operators
- **Temporal discretization** — Second-order leapfrog time stepping
- **Boundary conditions** — Hybrid Absorbing Boundary Condition (HABC) combining a one-way wave equation with exponential damping layers (Liu Yang)
- **Free surface** — Vacuum method: setting velocity and density to zero above the free surface naturally enforces the zero-traction condition

## Requirements

- Julia ≥ 1.10
- NVIDIA GPU with CUDA support
- CUDA.jl ≥ 5.0

## License

MIT

## Acknowledgments

- Hybrid Absorbing Boundary Condition: based on the work of Prof. Liu Yang
- Staggered grid FD formulation: Virieux (1986)