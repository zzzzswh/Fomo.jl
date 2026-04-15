# Fomo.jl

**GPU-accelerated 2D/3D acoustic & elastic wave equation simulator in Julia**

> 🌐 [中文文档](README_CN.md) | **English**

<p align="center">
  <img src="wavefield.gif" alt="Elastic wavefield simulation" width="600">
</p>

**Fomo** is a CUDA-based 2D & 3D wave equation simulator that solves acoustic and elastic wave equations on staggered grids using high-order finite differences. While originally developed for seismic modeling, it is equally applicable to any scenario involving acoustic or elastic wave propagation — seismic forward modeling, ultrasonics, non-destructive testing, medical imaging, underwater acoustics, and more.

## Features

- **Acoustic & Elastic 2D/3D** — Full support for both acoustic (pressure-velocity) and elastic (stress-velocity) formulations
- **🆕 Coupled P-S Potential 2D** — A novel solver based on Li et al. (2018) that directly propagates separated P- and S-wave potentials with natural mode decomposition (see [details below](#-coupled-p-s-potential-solver))
- **Staggered Grid Finite Differences** — Up to 10th-order spatial accuracy for minimal numerical dispersion
- **Hybrid Absorbing Boundary (HABC)** — Advanced absorbing boundary condition (Liu Yang) combining one-way wave equations with exponential damping. Achieves better absorption than traditional PML with significantly lower computational cost
- **Vacuum Free Surface** — Employs the vacuum method (setting velocity and density to zero) to model realistic free surfaces at arbitrary locations within the medium.  
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

### 🆕 Coupled P-S Potential 2D

```julia
using CUDA
using Fomo

nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# Only vp and vs needed — no density (assumes ρ=1)
vp = fill(2500.0f0, nx, nz)
vs = fill(1200.0f0, nx, nz)
vp[:, 150:end] .= 3500.0f0
vs[:, 150:end] .= 1800.0f0

# Source and receivers
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# Forward modeling — returns separated P and S potentials
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# seis_P: P-wave potential (contains PP reflections)
# seis_S: S-wave potential (contains PS conversions — only at Vs discontinuities!)
```

### Example Output: Shot Record

<p align="center">
  <img src="vz.png" alt="Shot record (Vz component)" width="600">
</p>

## 🆕 Coupled P-S Potential Solver

### Theory

Based on **Li et al. (2018)** *"Elastic reverse time migration using acoustic propagators"* (Geophysics, Vol. 83, No. 5, S399–S408), we derive a set of coupled second-order wave equations for P- and S-wave potentials in a constant-density isotropic elastic medium:

$$\ddot{P} = P\nabla^2\alpha + 2\nabla\alpha\cdot\nabla P - 2P\nabla^2\beta - 2\nabla\beta\cdot(\nabla\times\mathbf{S}) + \alpha\nabla^2 P + \nabla\cdot\mathbf{f}$$

$$\ddot{\mathbf{S}} = \nabla\beta\cdot\nabla\mathbf{S} - (\nabla\beta)\times(\nabla\times\mathbf{S}) + 2(\nabla\beta)\times(\nabla P) + \beta\nabla^2\mathbf{S} + \nabla\times\mathbf{f}$$

where α = V²_P and β = V²_S are the squared P- and S-wave velocities, P = ∇·**u** is the P-wave potential (scalar), and **S** = ∇×**u** is the S-wave potential (vector; in 2D, only the y-component S_y is nonzero).

**Key physics revealed by these equations:**

1. **Mode conversion occurs only at V_S discontinuities** — if ∇β = 0, P- and S-waves are fully decoupled
2. **V_P discontinuities are transparent to S-waves** — pure V_P perturbations do not generate PS-converted waves
3. P- and S-wave potentials are **naturally separated** without Helmholtz decomposition

### Implementation: Velocity-Position Split with Second-Order HABC

A direct leapfrog implementation of these second-order equations allows only **one** HABC application per time step, yielding a first-order Higdon ABC with poor oblique-incidence absorption.

Our key insight: **rewrite the leapfrog as a velocity-position split** that is structurally isomorphic to the velocity-stress staggered-time scheme:

| Velocity-Stress (elastic2d) | Velocity-Position (coupled2d) |
|---|---|
| v_x, v_z (particle velocity) | dP/dt, dS/dt (potential rate) |
| τ_xx, τ_zz, τ_xz (stress) | P, S (potential) |
| update_velocity → **HABC** | update_velocity → **HABC** |
| update_stress → **HABC** | update_position → **HABC** |

With **two HABC applications per time step**, the scheme achieves a **second-order Higdon ABC**, matching the absorption quality of the conventional elastic solver.

### Advantages over Conventional Elastic Solver

| | elastic2d | coupled2d |
|---|---|---|
| Wavefield arrays (2D) | 10 (vx, vz, τ_xx, τ_zz, τ_xz + backups) | 8 (P, S, dP/dt, dS/dt + backups) |
| Memory | 5n² | 4n² **(–20%)** |
| P/S separation | Requires Helmholtz decomposition | **Built-in** |
| Mode conversion | Implicit | **Explicit (only at ∇β ≠ 0)** |
| Imaging condition | Ad hoc phase correction needed | **Consistent with physical perturbations** |
| Density | Arbitrary | Constant (ρ = const) |

## API Reference

### Forward Modeling

| Function | Description |
|---|---|
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)` | Acoustic wave equation forward modeling |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)` | Elastic wave equation forward modeling |
| `coupled2d(vp, vs, dh, dt, nt, f0; ...)` | 🆕 Coupled P-S potential forward modeling |

**Common keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `sx, sz` | — | Source positions (grid indices) |
| `rx, rz` | — | Receiver positions (grid indices) |
| `nbc` | `50` | Number of absorbing boundary grid points |
| `fd_order` | `8` | Finite difference order (2, 4, 6, 8, or 10) |
| `snap_interval` | `0` | Snapshot interval (0 = no snapshots) |

**`coupled2d` additional keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `v_ref_p` | `min(vp)` | HABC reference velocity for P-field |
| `v_ref_s` | `min(vs)` | HABC reference velocity for S-field |

**Returns:**
- `acoustic2d` / `elastic2d` → `(vx_record, vz_record, snapshots)`
- `coupled2d` → `(P_record, S_record, P_snapshots, S_snapshots)`

### Utilities

| Function | Description |
|---|---|
| `ricker_wavelet(f0, dt, nt)` | Generate a Ricker wavelet source |
| `trace_norm(data; dims)` | Trace-by-trace normalization |
| `plot_shot(data, filename)` | Save a shot record figure |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)` | Export wavefield animation as video |

## Method

Fomo implements three wave equation solvers:

**Acoustic & Elastic (velocity-stress formulation)** — Standard staggered grid scheme (Virieux, 1986) with up to 10th-order FD operators, second-order leapfrog time stepping, and HABC absorbing boundaries (Liu Yang).

**Coupled P-S Potential** — Based on the coupled second-order P- and S-wave potential equations derived by Li et al. (2018). The second-order-in-time equations are decomposed into a velocity-position split that is structurally isomorphic to the velocity-stress scheme, enabling second-order equivalent HABC absorption. This solver uses centered (non-staggered) FD operators on a regular grid.

## Requirements

- Julia ≥ 1.10
- NVIDIA GPU with CUDA support
- CUDA.jl ≥ 5.0

## License

MIT

## References

- **Li, Y. E., Du, Y., Yang, J., Cheng, A., & Fang, X. (2018).** Elastic reverse time migration using acoustic propagators. *Geophysics*, 83(5), S399–S408. doi: [10.1190/GEO2017-0687.1](https://doi.org/10.1190/GEO2017-0687.1)
- **Virieux, J. (1986).** P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.

## Acknowledgments

- Hybrid Absorbing Boundary Condition: based on the work of Prof. Liu Yang
- Staggered grid FD formulation: Virieux (1986)
- Coupled P-S potential equations: Li et al. (2018)