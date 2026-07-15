# Fomo.jl

**GPU-accelerated 2D/3D acoustic & elastic wave equation simulator in Julia**

*Run seismic forward modeling on your laptop.*

> 🌐 [中文文档](README_CN.md) | **English**

![Julia](https://img.shields.io/badge/Julia-1.10+-9558B2?logo=julia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

<p align="center">
  <img src="wavefield.gif" alt="Elastic wavefield simulation" width="600">
</p>

**Fomo** is a CUDA-based 2D & 3D wave equation simulator that solves acoustic and elastic wave equations on staggered grids using high-order finite differences. While originally developed for seismic modeling, it is equally applicable to any scenario involving acoustic or elastic wave propagation — seismic forward modeling, ultrasonics, non-destructive testing, medical imaging, underwater acoustics, and more.

It also introduces a **novel coupled P-S potential solver**, derived from the work of Prof. Yunyue Elita Li (Li et al., 2018), that propagates naturally separated P- and S-wave potentials with explicit mode decomposition — see [details below](#the-coupled-p-s-potential-solver).

---

## Contents

- [Fomo.jl](#fomojl)
  - [Contents](#contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Example Output: Shot Record](#example-output-shot-record)
  - [API Reference](#api-reference)
    - [Forward Modeling](#forward-modeling)
    - [Utilities](#utilities)
  - [The Coupled P-S Potential Solver](#the-coupled-p-s-potential-solver)
    - [Theory](#theory)
    - [Implementation: Velocity-Position Split with Second-Order HABC](#implementation-velocity-position-split-with-second-order-habc)
    - [Advantages over Conventional Elastic Solver](#advantages-over-conventional-elastic-solver)
  - [Performance \& Reproducibility](#performance--reproducibility)
  - [Numerical Method](#numerical-method)
  - [Requirements](#requirements)
  - [References](#references)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

---

## Features

- **Acoustic & Elastic, 2D & 3D** — Full support for both acoustic (pressure-velocity) and elastic (stress-velocity) formulations, with dedicated 2D and 3D solvers
- **🆕 Coupled P-S Potential 2D** — A novel solver based on Li et al. (2018) that directly propagates separated P- and S-wave potentials with natural mode decomposition (see [details below](#the-coupled-p-s-potential-solver))
- **Staggered Grid Finite Differences** — Up to 10th-order spatial accuracy for minimal numerical dispersion
- **Hybrid Absorbing Boundary (HABC)** — Combines one-way wave equations with exponential damping (Liu Yang); better absorption than traditional PML at significantly lower computational cost. A classic Cerjan sponge boundary is also available via `boundary=:sponge` (see `example/benchmark/` for a comparison)
- **Vacuum Free Surface** — Models realistic free surfaces at arbitrary locations via the vacuum method (zero velocity and density)
- **Input Validation** — Source/receiver geometry and CFL stability are checked before any kernel launch (hard error on violation), with a warning when the grid under-samples the shortest wavelength
- **🆕 Second-Order Scalar Solver** — `scalar2d` solves the constant-density scalar wave equation with ~1/3 the memory traffic of the first-order formulation: 2.5–3× faster on bandwidth-bound grids (the same formulation as deepwave's `scalar`)
- **🆕 Multi-Shot Batching** — `acoustic2d_batch` / `elastic2d_batch` propagate `n_shots` simultaneously in one GPU pass; each shot is bit-identical to its single-shot run
- **🆕 Fast & Deterministic Engine** — Fused CUDA kernels + CUDA Graphs (one graph launch per time step, up to ~2.6× on small/medium grids), and a deterministic two-pass HABC that makes results bit-reproducible across runs and GPUs (see [Performance & Reproducibility](#performance--reproducibility))
- **GPU Acceleration** — Built on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for high-performance computation on NVIDIA GPUs
- **Built-in Visualization** — Shot-record plotting and wavefield animation export, loaded lazily as a package extension when you `using Plots` (so `using Fomo` stays fast)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/zzzzswh/Fomo.jl")
```

## Quick Start

All solvers share the same calling style. Expand an example below to get started:

<details open>
<summary><b>Acoustic 2D</b></summary>

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
res = acoustic2d(vp, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)
# res.seis_p / res.seis_vx / res.seis_vz / res.snaps / res.stats.kernel_time_s

# Visualization (requires `using Plots` to load the plotting extension)
using Plots
plot_shot(trace_norm(res.seis_vz, dims=2), "acoustic_vz.png")
plot_wavefield_video(res.snaps, 50, "acoustic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

</details>

<details>
<summary><b>Elastic 2D</b></summary>

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
res = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)
# res.seis_vx / res.seis_vz / res.snaps / res.stats.kernel_time_s

# Visualization (requires `using Plots` to load the plotting extension)
using Plots
plot_shot(trace_norm(res.seis_vz, dims=2), "elastic_vz.png")
plot_wavefield_video(res.snaps, 50, "elastic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

</details>

<details>
<summary><b>Coupled P-S Potential 2D 🆕</b></summary>

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
res = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# res.seis_P: P-wave potential (contains PP reflections)
# res.seis_S: S-wave potential (contains PS conversions — only at Vs discontinuities!)
```

</details>

<details>
<summary><b>Acoustic & Elastic 3D</b></summary>

```julia
using CUDA
using Fomo

# Grid parameters
nx, ny, nz = 101, 101, 101
dh = 10.0f0
dt = 0.001f0
nt = 500
f0 = 15.0f0

# 3D velocity model
vp  = fill(3000.0f0, nx, ny, nz)
rho = fill(2000.0f0, nx, ny, nz)

# Source at the model center; receiver line along x
sx = [nx ÷ 2]; sy = [ny ÷ 2]; sz = [nz ÷ 2]
rx = collect(1:nx)
ry = fill(ny ÷ 2, nx)
rz = fill(nz ÷ 2, nx)

# Forward modeling — 3D solvers return a plain tuple
seis_vx, seis_vy, seis_vz, snaps = acoustic3d(vp, rho, dh, dt, nt, f0;
    sx, sy, sz, rx, ry, rz,
    nbc=50, fd_order=8,
    snap_interval=50,
    snap_plane=:xz,     # snapshot slice plane: :xy, :xz, or :yz
    snap_index=ny ÷ 2)  # slice index along the remaining axis

# Elastic 3D works the same way — just add vs:
# vs = fill(1700.0f0, nx, ny, nz)
# seis_vx, seis_vy, seis_vz, snaps = elastic3d(vp, vs, rho, dh, dt, nt, f0;
#     sx, sy, sz, rx, ry, rz, nbc=50, fd_order=8)
```

</details>

<details>
<summary><b>Scalar 2D & Multi-Shot Batching 🆕</b></summary>

```julia
using CUDA
using Fomo

nx, nz = 500, 400
dh, dt, nt, f0 = 10.0f0, 0.001f0, 2000, 15.0f0
vp  = fill(2500.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)
rx = collect(1:2:nx); rz = fill(10, length(rx))

# Second-order scalar (constant density): ~1/3 the memory traffic of the
# first-order path — the same formulation as deepwave's `scalar`
res = scalar2d(vp, dh, dt, nt, f0;
    sx=[nx ÷ 2], sz=[10], rx, rz)
# res.seis_u :: (n_rec, nt)

# Many shots in one GPU pass (shared receiver spread, per-shot sources).
# NOTE: here a vector means ONE SOURCE PER SHOT — unlike the single-shot
# API, where a vector lists multiple sources within a single shot.
res = acoustic2d_batch(vp, rho, dh, dt, nt, f0;
    sx=[100, 250, 400], sz=fill(10, 3), rx, rz)
# res.seis_p :: (n_rec, nt, 3); each shot is bit-identical to a single-shot run
```

</details>

### Example Output: Shot Record

<p align="center">
  <img src="vz.png" alt="Shot record (Vz component)" width="600">
</p>

## API Reference

### Forward Modeling

| Function | Description |
|---|---|
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)` | 2D acoustic wave equation |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)` | 2D elastic wave equation |
| `scalar2d(vp, dh, dt, nt, f0; ...)` | 🆕 2D second-order scalar (constant density), deepwave-`scalar`-equivalent formulation |
| `acoustic2d_batch(vp, rho, dh, dt, nt, f0; ...)` | 🆕 2D acoustic, many shots per GPU pass |
| `elastic2d_batch(vp, vs, rho, dh, dt, nt, f0; ...)` | 🆕 2D elastic, many shots per GPU pass |
| `coupled2d(vp, vs, dh, dt, nt, f0; ...)` | 🆕 2D coupled P-S potential |
| `acoustic3d(vp, rho, dh, dt, nt, f0; ...)` | 3D acoustic wave equation |
| `elastic3d(vp, vs, rho, dh, dt, nt, f0; ...)` | 3D elastic wave equation |

**Common keyword arguments (all solvers):**

| Argument | Default | Description |
|---|---|---|
| `sx, sz` (+ `sy` in 3D) | — | Source positions (grid indices) |
| `rx, rz` (+ `ry` in 3D) | — | Receiver positions (grid indices) |
| `nbc` | `50` | Number of absorbing boundary grid points |
| `fd_order` | `8` | Finite difference order (2, 4, 6, 8, or 10) |
| `snap_interval` | `0` | Snapshot interval (0 = no snapshots) |

**All 2D solvers (`acoustic2d` / `elastic2d` / `coupled2d`):**

| Argument | Default | Description |
|---|---|---|
| `wavelet` | `nothing` | Custom source wavelet (length-`nt` vector; `nothing` → Ricker(f0)) |
| `verbose` | `true` | Print progress logs |

**`acoustic2d` / `elastic2d` only:**

| Argument | Default | Description |
|---|---|---|
| `boundary` | `:habc` | Absorbing boundary type: `:habc` or `:sponge` (Cerjan) |
| `v_ref` | `min(vp)` | HABC reference velocity (ignored for `:sponge`) |
| `sponge_factor` | `0.015` | Cerjan damping factor (ignored for `:habc`) |
| `scale_source_by_dt` | `false` | Multiply the wavelet by `dt` before injection, keeping the source amplitude physically consistent across `dt` (matches deepwave's convention) |
| `use_cuda_graph` | `true` | With `boundary=:habc` and no snapshots, capture the whole time step as one CUDA Graph (auto-fallback to the fused loop if capture fails) |

**`scalar2d` only** (boundary fixed to HABC; no snapshots):

| Argument | Default | Description |
|---|---|---|
| `source_scale` | `:v2dt2` | Source scaling: `:v2dt2` (× v²·dt² at the source point, deepwave's convention), `:dt`, or `:none` |

**`acoustic2d_batch` / `elastic2d_batch`** (boundary fixed to HABC; no snapshots; receivers shared across shots):

| Argument | Default | Description |
|---|---|---|
| `sx, sz` | — | `(n_shots × n_src_per_shot)` integer matrix; a **vector means one source per shot** — note this differs from the single-shot API, where a vector lists multiple sources within one shot |

**`coupled2d` additional keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `v_ref_p` | `min(vp)` | HABC reference velocity for P-field |
| `v_ref_s` | `min(vs)` | HABC reference velocity for S-field |
| `smooth_sigma` | `3.0` | Gaussian smoothing σ (in grid points) applied to α, β for the coupling/scattering terms (∇α, ∇β, ∇²α, ∇²β); the propagation terms use the raw model. Set `0.0` to disable |

**`acoustic3d` / `elastic3d` additional keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `snap_plane` | `:xz` | Snapshot slice plane: `:xy`, `:xz`, or `:yz` |
| `snap_index` | `ny ÷ 2` | Slice index along the axis normal to `snap_plane` |
| `v_ref` | `min(vp)` | HABC reference velocity |

**Returns:**
- `acoustic2d` → NamedTuple `(; seis_p, seis_vx, seis_vz, snaps, stats)`
- `elastic2d` → NamedTuple `(; seis_vx, seis_vz, snaps, stats)`
- `coupled2d` → NamedTuple `(; seis_P, seis_S, snaps_P, snaps_S, stats)`
- `scalar2d` → NamedTuple `(; seis_u, stats)`
- `acoustic2d_batch` → NamedTuple `(; seis_p, seis_vx, seis_vz, stats)`, seismograms shaped `(n_rec, nt, n_shots)`
- `elastic2d_batch` → NamedTuple `(; seis_vx, seis_vz, stats)`, same `(n_rec, nt, n_shots)` shape
- `acoustic3d` / `elastic3d` → plain tuple `(seis_vx, seis_vy, seis_vz, snaps)`; snapshots are 2D slices (pressure for acoustic, `vz` for elastic) taken at `snap_plane` / `snap_index`
- `stats.kernel_time_s`: GPU main-loop time (after in-call warmup)

**Staggered field positions** (mind the half-cell offsets when comparing with other codes):
`p`/`τxx`/`τzz` at `(i, j)`, `vx` at `(i−1/2, j)`, `vz` at `(i, j+1/2)`, `τxz` at `(i−1/2, j+1/2)`.

### Utilities

| Function | Description |
|---|---|
| `ricker_wavelet(f0, dt, nt)` | Generate a Ricker wavelet source |
| `trace_norm(data; dims)` | Trace-by-trace normalization |
| `plot_shot(data, filename)` | Save a shot record figure |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)` | Export wavefield animation as video |

> `plot_shot` and `plot_wavefield_video` live in a package extension — run `using Plots` first to load them.

## The Coupled P-S Potential Solver

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

## Performance & Reproducibility

On the HABC path, the 2D engine fuses each phase's boundary backup and PDE update into single kernels, runs boundary work on frame-mapped threads (zero idle lanes), and captures the whole time step as one **CUDA Graph** — one graph launch per step. The Higdon boundary correction uses a **deterministic two-pass scheme**, so results are **bit-reproducible** across runs, launch geometries, and GPUs; each shot in a batch is likewise bit-identical to its single-shot run.

Throughput measured on an RTX 3060 (Float32, HABC, `fd_order=8`, `nbc=50`, medians of repeated runs; data-center GPUs scale further):

| Grid | `acoustic2d` (steps/s) | `scalar2d` (steps/s) | `elastic2d` (steps/s) |
|---|---|---|---|
| 240×200 | ~25,000 | ~24,400 | ~17,000 |
| 500×400 | ~11,800 | ~23,800 | ~7,700 |
| 1000×800 | ~4,400 | ~11,800 | ~2,600 |
| 2000×1600 | ~1,300 | ~4,100 | — |

For constant-density acoustic work on large (bandwidth-bound) grids, `scalar2d` is the recommended solver: it moves ~1/3 the memory per step of the first-order formulation. Verification scripts live in `test/`: `verify_det_graph.jl` (determinism + CUDA-Graph bit-exactness), `verify_batch.jl` (per-shot bit-exactness of batching), `verify_scalar.jl` (scalar scheme vs a race-free CPU reference + grid self-convergence), `compare_fused*.jl` (kernel-fusion equivalence vs the legacy kernel sequence), and `bench_*.jl` for throughput.

## Numerical Method

Fomo implements four families of wave equation solvers:

- **Acoustic & Elastic (2D and 3D)** — Standard staggered-grid velocity-stress scheme (Virieux, 1986) with up to 10th-order FD operators, second-order leapfrog time stepping, and HABC absorbing boundaries (Liu Yang). The 2D solvers can alternatively use a Cerjan sponge boundary (`boundary=:sponge`).
- **Coupled P-S Potential (2D)** — Based on the coupled second-order potential equations of Li et al. (2018), decomposed into a velocity-position split that is structurally isomorphic to the velocity-stress scheme (see [above](#the-coupled-p-s-potential-solver) for details). Uses centered (non-staggered) FD operators on a regular grid.
- **Second-Order Scalar (2D)** — Constant-density scalar wave equation on a regular grid: centered FD operators (up to 10th order), leapfrog time stepping, and the same deterministic HABC. This is the formulation used by deepwave's `scalar` solver, at ~1/3 the memory traffic of the first-order acoustic path.

All entry points validate source/receiver geometry and the CFL stability condition before launching any GPU kernel (throwing an error on violation), and issue a warning when the grid under-samples the shortest wavelength (dispersion risk).

## Requirements

- Julia ≥ 1.10
- NVIDIA GPU with CUDA support
- CUDA.jl ≥ 5.0

## References

- **Li, Y. E., Du, Y., Yang, J., Cheng, A., & Fang, X. (2018).** Elastic reverse time migration using acoustic propagators. *Geophysics*, 83(5), S399–S408. doi: [10.1190/GEO2017-0687.1](https://doi.org/10.1190/GEO2017-0687.1)
- **Virieux, J. (1986).** P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.
- **Cerjan, C., Kosloff, D., Kosloff, R., & Reshef, M. (1985).** A nonreflecting boundary condition for discrete acoustic and elastic wave equations. *Geophysics*, 50(4), 705–708.

## Acknowledgments

- Hybrid Absorbing Boundary Condition: based on the work of Prof. Liu Yang
- Sponge absorbing boundary: Cerjan et al. (1985)
- Staggered grid FD formulation: Virieux (1986)
- Coupled P-S potential equations: Li et al. (2018)

## License

MIT