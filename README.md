# Fomo.jl

**A hackable GPU wave solver in Julia — featuring capabilities like built-in P/S wave separation.**

> 🌐 [中文文档](README_CN.md) | **English**

<p align="center">
  <img src="wavefield.gif" alt="Elastic wavefield simulation" width="600">
</p>

---

## About

Fomo.jl is a GPU wave equation solver written in Julia + CUDA.jl. The goal is a readable, extensible codebase that runs on a single consumer NVIDIA GPU with reasonable performance.

The package currently provides 2D/3D acoustic and elastic forward modeling (velocity-stress formulation), HABC absorbing boundaries, vacuum free surfaces, and a coupled P-S potential 2D solver based on Li et al. (2018) — which under the assumption of constant-density isotropic media gives natively separated P- and S-wavefields without requiring Helmholtz post-processing.

### How it compares

A rough feature comparison with related open-source packages:

|                                  | SOFI2D/3D | Devito  | Deepwave           | JUDI.jl            | Fomo.jl                    |
| -------------------------------- | --------- | ------- | ------------------ | ------------------ | -------------------------- |
| Backend                          | Fortran   | DSL→C   | PyTorch + C/CUDA   | Julia (via Devito) | Julia + CUDA.jl            |
| Single consumer GPU              | ❌         | ⚠️      | ✅                  | ⚠️                 | ✅                          |
| End-to-end readable source       | ⚠️ Fortran | Generated | ❌ C kernels      | Wrapper layer      | ✅ All Julia                |
| Coupled P-S potential solver     | ❌         | ❌       | ❌                  | ❌                  | ✅                          |
| Native P/S separation            | Post-process | Post-process | N/A         | Post-process       | ✅ (in ρ=const, isotropic)  |
| Absorbing boundary               | PML       | PML     | PML                | PML                | HABC                       |

Performance on a single RTX-class GPU: Fomo is roughly 60× faster than SOFI2D and within ~1.6× of Deepwave on comparable problems.

---

## Features

- **Acoustic & Elastic 2D/3D** — full velocity-stress formulation on staggered grids, up to 10th-order spatial accuracy
- **🆕 Coupled P-S Potential 2D** — based on Li et al. (2018); the second-order leapfrog is rewritten as a velocity-position split so HABC can be applied twice per step (see [the coupled2d section](#-the-coupled-p-s-potential-solver) below)
- **Hybrid Absorbing Boundary (HABC)** — combines one-way wave equations with exponential damping. Better absorption than PML at lower memory cost
- **Vacuum Free Surface** — realistic free surfaces at arbitrary locations via the vacuum method
- **PyTorch-style API** — one function call to run a full forward simulation; no hand-written time loops, no config files, no MPI launchers
- **GPU acceleration** — built on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), kernels are plain Julia code you can read and modify
- **Built-in visualization** — shot records and wavefield videos out of the box

---

## Quick Start

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

### Elastic 2D — one function call

```julia
using CUDA, Fomo

nx, nz = 400, 300
dh, dt, nt, f0 = 10.0f0, 0.001f0, 2000, 15.0f0

vp  = fill(2500.0f0, nx, nz)
vs  = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)
vp[:,1] .= 0; vs[:,1] .= 0; rho[:,1] .= 0  # vacuum free surface

sx, sz = [nx ÷ 2], [10]
rx, rz = collect(1:2:nx), fill(10, length(1:2:nx))

vx, vz, snaps = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=100, fd_order=8, snap_interval=50)

plot_shot(trace_norm(vz, dims=2), "elastic_vz.png")
plot_wavefield_video(snaps, 50, "elastic.mp4", fps=10, adaptive_clims=true)
```

### Coupled P-S potential — natively separated P/S wavefields

```julia
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=100, fd_order=8, snap_interval=50)
# seis_P: pure P-wave potential (PP reflections only)
# seis_S: pure S-wave potential (PS conversions only at Vs discontinuities)
```

No Helmholtz decomposition. No post-processing. The physics is separated at the equation level.

---

## 🆕 The Coupled P-S Potential Solver

### Context

For elastic reverse-time migration (RTM) and elastic FWI, separating P- and S-wavefields is often necessary. The standard workflow propagates the full velocity-stress wavefield and then applies Helmholtz decomposition (`∇·` and `∇×` operators) on saved snapshots, which adds storage, computation, and post-processing complexity.

The coupled P-S potential formulation of Li et al. (2018) propagates separated potentials directly at the equation level — under the assumption of **constant-density, isotropic** media. `coupled2d` is an open-source GPU implementation of that formulation; I haven't come across another one, though the literature is large and I may simply have missed it.

### Theory

In a constant-density isotropic elastic medium, the second-order coupled equations for the P-wave potential P = ∇·**u** and S-wave potential **S** = ∇×**u** are:

$$\ddot{P} = P\nabla^2\alpha + 2\nabla\alpha\cdot\nabla P - 2P\nabla^2\beta - 2\nabla\beta\cdot(\nabla\times\mathbf{S}) + \alpha\nabla^2 P + \nabla\cdot\mathbf{f}$$

$$\ddot{\mathbf{S}} = \nabla\beta\cdot\nabla\mathbf{S} - (\nabla\beta)\times(\nabla\times\mathbf{S}) + 2(\nabla\beta)\times(\nabla P) + \beta\nabla^2\mathbf{S} + \nabla\times\mathbf{f}$$

where α = V²ₚ, β = V²ₛ. The physics these equations reveal:

1. **Mode conversion happens only at V_S discontinuities** — if ∇β = 0, P and S decouple completely.
2. **V_P discontinuities are transparent to S-waves** — pure V_P contrasts produce no PS conversions.
3. P and S are **naturally separated** without Helmholtz decomposition.

### Handling absorbing boundaries

When implementing these equations, absorbing boundaries become the tricky part.

A naive second-order leapfrog only allows one HABC application per time step, which is equivalent to a first-order Higdon absorbing boundary — poor absorption at oblique incidence.

One workable approach is to rewrite the leapfrog as a velocity-position split, making it structurally isomorphic to the standard velocity-stress scheme used in `elastic2d`:

| velocity-stress (`elastic2d`)        | velocity-position (`coupled2d`)            |
| ------------------------------------ | ------------------------------------------ |
| `v_x, v_z` (particle velocity)       | `dP/dt, dS/dt` (potential rate)            |
| `τ_xx, τ_zz, τ_xz` (stress)          | `P, S` (potential)                         |
| `update_velocity → HABC`             | `update_velocity → HABC`                   |
| `update_stress → HABC`               | `update_position → HABC`                   |

This allows two HABC applications per step, equivalent to a second-order Higdon ABC, matching the absorption quality of `elastic2d`. The split is not addressed in Li et al.; it's something `coupled2d` works out at the implementation level.

### Advantages over the conventional elastic solver

|                              | elastic2d                       | **coupled2d**                       |
| ---------------------------- | ------------------------------- | ----------------------------------- |
| Wavefield arrays (2D)        | 10 (vx, vz, 3 stresses + backups) | 8 (P, S, dP/dt, dS/dt + backups)  |
| Memory footprint             | 5n²                             | **4n² (–20%)**                      |
| P/S separation               | Requires Helmholtz post-proc.   | **Built in**                        |
| Mode conversion              | Implicit                        | **Explicit (only at ∇β ≠ 0)**       |
| RTM imaging condition        | Ad-hoc phase correction         | **Consistent with physics**         |
| Density                      | Arbitrary                       | Constant (ρ = const)                |

---

## API Reference

### Forward Modeling

| Function                                           | Description                                   |
| -------------------------------------------------- | --------------------------------------------- |
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)`         | 2D acoustic forward modeling                  |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)`      | 2D elastic forward modeling                   |
| `coupled2d(vp, vs, dh, dt, nt, f0; ...)`           | 🆕 2D coupled P-S potential forward modeling  |
| `acoustic3d(vp, rho, dh, dt, nt, f0; ...)`         | 3D acoustic forward modeling                  |
| `elastic3d(vp, vs, rho, dh, dt, nt, f0; ...)`      | 3D elastic forward modeling                   |

**Common keyword arguments:**

| Argument        | Default | Description                                   |
| --------------- | ------- | --------------------------------------------- |
| `sx, sz`        | —       | Source positions (grid indices)               |
| `rx, rz`        | —       | Receiver positions (grid indices)             |
| `nbc`           | `50`    | Number of absorbing boundary grid points      |
| `fd_order`      | `8`     | Finite-difference order (2, 4, 6, 8, or 10)   |
| `snap_interval` | `0`     | Snapshot interval (0 = no snapshots)          |

**`coupled2d`-specific:**

| Argument        | Default     | Description                            |
| --------------- | ----------- | -------------------------------------- |
| `v_ref_p`       | `min(vp)`   | HABC reference velocity for P-field    |
| `v_ref_s`       | `min(vs)`   | HABC reference velocity for S-field    |
| `smooth_sigma`  | `3.0`       | Medium parameter Gaussian smoothing σ  |

**Returns:**
- `acoustic2d` / `elastic2d` → `(vx_record, vz_record, snapshots)`
- `coupled2d` → `(P_record, S_record, P_snapshots, S_snapshots)`

### Utilities

| Function                                                                 | Description                  |
| ------------------------------------------------------------------------ | ---------------------------- |
| `ricker_wavelet(f0, dt, nt)`                                             | Generate Ricker source       |
| `trace_norm(data; dims)`                                                 | Trace-by-trace normalization |
| `plot_shot(data, filename)`                                              | Save shot-record figure      |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)`   | Export wavefield video       |

---

## Architecture

```
src/
├── Fomo.jl                  # Module entry
├── acquisition/             # Sources & receivers (2D + 3D)
├── boundary/
│   ├── habc/                # Hybrid Absorbing Boundary (Liu Yang)
│   └── sponge.jl            # Optional sponge boundary
├── equations/
│   ├── acoustic2d/          # 2D acoustic (velocity-pressure)
│   ├── acoustic3d/          # 3D acoustic
│   ├── elastic2d/           # 2D elastic (velocity-stress)
│   ├── elastic3d/           # 3D elastic
│   └── coupled2d/           # 🆕 Coupled P-S potential
├── utils/                   # FD coefficients, padding, wavelets
└── visualization/           # Plotting & video export
```

Adding a new equation = drop a folder into `equations/` with `medium.jl`, `wavefield.jl`, `update_*.jl`, and an entry function. **The coupled2d solver was added in a few weeks using this structure** — that's the design payoff.

---

## Method

Fomo implements three families of wave equation solvers:

**Acoustic & Elastic (velocity-stress)** — Standard staggered-grid scheme (Virieux, 1986) with up to 10th-order FD operators, second-order leapfrog time stepping, and HABC absorbing boundaries.

**Coupled P-S Potential** — Based on Li et al. (2018). The second-order-in-time equations are recast as a velocity-position split that is structurally isomorphic to velocity-stress, enabling second-order-equivalent HABC absorption. Uses centered (non-staggered) FD operators on a regular grid.

---

## Requirements

- Julia ≥ 1.10
- NVIDIA GPU with CUDA support
- CUDA.jl ≥ 5.0

---

## License

MIT

---

## References

- **Li, Y. E., Du, Y., Yang, J., Cheng, A., & Fang, X. (2018).** Elastic reverse time migration using acoustic propagators. *Geophysics*, 83(5), S399–S408. doi: [10.1190/GEO2017-0687.1](https://doi.org/10.1190/GEO2017-0687.1)
- **Virieux, J. (1986).** P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.
- **Liu, Y., & Sen, M. K. (2010).** A hybrid scheme for absorbing edge reflections in numerical modeling of wave propagation. *Geophysics*, 75(2), A1–A6.

## Acknowledgments

- HABC formulation: Prof. Liu Yang
- Staggered-grid FD: Virieux (1986)
- Coupled P-S potential equations: Li et al. (2018)
- Implementation, GPU port, and velocity-position split scheme: this work
