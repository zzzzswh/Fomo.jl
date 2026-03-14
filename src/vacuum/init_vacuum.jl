function init_medium_vacuum(vp::Matrix, vs::Matrix, rho::Matrix,
    dh::Real, nbc::Int, fd_order::Int,
    backend::AbstractBackend; surface_elevation=nothing)

    M = fd_order ÷ 2
    pad = nbc + M

    vp_t = permutedims(vp)
    vs_t = permutedims(vs)
    rho_t = permutedims(rho)

    nx_inner, nz_inner = size(vp_t)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    vp_pad = _pad_array(vp_t, pad)
    vs_pad = _pad_array(vs_t, pad)
    rho_pad = _pad_array(rho_t, pad)

    if surface_elevation !== nothing
        surface_j = setup_vacuum_formulation!(vp_pad, vs_pad, rho_pad, surface_elevation, dh, pad)
    end

    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = compute_staggered_params_vacuum(vp_pad, vs_pad, rho_pad)

    return Medium(
        nx, nz, Float32(dh), x_max, z_max,
        M, pad, true,
        to_device(lam, backend),
        to_device(mu_txx, backend),
        to_device(mu_txz, backend),
        to_device(buoy_vx, backend),
        to_device(buoy_vz, backend),
        to_device(lam_2mu, backend)
    )
end

"""
    compute_staggered_params_vacuum(vp, vs, rho, surface_j)

Compute staggered grid parameters using the improved vacuum formulation.

This is the core of the vacuum formulation. The key modifications from
standard parameter averaging are:

1. **Effective buoyancy b̄x** (for vx update):
   - If both ρ[i,j] and ρ[i+1,j] are 0 → b̄x = 0
   - Otherwise: b̄x = 2 / (ρ[i,j] + ρ[i+1,j])

2. **Effective buoyancy b̄z** (for vz update):
   - If both ρ[i,j] and ρ[i,j+1] are 0 → b̄z = 0  
   - Otherwise: b̄z = 2 / (ρ[i,j] + ρ[i,j+1])

3. **Effective shear modulus μ̄xz** (for τxz update) - THE KEY!
   - If ANY of μ[i,j], μ[i+1,j], μ[i,j+1], μ[i+1,j+1] is 0 → μ̄xz = 0
   - Otherwise: harmonic average of 4 points

The μ̄xz = 0 condition ensures τxz = 0 at the vacuum-solid interface
automatically, without explicit boundary condition application.

# Arguments
- `vp`, `vs`, `rho`: Padded material arrays [nx, nz]
- `surface_j`: Surface j-indices for each i [nx] (optional, for validation)

# Returns
- `lam`: First Lamé parameter λ
- `mu`: Shear modulus μ at τxx/τzz positions
- `mu_txz`: Effective shear modulus at τxz positions (with vacuum formulation!)
- `buoy_vx`: Effective buoyancy for vx update
- `buoy_vz`: Effective buoyancy for vz update  
- `lam_2mu`: Precomputed λ + 2μ
"""
function compute_staggered_params_vacuum(vp::Matrix{Float32}, vs::Matrix{Float32},
    rho::Matrix{Float32})
    nx, nz = size(vp)

    # Basic Lamé parameters
    mu = @. Float32(rho * vs^2)
    lam = @. Float32(rho * vp^2 - 2.0f0 * mu)

    # Precompute λ + 2μ
    lam_2mu = @. Float32(lam + 2.0f0 * mu)

    # Initialize effective parameters
    buoy_vx = zeros(Float32, nx, nz)
    buoy_vz = zeros(Float32, nx, nz)
    mu_txz = zeros(Float32, nx, nz)

    # =========================================================================
    # Effective buoyancy for vx: b̄x at (i+1/2, j)
    # Arithmetic average of ρ at (i,j) and (i+1,j), then invert
    # VACUUM RULE: If both ρ are 0, b̄x = 0
    # =========================================================================
    @inbounds for j in 1:nz
        for i in 1:nx-1
            ρ1, ρ2 = rho[i, j], rho[i+1, j]

            if ρ1 == 0.0f0 && ρ2 == 0.0f0
                # Both in vacuum → buoyancy is 0 (no motion in vacuum)
                buoy_vx[i, j] = 0.0f0
            elseif ρ1 == 0.0f0
                # Point (i,j) is vacuum, (i+1,j) is solid
                # Use 2/ρ2 as per Mittet (2002) / Zeng et al. (2012)
                buoy_vx[i, j] = 2.0f0 / ρ2
            elseif ρ2 == 0.0f0
                # Point (i,j) is solid, (i+1,j) is vacuum
                buoy_vx[i, j] = 2.0f0 / ρ1
            else
                # Both in solid → standard arithmetic average
                buoy_vx[i, j] = 2.0f0 / (ρ1 + ρ2)
            end
        end
    end
    buoy_vx[nx, :] .= buoy_vx[nx-1, :]  # Boundary extension

    # =========================================================================
    # Effective buoyancy for vz: b̄z at (i, j+1/2)
    # VACUUM RULE: If both ρ are 0, b̄z = 0
    # =========================================================================
    @inbounds for j in 1:nz-1
        for i in 1:nx
            ρ1, ρ2 = rho[i, j], rho[i, j+1]

            if ρ1 == 0.0f0 && ρ2 == 0.0f0
                buoy_vz[i, j] = 0.0f0
            elseif ρ1 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / ρ2
            elseif ρ2 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / ρ1
            else
                buoy_vz[i, j] = 2.0f0 / (ρ1 + ρ2)
            end
        end
    end
    buoy_vz[:, nz] .= buoy_vz[:, nz-1]  # Boundary extension

    # =========================================================================
    # Effective shear modulus for τxz: μ̄xz at (i+1/2, j+1/2)
    # THIS IS THE KEY TO THE VACUUM FORMULATION!
    # 
    # VACUUM RULE: If ANY of the 4 μ values is 0 → μ̄xz = 0
    # This ensures τxz = 0 at the vacuum-solid interface automatically!
    # 
    # Otherwise: 4-point harmonic average
    # =========================================================================
    @inbounds for j in 1:nz-1
        for i in 1:nx-1
            μ1 = mu[i, j]
            μ2 = mu[i+1, j]
            μ3 = mu[i, j+1]
            μ4 = mu[i+1, j+1]

            # THE KEY: if any μ is 0, the effective μ is 0
            if μ1 * μ2 * μ3 * μ4 == 0.0f0
                mu_txz[i, j] = 0.0f0
            else
                # 4-point harmonic average
                mu_txz[i, j] = 4.0f0 / (1.0f0 / μ1 + 1.0f0 / μ2 + 1.0f0 / μ3 + 1.0f0 / μ4)
            end
        end
    end
    mu_txz[nx, :] .= mu_txz[nx-1, :]
    mu_txz[:, nz] .= mu_txz[:, nz-1]

    return lam, mu, mu_txz, buoy_vx, buoy_vz, lam_2mu
end

# ==============================================================================
# High-level interface
# ==============================================================================

"""
    setup_vacuum_formulation!(vp, vs, rho, surface_elevation, dz, pad)

Complete setup for vacuum formulation with irregular surface.

# Arguments
- `vp`, `vs`, `rho`: Material arrays [nx, nz] (will be modified!)
- `surface_elevation`: Surface elevation in physical coordinates [nx_inner]
- `dz`: Grid spacing
- `pad`: Padding size

# Returns
- `surface_j`: Surface j-indices
- Modified vp, vs, rho in-place (vacuum region set to 0)
"""
function setup_vacuum_formulation!(vp::Matrix{Float32}, vs::Matrix{Float32},
    rho::Matrix{Float32},
    surface_elevation::Vector{<:Real},
    dh::Real, pad::Int)
    nx, nz = size(vp)

    # Compute surface indices
    surface_j = compute_surface_indices(surface_elevation, dz, pad, nx)

    # Clamp to valid z-index range
    @inbounds for i in 1:nx
        surface_j[i] = clamp(surface_j[i], 1, nz)
    end

    # Apply vacuum mask
    apply_vacuum_mask!(vp, vs, rho, surface_j, pad)

    return surface_j
end

# ==============================================================================
# Utility functions
# ==============================================================================

"""
    create_flat_surface(nx_inner, pad) -> Vector{Int}

Create surface indices for a flat surface at z=0.

# Returns
- `surface_j`: All values are pad+1 (the original top surface position)
"""
function create_flat_surface(nx::Int, pad::Int)
    return fill(pad + 1, nx)
end

"""
    create_sinusoidal_surface(nx_inner, amplitude, wavelength, dh, pad) -> Vector{Float32}

Create a sinusoidal surface elevation profile (for testing).

# Arguments
- `nx_inner`: Number of grid points in x (inner region)
- `amplitude`: Amplitude of sine wave in meters
- `wavelength`: Wavelength of sine wave in meters
- `dh`: Grid spacing in both x and z directions
- `pad`: Padding size

# Returns
- `elevation`: Surface elevation array [nx_inner]
"""
function create_sinusoidal_surface(nx_inner::Int, amplitude::Real, wavelength::Real,
    dh::Real, pad::Int)
    elevation = Vector{Float32}(undef, nx_inner)

    for i in 1:nx_inner
        x = (i - 1) * dh
        elevation[i] = Float32(amplitude * sin(2π * x / wavelength))
    end

    return elevation
end
