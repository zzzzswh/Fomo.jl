
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

