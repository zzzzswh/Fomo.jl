# src/structures/medium.jl
"""
    Medium{T}

Physical properties of the simulation domain.
OPTIMIZATION: Includes precomputed buoyancy (1/rho) and lam_2mu (lambda + 2*mu)
              to eliminate expensive divisions in the simulation loop.
"""
struct Medium{T}
    nx::Int
    nz::Int
    dh::Float32
    x_max::Float32
    z_max::Float32
    M::Int              # FD half-stencil width
    pad::Int            # Boundary padding

    # Original material properties
    lam::T              # Lambda (Lamé's first parameter)
    mu_txx::T           # Mu at txx/tzz positions
    mu_txz::T           # Mu at txz positions (harmonic average)

    # OPTIMIZED: Precomputed values to eliminate divisions in hot loops
    buoy_vx::T          # 1/rho at vx positions (buoyancy)
    buoy_vz::T          # 1/rho at vz positions (buoyancy)
    lam_2mu::T          # lambda + 2*mu (precomputed)
end