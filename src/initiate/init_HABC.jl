# src/initiate/init_habc.jl
"""
    HABCConfig{T}

Higdon Absorbing Boundary Condition parameters.
"""
struct HABCConfig{T<:AbstractMatrix{Float32}}
    nbc::Int
    qx::Float32
    qz::Float32
    qt_x::Float32
    qt_z::Float32
    qxt::Float32

    w_vx::T
    w_vz::T
    w_tau::T
end


function init_habc(nx::Int, nz::Int, pad::Int, dt::Real, dh::Real,
    v_ref::Real)

    nx_pad = nx + 2 * pad
    nz_pad = nz + 2 * pad

    r = Float32(v_ref * dt / dh)
    b_p = 0.45f0
    beta = 1.0f0

    qx = Float32((b_p * (beta + r) - r) / ((beta + r) * (1 - b_p)))
    qz = Float32((b_p * (beta + r) - r) / ((beta + r) * (1 - b_p)))
    qt_x = Float32((b_p * (beta + r) - beta) / ((beta + r) * (1 - b_p)))
    qt_z = Float32((b_p * (beta + r) - beta) / ((beta + r) * (1 - b_p)))
    qxt = Float32(b_p / (b_p - 1.0f0))

    dist(i, j) = min(i - 1, nx - i, j - 1, nz - j)

    w_vx = [Float32(clamp((dist(i, j) - 0.0) / (pad - 1), 0.0, 1.0)) for j in 1:nz_pad, i in 1:nx_pad]
    w_vz = [Float32(clamp((dist(i, j) - 0.5) / (pad - 1), 0.0, 1.0)) for j in 1:nz_pad, i in 1:nx_pad]
    w_tau = [Float32(clamp((dist(i, j) - 0.75) / (pad - 1), 0.0, 1.0)) for j in 1:nz_pad, i in 1:nx_pad]

    return HABCConfig(
        pad - 1,
        qx, qz, qt_x, qt_z, qxt,
        to_device(w_vx),
        to_device(w_vz),
        to_device(w_tau)
    )
end
