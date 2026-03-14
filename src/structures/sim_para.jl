"""
    SimParams

Time stepping and grid parameters.
"""
struct SimParams
    dt::Float32
    nt::Int
    dtx::Float32        # dt/dh
    dtz::Float32        # dt/dh
    fd_order::Int
    M::Int              # FD half-stencil width
    a::Vector{Float32}  # FD coefficients
    snapshot_interval::Int # Interval for printing progress
end

const FD_COEFFICIENTS = Dict(
    2 => Float32[1.0],
    4 => Float32[1.125, -0.041666667],
    6 => Float32[1.171875, -0.065104167, 0.0046875],
    8 => Float32[1.1962890625, -0.079752604167, 0.0095703125, -0.000697544643],
    10 => Float32[1.2115478515625, -0.089721679687, 0.0138427734375, -0.00176565987723, 0.0001186795166]
)

function get_fd_coefficients(order::Int)
    haskey(FD_COEFFICIENTS, order) || error("Unsupported FD order: $order")
    return FD_COEFFICIENTS[order]
end

function SimParams(dt, nt, dh, fd_order; snapshot_interval=100)
    M = fd_order ÷ 2
    a = get_fd_coefficients(fd_order)
    SimParams(Float32(dt), nt, Float32(dt / dh), Float32(dt / dh), fd_order, M, a, snapshot_interval)
end