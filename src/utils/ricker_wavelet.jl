"""
    ricker_wavelet(f0, dt, nt)

Generate a Ricker wavelet.
"""
function ricker_wavelet(f0, dt, nt)
    t = (0:nt-1) .* dt
    t0 = 1.0f0 / f0
    tau = t .- 1.5f0 * t0
    val = (1.0f0 .- 2.0f0 .* (pi * f0 .* tau) .^ 2) .* exp.(-(pi * f0 .* tau) .^ 2)
    return Float32.(val)
end