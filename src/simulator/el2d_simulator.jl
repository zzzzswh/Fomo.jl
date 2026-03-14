# src/simulator/el2d_simulator.jl
using ProgressMeter

function _run_core!(W, M_med, S, R, H, p, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)
    nt = p.nt
    dt = p.dt
    pad = M_med.pad

    @showprogress 1 "Simulation: " for it in 1:nt
        # A update_velocity
        backup_boundary!(W, H, M_med)
        update_velocity!(W, M_med, a_static, p, inner_nx, inner_nz)
        apply_habc_velocity!(W, H, M_med)

        # B update_stress
        backup_boundary!(W, H, M_med)
        update_stress!(W, M_med, a_static, p, inner_nx, inner_nz)
        apply_habc_stress!(W, H, M_med)

        # C inject_source
        inject_source!(W.txx, S, it, dt)
        inject_source!(W.tzz, S, it, dt)

        # D record seismograms
        record_receivers!(seis_vx, W.vx, R, it)
        record_receivers!(seis_vz, W.vz, R, it)

        # save snapshots
        if snap_interval > 0 && it % snap_interval == 0
            vz_inner = Array(W.vz)[pad+1:end-pad, pad+1:end-pad]
            push!(snaps, vz_inner)
        end
    end
end


"""
Minimized entry function:
Only need to pass in snap_interval to return vx receiver data, vz receiver data, and wavefield snapshots list.
"""
function run_simulation!(W, M_med, S, R, H, p; snap_interval::Int=0)
    a_static = SVector{p.M,Float32}(p.a)
    inner_nx, inner_nz = M_med.nx - 2 * p.M, M_med.nz - 2 * p.M

    seis_vx = Data.Array(zeros(Float32, length(R.rx), p.nt))
    seis_vz = Data.Array(zeros(Float32, length(R.rx), p.nt))

    snaps = Vector{Matrix{Float32}}()

    @info "Starting Simulation..."
    _run_core!(W, M_med, S, R, H, p, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)

    @info "Simulation Complete!"

    return Array(seis_vx), Array(seis_vz), snaps
end