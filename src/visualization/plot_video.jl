using Plots

# Set headless mode (silent execution without pop-up windows) and use GR backend
ENV["GKSwstype"] = "100"
gr()

"""
    plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10, adaptive_clims=false)

Create a video from wavefield snapshots without popping up windows.
If adaptive_clims is true, each frame will have its own color limits based on its maximum value.
If adaptive_clims is false (default), global color limits are used for all frames to avoid flickering.
"""
function plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10, adaptive_clims=false)
    isempty(snaps) && (println("No snapshots available for video creation."); return)

    # Calculate global maximum for fixed colorbar to avoid video flickering (when adaptive_clims is false)
    global_max = adaptive_clims ? 0.0f0 : maximum(maximum.(abs, snaps))
    global_scale = global_max < 1e-10 ? 1e-10 : global_max * 0.5
    global_clims = (-global_scale, global_scale)

    # Render each frame in the loop
    anim = @animate for (i, snap) in enumerate(snaps)
        if adaptive_clims
            # Calculate adaptive color limits for each frame
            frame_max = maximum(abs, snap)
            scale = frame_max < 1e-10 ? 1e-10 : frame_max * 0.5
            clims = (-scale, scale)
        else
            # Use global color limits
            clims = global_clims
        end
        
        heatmap(snap',
            title="VZ Wavefield (Step $(i * snapshot_interval))",
            xlabel="X (grid)", ylabel="Z (grid)",
            color=:seismic, clims=clims,
            aspect_ratio=:equal, legend=true, yflip=true
        )
    end

    # Export video
    mp4(anim, save_path, fps=fps)
    println("Wavefield video saved as $(save_path)")
end