using Plots
using Statistics

"""
    plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10)

Create a video from wavefield snapshots.
- `snaps`: Vector of matrices containing wavefield snapshots.
- `snapshot_interval`: Time interval between snapshots.
- `save_path`: Path to save the video file (default: "wavefield.mp4").
- `fps`: Frames per second for the output video (default: 10).
"""
function plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10)
    if isempty(snaps)
        println("No snapshots available for video creation.")
        return
    end

    anim = Animation()

    for (i, snap) in enumerate(snaps)
        # Calculate the corresponding actual time step
        actual_time_step = i * snapshot_interval

        # Use similar logic as create_video_callback
        max_val = maximum(abs.(snap))

        if max_val < 1e-10
            clims = (-1e-10, 1e-10)
        else
            scale = max_val * 0.5
            clims = (-scale, scale)
        end

        p1 = heatmap(snap',
            title="VZ Wavefield (Step $(actual_time_step))",
            xlabel="X (grid)", ylabel="Z (grid)",
            color=:seismic, clims=clims,
            aspect_ratio=:equal, legend=true,
            yflip=true
        )

        frame(anim)
    end

    mp4(anim, save_path, fps=fps)
    println("Wavefield video saved as $(save_path)")
end