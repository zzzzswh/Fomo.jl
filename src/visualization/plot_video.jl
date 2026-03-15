using Plots

# 设置无头模式（静默运行无弹窗）并使用 GR 后端
ENV["GKSwstype"] = "100"
gr()

"""
    plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10)

Create a video from wavefield snapshots without popping up windows.
"""
function plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10)
    isempty(snaps) && (println("No snapshots available for video creation."); return)

    # 计算全局最大值以固定颜色条，避免视频闪烁
    global_max = maximum(maximum.(abs, snaps))
    scale = global_max < 1e-10 ? 1e-10 : global_max * 0.5
    clims = (-scale, scale)

    # 循环渲染每一帧
    anim = @animate for (i, snap) in enumerate(snaps)
        heatmap(snap',
            title="VZ Wavefield (Step $(i * snapshot_interval))",
            xlabel="X (grid)", ylabel="Z (grid)",
            color=:seismic, clims=clims,
            aspect_ratio=:equal, legend=true, yflip=true
        )
    end

    # 导出视频
    mp4(anim, save_path, fps=fps)
    println("Wavefield video saved as $(save_path)")
end