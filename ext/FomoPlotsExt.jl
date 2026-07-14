# ext/FomoPlotsExt.jl
#
# Plots 可视化扩展：仅当用户 `using Plots` 时由 Pkg 自动加载，
# 使 `using Fomo` 本体不再依赖 Plots/GR（显著缩短 TTFX / 冷启动）。
# 原 src/visualization/{plot_shot.jl, plot_video.jl} 的实现原样搬入，
# 仅函数名加 `Fomo.` 前缀、无头模式设置移入 __init__。

module FomoPlotsExt

using Fomo
using Plots
using Statistics

function __init__()
    # 无头模式（静默运行，防止绘图弹窗；get! 不覆盖用户已有设置）
    get!(ENV, "GKSwstype", "100")
    gr()
    return nothing
end

"""
    plot_shot(shot::AbstractMatrix{<:Real}, save_path::AbstractString)

Plot a seismic shot record as a heatmap and save it to the specified path. 
Gracefully handles `NaN` values and executes in headless mode.

# Arguments
- `shot::AbstractMatrix{<:Real}`: The 2D array representing the shot record.
- `save_path::AbstractString`: The destination file path for the saved image.

# Example
```julia
shot_data = randn(Float32, 1000, 100)
plot_shot(shot_data, "output/shot_record.png")
"""
function Fomo.plot_shot(shot::AbstractMatrix{<:Real}, save_path::AbstractString)
    # 1. 检测 NaN
    has_nan = any(isnan, shot)
    if has_nan
        @warn "NaN values detected in the shot record. Plotting with available data to identify issues..."
    end

    # 2. 提取有效数据以计算色阶
    valid_data = filter(!isnan, shot)
    if isempty(valid_data)
        @error "The shot record contains only NaNs. Plotting aborted."
        return
    end

    # 动态计算色阶，并保持类型稳定
    scale = quantile(abs.(valid_data), 0.98)
    scale = iszero(scale) ? eps(typeof(scale)) : scale

    # 3. 绘制并保存热力图
    # 注意: 此处使用 shot'，假设输入矩阵为 (Time, Receiver) 以匹配常规视觉习惯
    p = heatmap(
        shot',
        title=has_nan ? "Shot Record (NaNs Detected)" : "Shot Record",
        xlabel="Receiver Number",
        ylabel="Time Step",
        color=:seismic,
        clims=(-scale, scale),
        yflip=true,
        colorbar=true,
        size=(800, 600)
    )

    savefig(p, save_path)
    @info "Shot record successfully saved to: $save_path"
end

"""
    plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10, adaptive_clims=false)

Create a video from wavefield snapshots without popping up windows.
If adaptive_clims is true, each frame will have its own color limits based on its maximum value.
If adaptive_clims is false (default), global color limits are used for all frames to avoid flickering.
"""
function Fomo.plot_wavefield_video(snaps::Vector{Matrix{Float32}}, snapshot_interval::Int, save_path::String="wavefield.mp4"; fps=10, adaptive_clims=false)
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

end  # module FomoPlotsExt
