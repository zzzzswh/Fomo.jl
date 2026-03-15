using Plots
using Statistics

# 设置无头模式（静默运行，防止绘图时弹出窗口）
ENV["GKSwstype"] = "100"
gr()

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
function plot_shot(shot::AbstractMatrix{<:Real}, save_path::AbstractString)
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