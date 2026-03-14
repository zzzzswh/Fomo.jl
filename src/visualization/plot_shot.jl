using Plots
using Statistics

"""
    plot_shot(shot, save_path)
"""
function plot_shot(shot, save_path)
    # 1. 发现 NaN，报个信，但绝不罢工
    has_nan = any(isnan.(shot))
    if has_nan
        println(">>> ⚠️ 警告：检测到 NaN！正在强行绘图，帮你看看是在哪里爆炸的...")
    end

    # 2. 把正常的数字挑出来，仅仅是为了计算合适的色阶
    valid_data = filter(!isnan, shot)

    # 如果连一个正常数字都没有（比如第一步就炸了），那就真画不出来了
    if isempty(valid_data)
        println(">>> ❌ 彻底没救了：全屏全是 NaN，一个正常数字都没有，画不了图！")
        return
    end

    scale = quantile(abs.(valid_data), 0.98)
    scale = scale == 0.0f0 ? eps(Float32) : scale

    # 3. 强行把原始数据 shot' (含 NaN) 扔给 heatmap 画图！
    # NaN 区域通常会显示为空白或超出色带的颜色
    p = heatmap(shot',
        title=has_nan ? "Shot Record (EXPLODED)" : "Shot Record",
        xlabel="Receiver Number",
        ylabel="Time Step",
        color=:seismic,
        clims=(-scale, scale),
        yflip=true,
        colorbar=true,
        size=(800, 600)
    )

    savefig(p, save_path)
    println(">>> 成功保存图纸至: $save_path")
end