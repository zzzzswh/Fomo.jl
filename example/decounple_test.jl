# example/comparison_experiment.jl
#
# ═══════════════════════════════════════════════════════════════════
# 实验：弹性波 vs 耦合 P-S 势场求解器对比
#
# 模型设计（3 个界面，各有不同物理含义）：
#   界面 1 (z≈120): 纯 VP 变化 → PP反射 ✓, PS转换 ✗
#   界面 2 (z≈220): 纯 VS 变化 → PP反射 ✓, PS转换 ✓
#   界面 3 (z≈320, 倾斜): VP+VS 同时变化 → PP反射 ✓, PS转换 ✓
#
# 对比内容：
#   1. 弹性波 Vz 道集（P+S 混合）
#   2. P 势场道集（仅含 P 波相关事件）
#   3. S 势场道集（仅含 PS 转换波）
#   4. 波场快照对比
# ═══════════════════════════════════════════════════════════════════

using CUDA
using Plots
using Printf
using Statistics

using Fomo

ENV["GKSwstype"] = "100"
gr()

println("="^60)
println("  弹性波 vs 耦合 P-S 势场求解器 对比实验")
println("="^60)

# ══════════════════════════════════════════════════════════════
# 1. 模型参数
# ══════════════════════════════════════════════════════════════

nx, nz = 600, 400            # 3000m × 2000m
dh = 5.0f0                   # 5m 网格
dt = 0.0005f0                # 0.5ms 时间步
nt = 5000                    # 2.5s 总时长
f0 = 15.0f0                  # 15Hz Ricker 子波
nbc = 100                    # 吸收边界层数

# ══════════════════════════════════════════════════════════════
# 2. 构建分层介质模型
# ══════════════════════════════════════════════════════════════

println("\n[1/4] 构建模型...")

# 背景模型
vp = fill(3000.0f0, nx, nz)
vs = fill(1500.0f0, nx, nz)
rho = fill(1.0f0, nx, nz)    # ρ=1 常密度（与 coupled2d 假设一致）

# ── 界面 1 (z≈120): 仅 VP 变化 ──
# 论文预测：P 势场有反射，S 势场无反射
z1 = 120
vp[:, z1:end] .= 3600.0f0    # VP +20%
# vs 不变

# ── 界面 2 (z≈220): 仅 VS 变化 ──
# 论文预测：P 势场有反射（极性反转），S 势场有 PS 转换
z2 = 220
vs[:, z2:end] .= 1900.0f0    # VS +27%
# vp 不变（此层 vp 已经是 3600）

# ── 界面 3 (z≈320, 倾斜): VP + VS 同时变化 ──
# 倾斜界面：从左边 z=300 到右边 z=340
# 论文预测：P 和 S 势场都有反射
for ix in 1:nx
    z3 = 300 + round(Int, 40.0 * (ix - 1) / (nx - 1))  # 300→340 倾斜
    vp[ix, z3:end] = 4500.0f0   # VP +25%
    vs[ix, z3:end] = 2400.0f0   # VS +26%
end

println("  模型尺寸: $(nx)×$(nz) = $(nx*dh)m × $(nz*dh)m")
println("  界面 1 (z=$(z1*dh)m): 纯 VP 扰动 (3000→3600 m/s)")
println("  界面 2 (z=$(z2*dh)m): 纯 VS 扰动 (1500→1900 m/s)")
println("  界面 3 (z=1500~1700m): VP+VS 扰动, 倾斜界面")

# ══════════════════════════════════════════════════════════════
# 3. 观测系统
# ══════════════════════════════════════════════════════════════

sx = [nx ÷ 2]                 # 震源在中心
sz = [5]                       # 近地表
rx = collect(1:3:nx)           # 检波器每 3 格一个
rz = fill(5, length(rx))      # 与震源同深度

snap_interval = 100            # 每 100 步保存快照

# ══════════════════════════════════════════════════════════════
# 4. 运行弹性波求解器
# ══════════════════════════════════════════════════════════════

println("\n[2/4] 运行弹性波求解器 (elastic2d)...")
seis_vx, seis_vz, snaps_el = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=nbc, fd_order=8, snap_interval=snap_interval)

# ══════════════════════════════════════════════════════════════
# 5. 运行耦合 P-S 势场求解器
# ══════════════════════════════════════════════════════════════

println("\n[3/4] 运行耦合 P-S 势场求解器 (coupled2d)...")
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=nbc, fd_order=8, snap_interval=snap_interval)

# ══════════════════════════════════════════════════════════════
# 6. 绘图对比
# ══════════════════════════════════════════════════════════════

println("\n[4/4] 绘制对比图...")

# ── 辅助函数 ──
function shot_heatmap(data, title_str; clim_pct=0.98)
    d = trace_norm(data, dims=2)
    scale = quantile(abs.(filter(!isnan, d[:])), clim_pct)
    scale = max(scale, eps(Float32))
    heatmap(d',
        title=title_str, c=:seismic, clims=(-scale, scale),
        xlabel="Receiver", ylabel="Time Sample",
        yflip=true, size=(600, 500),
        titlefontsize=11)
end

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1: 三道集对比 — 弹性波 Vz | P 势场 | S 势场
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

p1 = shot_heatmap(seis_vz, "Elastic Vz (P+S mixed)")
p2 = shot_heatmap(seis_P, "P Potential (PP only)")
p3 = shot_heatmap(seis_S, "S Potential (PS only)")

fig1 = plot(p1, p2, p3, layout=(1, 3), size=(1800, 550),
    plot_title="Seismogram Comparison: Elastic vs Coupled P-S",
    plot_titlefontsize=13)
savefig(fig1, "comparison_seismograms.png")
println("  ✓ comparison_seismograms.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2: P 和 S 势场分别展示（带标注）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P_n = trace_norm(seis_P, dims=2)
S_n = trace_norm(seis_S, dims=2)

scale_P = quantile(abs.(filter(!isnan, P_n[:])), 0.98)
scale_S = quantile(abs.(filter(!isnan, S_n[:])), 0.98)
scale_P = max(scale_P, eps(Float32))
scale_S = max(scale_S, eps(Float32))

p4 = heatmap(P_n', title="P Potential — PP Reflections",
    c=:seismic, clims=(-scale_P, scale_P),
    xlabel="Receiver", ylabel="Time Sample", yflip=true,
    titlefontsize=11)
# 标注界面位置（大致走时）
annotate!(p4, [
    (10, 240, text("PP1 (VP jump)", 8, :left, :yellow)),
    (10, 530, text("PP2 (VS jump)", 8, :left, :yellow)),
    (10, 900, text("PP3 (VP+VS, dipping)", 8, :left, :yellow)),
])

p5 = heatmap(S_n', title="S Potential — PS Conversions",
    c=:seismic, clims=(-scale_S, scale_S),
    xlabel="Receiver", ylabel="Time Sample", yflip=true,
    titlefontsize=11)
annotate!(p5, [
    (10, 240, text("← No PS here!", 8, :left, :lime)),
    (10, 750, text("PS2 (VS jump)", 8, :left, :yellow)),
    (10, 1300, text("PS3 (VP+VS, dipping)", 8, :left, :yellow)),
])

fig2 = plot(p4, p5, layout=(1, 2), size=(1300, 600),
    plot_title="Mode Separation: P vs S Potential",
    plot_titlefontsize=13)
savefig(fig2, "comparison_mode_separation.png")
println("  ✓ comparison_mode_separation.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3: 波场快照对比（选择合适的时间步）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function snap_heatmap(snap, title_str)
    scale = quantile(abs.(filter(!isnan, snap[:])), 0.99)
    scale = max(scale, eps(Float32))
    heatmap(snap', title=title_str, c=:seismic,
        clims=(-scale, scale),
        xlabel="X (grid)", ylabel="Z (grid)",
        yflip=true, aspect_ratio=:auto,
        titlefontsize=10)
end

# 选取 P 波刚过界面2、S 波正在传播的时刻
n_snaps = length(snaps_el)
snap_ids = [
    min(max(1, n_snaps ÷ 5), n_snaps),       # 早期
    min(max(1, 2 * n_snaps ÷ 5), n_snaps),   # 中期
    min(max(1, 3 * n_snaps ÷ 5), n_snaps),   # 晚期
]

for (k, si) in enumerate(snap_ids)
    t_sec = si * snap_interval * dt
    t_str = @sprintf("t = %.2fs", t_sec)

    pa = snap_heatmap(snaps_el[si], "Elastic Vz  ($t_str)")
    pb = snap_heatmap(snaps_P[si], "P Potential  ($t_str)")
    pc = snap_heatmap(snaps_S[si], "S Potential  ($t_str)")

    fig_snap = plot(pa, pb, pc, layout=(1, 3), size=(1800, 450),
        plot_title="Wavefield Snapshot Comparison (#$si)",
        plot_titlefontsize=12)
    fname = "comparison_snapshot_$(k).png"
    savefig(fig_snap, fname)
    println("  ✓ $fname")
end

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4: 单道波形对比（中间道）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mid_rec = length(rx) ÷ 2   # 中间道

# 归一化单道
norm1(x) = x ./ maximum(abs.(x) .+ eps(Float32))

trace_vz = norm1(seis_vz[mid_rec, :])
trace_P = norm1(seis_P[mid_rec, :])
trace_S = norm1(seis_S[mid_rec, :])

time_axis = (1:nt) .* dt

p_trace = plot(time_axis, trace_vz, label="Elastic Vz", color=:black, lw=1.2,
    xlabel="Time (s)", ylabel="Normalized Amplitude",
    title="Single Trace Comparison (mid-receiver)",
    titlefontsize=11, legend=:topright, size=(900, 400))
plot!(p_trace, time_axis, trace_P, label="P Potential", color=:blue, lw=1.0, alpha=0.8)
plot!(p_trace, time_axis, trace_S, label="S Potential", color=:red, lw=1.0, alpha=0.8)

savefig(p_trace, "comparison_single_trace.png")
println("  ✓ comparison_single_trace.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5: 速度模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pv1 = heatmap(vp', title="VP Model (m/s)", c=:viridis,
    xlabel="X (grid)", ylabel="Z (grid)", yflip=true)
pv2 = heatmap(vs', title="VS Model (m/s)", c=:viridis,
    xlabel="X (grid)", ylabel="Z (grid)", yflip=true)
fig_model = plot(pv1, pv2, layout=(1, 2), size=(1200, 400),
    plot_title="Velocity Model", plot_titlefontsize=12)
savefig(fig_model, "comparison_model.png")
println("  ✓ comparison_model.png")

# ══════════════════════════════════════════════════════════════
# 7. 结果说明
# ══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  实验完成！")
println("="^60)
println("""

输出文件：
  comparison_model.png           速度模型
  comparison_seismograms.png     三道集对比（弹性Vz | P势场 | S势场）
  comparison_mode_separation.png P/S 势场分离展示（带标注）
  comparison_single_trace.png    中间道波形叠合对比
  comparison_snapshot_1~3.png    波场快照对比

验证要点（参考 Li et al. 2018）：
  ✓ 界面 1（纯 VP 扰动）：P 势场有反射，S 势场无反射
  ✓ 界面 2（纯 VS 扰动）：P 和 S 势场都有反射
  ✓ 界面 3（VP+VS 倾斜）：P 和 S 势场都有反射
  ✓ 弹性波 Vz 中所有事件都能在 P 或 S 势场中找到对应
  ✓ P 势场中 VP 扰动的反射与 VS 扰动的反射极性相反
""")