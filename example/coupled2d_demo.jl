# example/coupled2d_demo.jl
#
# 复现 Li et al. (2018) Figure 1 的验证实验：
#   - 均匀弹性背景中放置 3 个平面散射体
#   - 散射体 1: 纯 VP 扰动 → 只产生 PP 反射，无模式转换
#   - 散射体 2: 纯 VS 扰动 → 产生 PP 反射 + PS 转换
#   - 散射体 3: VP+VS 扰动 → 产生 PP 反射 + PS 转换
#
# 验证：模式转换只在 VS 不连续处发生

using CUDA
using Plots
using Printf

using Fomo

# ── 参数 ──
nx, nz = 601, 601
dh = 5.0f0                      # 5 m 网格
dt = 0.0005f0                   # 0.5 ms 时间步
nt = 4000                       # 总时间 2.0 s
f0 = 15.0f0                    # 15 Hz Ricker 子波

# ── 均匀背景模型 ──
vp_bg = 3000.0f0                # 背景 VP = 3000 m/s
vs_bg = 1500.0f0                # 背景 VS = 1500 m/s

vp = fill(vp_bg, nx, nz)
vs = fill(vs_bg, nx, nz)

# ── 三个平面散射体 ──
# 散射体 1（z ≈ 150）: 纯 VP 扰动
z1 = 150
vp[:, z1:z1+2] .= 3300.0f0     # VP +10%
# vs 不变 → 不应产生 PS 转换

# 散射体 2（z ≈ 300）: 纯 VS 扰动
z2 = 300
vs[:, z2:z2+2] .= 1650.0f0     # VS +10%
# vp 不变 → 应产生 PS 转换

# 散射体 3（z ≈ 450）: VP + VS 扰动
z3 = 450
vp[:, z3:z3+2] .= 3300.0f0     # VP +10%
vs[:, z3:z3+2] .= 1650.0f0     # VS +10%
# 两者同时变化 → 应产生 PS 转换

# ── 观测系统 ──
sx = [nx ÷ 2]                  # 震源在中心
sz = [5]                        # 近地表
rx = collect(1:2:nx)            # 接收器覆盖全表面
rz = fill(5, length(rx))       # 与震源同深度

# ── RUN: 耦合 P-S 势场求解器 ──
@info "Running coupled P-S potential solver..."
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=100)

# ── 对比: 传统弹性波求解器 ──
rho = fill(1.0f0, nx, nz)      # ρ=1 常密度（与耦合方程假设一致）
@info "Running conventional elastic solver for comparison..."
seis_vx, seis_vz, snaps_el = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=100)

# ── 绘图: 地震记录对比 ──
seis_P_n = trace_norm(seis_P, dims=2)
seis_S_n = trace_norm(seis_S, dims=2)

p1 = heatmap(seis_P_n', title="P potential (coupled)", c=:seismic,
    xlabel="Receiver", ylabel="Time sample", clims=(-0.3, 0.3))
p2 = heatmap(seis_S_n', title="S potential (coupled)", c=:seismic,
    xlabel="Receiver", ylabel="Time sample", clims=(-0.3, 0.3))

plot(p1, p2, layout=(1, 2), size=(1200, 500))
savefig("coupled2d_seismograms.png")

# ── 绘图: 波场快照（如有）──
if length(snaps_P) > 0
    idx = min(20, length(snaps_P))
    p3 = heatmap(snaps_P[idx]', title="P wavefield (t=$(idx*100*dt)s)",
        c=:seismic, aspect_ratio=:equal)
    p4 = heatmap(snaps_S[idx]', title="S wavefield (t=$(idx*100*dt)s)",
        c=:seismic, aspect_ratio=:equal)
    plot(p3, p4, layout=(1, 2), size=(1200, 500))
    savefig("coupled2d_snapshots.png")
end

@info "Done! Check coupled2d_seismograms.png and coupled2d_snapshots.png"
@info """
验证要点（参考 Li et al. 2018, Figure 2）：
  1. P 记录应有 3 个反射事件（PP1, PP2, PP3）
  2. S 记录应只有 2 个强反射事件（PS2, PS3）
     → 散射体 1（纯 VP 扰动）不产生 PS 转换
  3. PP2 与 PP1/PP3 极性相反（VS 扰动对 P 反射的贡献为负）
"""
