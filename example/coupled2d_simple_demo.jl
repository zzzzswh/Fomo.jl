# example/coupled2d_simple_demo.jl
#
# 最简耦合 P-S 势场求解器示例
# 对标 elastic2d_demo.jl 的模型设置

using CUDA
using Plots
using Printf

using Fomo

# ── 参数 ──
nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# ── 双层速度模型 ──
vp = fill(2500.0f0, nx, nz)
vs = fill(1200.0f0, nx, nz)
vp[:, 150:end] .= 3500.0f0
vs[:, 150:end] .= 1800.0f0

# ── 观测系统 ──
sx = [nx ÷ 2]
sz = [10]
rx = collect(1:2:nx)
rz = fill(10, length(rx))

# ── RUN ──
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# ── Plots ──
seis_P = trace_norm(seis_P, dims=2)
seis_S = trace_norm(seis_S, dims=2)

plot_shot(seis_P, "P_potential.png")
plot_shot(seis_S, "S_potential.png")

if length(snaps_P) > 0
    plot_wavefield_video(snaps_P, 50, "P_wavefield.mp4", fps=10, adaptive_clims=true)
    plot_wavefield_video(snaps_S, 50, "S_wavefield.mp4", fps=10, adaptive_clims=true)
end
