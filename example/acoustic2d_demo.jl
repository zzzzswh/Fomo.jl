# example/acoustic_demo.jl

using CUDA
using Plots
using Printf

include("../src/Fomo.jl")
using .Fomo

# ── 参数 ──
nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# ── 速度模型（声波：只需要 vp 和 rho）──
vp = fill(2500.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# 真空自由表面
vp[:, 1] .= 0.0f0
rho[:, 1] .= 0.0f0

# ── 观测系统 ──
sx = [nx ÷ 2]
sz = [10]
rx = collect(1:2:nx)
rz = fill(10, length(rx))

# ── CFL check ──
v_max = maximum(vp)
coef_sum_8th = 1.19625f0
dt_max = dh / (v_max * sqrt(2.0f0) * coef_sum_8th)
@printf("  CFL: dt=%.6f, dt_max=%.6f\n", dt, dt_max)
dt > dt_max && error("CFL violation!")

# ── RUN ──
vx, vz, snaps = acoustic2d(vp, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# ── Plots ──
vz_norm = trace_norm(vz, dims=2)
plot_shot(vz_norm, "acoustic_vz.png")

plot_wavefield_video(snaps, 50, "acoustic_wavefield.mp4",
    fps=10, adaptive_clims=true)