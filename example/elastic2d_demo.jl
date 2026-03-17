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

# ── 速度模型 ──
vp = fill(2500.0f0, nx, nz)
vs = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)
vp[:, 1] .= 0.0f0
vs[:, 1] .= 0.0f0
rho[:, 1] .= 0.0f0

# ── 观测系统 ──
sx = [nx ÷ 2]
sz = [10]
rx = collect(1:2:nx)
rz = fill(10, length(rx))

# ── RUN ──
vx, vz, snaps = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# ── Plots ──
vz = trace_norm(vz, dims=2)
plot_shot(vz, "vz.png")
plot_wavefield_video(snaps, 50, "wavefield.mp4", fps=10, adaptive_clims=true)