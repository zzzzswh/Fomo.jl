# example/make_wavefield_video.jl — 波场快照 → MP4 / GIF
#
# 一次性准备（Plots 装到全局默认环境，不污染 Fomo 的 Project.toml）:
#   julia -e 'using Pkg; Pkg.add("Plots")'
# 运行:
#   julia --project=. example/make_wavefield_video.jl

using Fomo, CUDA
using Plots        # ← 触发 FomoPlotsExt 扩展加载，plot_wavefield_video 才可用

# ── 模型：两层曲界面 + 高斯高速异常体 ──
nx, nz = 301, 201
dh, dt, nt, f0 = 10.0f0, 1.0f-3, 1500, 10.0f0
vp = fill(2000.0f0, nx, nz)
for i in 1:nx
    zint = 100 + round(Int, 20 * cos(2π * (i - 151) / nx))
    vp[i, zint:end] .= 3000.0f0
end
xg = ((1:nx) .- 151) ./ 30
zg = ((1:nz)' .- 60) ./ 15
vp .+= 400.0f0 .* Float32.(exp.(-(xg .^ 2 .+ zg .^ 2)))
vs  = vp ./ 1.73f0
rho = Float32.(310.0 .* Float64.(vp) .^ 0.25)

# ── 正演：每 10 步存一帧快照（elastic 快照是 vz，能同时看到 P/S/转换波）──
snap_every = 10
res = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx=[151], sz=[5],
    rx=collect(1:4:nx), rz=fill(5, length(1:4:nx)),
    boundary=:habc, nbc=50, snap_interval=snap_every)

# ── MP4（150 帧 / fps=15 → 10 秒；adaptive_clims=true 每帧自适应色阶，传播过程更清楚）──
plot_wavefield_video(res.snaps, snap_every, "wavefield_vz.mp4"; fps=15, adaptive_clims=true)

# ── 想要 GIF 的话，取消注释（GIF 体积大，帧数多时建议先抽稀）──
# c = maximum(maximum.(abs, res.snaps)) * 0.3
# anim = @animate for (i, s) in enumerate(res.snaps)
#     heatmap(s', title="VZ  step $(i * snap_every)",
#         color=:seismic, clims=(-c, c), yflip=true,
#         aspect_ratio=:equal, xlabel="X (grid)", ylabel="Z (grid)")
# end
# gif(anim, "wavefield_vz.gif", fps=15)

println("GPU 用时: $(round(res.stats.kernel_time_s, digits=3)) s")
