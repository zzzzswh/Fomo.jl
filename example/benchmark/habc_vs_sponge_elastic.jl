# =====================================================================
#  HABC vs Cerjan sponge on elastic2d — absorbing-boundary comparison
#
#  同一份 elastic2d 代码、同一份模型、同一个 nbc，仅切换 boundary kwarg。
#  均匀介质 + 中心爆炸源：P 波和 P→S 转换都会到达边界 → 比 acoustic 更严苛。
#
#  跑：
#      cd Fomo.jl
#      julia --project=. example/benchmark/habc_vs_sponge_elastic.jl
# =====================================================================

ENV["GKSwstype"] = "100"

using Fomo, CUDA, Plots, Statistics, Printf

# ── 模型 ─────────────────────────────────────────────────────────────
const NX, NZ = 401, 401
const DH = 10.0f0
const DT = 0.001f0
const NT = 2500           # 2.5 s — 给 S 波足够时间撞边界往回反射
const F0 = 15.0f0
const VP = 2500.0f0
const VS = 1500.0f0
const RHO = 2000.0f0
const NBC = 50
const FDORD = 8
const SNAP_INTERVAL = 50      # 50 帧

const SX, SZ = [NX ÷ 2], [NZ ÷ 2]   # 震源放正中
const RX = collect(1:2:NX)
const RZ = fill(NZ ÷ 2, length(RX))

const OUTDIR = joinpath(@__DIR__, "output")
mkpath(OUTDIR)
out(s) = joinpath(OUTDIR, s)

vp = fill(VP, NX, NZ)
vs = fill(VS, NX, NZ)
rho = fill(RHO, NX, NZ)

# ── 跑两次 ────────────────────────────────────────────────────────────

println("─── HABC ───")
t_habc = @elapsed _, _, snaps_habc = elastic2d(
    vp, vs, rho, DH, DT, NT, F0;
    sx=SX, sz=SZ, rx=RX, rz=RZ,
    nbc=NBC, fd_order=FDORD,
    snap_interval=SNAP_INTERVAL, boundary=:habc)

println("─── Sponge (Cerjan, factor=0.015) ───")
t_sp = @elapsed _, _, snaps_sp = elastic2d(
    vp, vs, rho, DH, DT, NT, F0;
    sx=SX, sz=SZ, rx=RX, rz=RZ,
    nbc=NBC, fd_order=FDORD,
    snap_interval=SNAP_INTERVAL, boundary=:sponge)

println("\nTimings (full run, includes warmup):")
println("  HABC:   $(round(t_habc, digits=3)) s")
println("  Sponge: $(round(t_sp,   digits=3)) s")

# ── (1) Snapshot 网格：4 时刻 × {HABC, Sponge}
#       共享 clims = 0.6 × max(|sponge|,|HABC|)，让两边亮度可比，
#       并让中等振幅的残余反射清晰可见。
#       title 里标 max 幅度，量化差距透明。 ─────────────────────────────
println("\nPlotting snapshots…")
nframes = min(length(snaps_habc), length(snaps_sp))
idxs = round.(Int, range(nframes ÷ 4, nframes; length=4))

function panel(snap, label, t_ms, clims)
    peak = Float32(maximum(abs.(snap)))
    heatmap(snap',
        title="$label  t=$(t_ms) ms  (max=$(@sprintf("%.1e", peak)))",
        color=:seismic, clims=clims,
        yflip=true, aspect_ratio=:equal,
        titlefontsize=8, colorbar=false, xlabel="", ylabel="")
end

plots_arr = []
for k in idxs
    # 用绝对最大值做 clims，0.6 因子让最亮处略饱和、中等亮度可见
    peak_k = Float32(max(
        maximum(abs.(snaps_habc[k])),
        maximum(abs.(snaps_sp[k])),
    ))
    peak_k = max(peak_k, eps(Float32))
    clims_k = (-0.6f0 * peak_k, 0.6f0 * peak_k)

    t_ms = round(k * SNAP_INTERVAL * DT * 1000, digits=0)
    push!(plots_arr, panel(snaps_habc[k], "HABC", t_ms, clims_k))
    push!(plots_arr, panel(snaps_sp[k], "Sponge", t_ms, clims_k))
end

savefig(plot(plots_arr...; layout=(4, 2), size=(900, 1500),
        plot_title="Wavefield snapshots (vz) — HABC (left) vs Cerjan sponge (right), nb=$NBC, shared clims per row"),
    out("snapshots_grid_elastic.png"))

# ── (2) 内域能量曲线 ──────────────────────────────────────────────────
println("Plotting energy curve…")

# 去掉边界层范围，只看"实际计算域"
interior_energy(snaps, margin) =
    [sum(@views (s[margin+1:end-margin, margin+1:end-margin]) .^ 2)
     for s in snaps]

e_h = interior_energy(snaps_habc[1:nframes], NBC)
e_s = interior_energy(snaps_sp[1:nframes], NBC)
# 用各自的峰值归一化，便于在同一张图上对比形状
e_h_n = e_h ./ maximum(e_h)
e_s_n = e_s ./ maximum(e_s)

t_axis = (1:nframes) .* SNAP_INTERVAL .* DT

p_lin = plot(t_axis, e_h_n, lw=2, color=:blue, label="HABC",
    xlabel="Time (s)", ylabel="∑ vz² (interior, normalized)",
    title="Interior energy decay — linear",
    legend=:topright, titlefontsize=10)
plot!(p_lin, t_axis, e_s_n, lw=2, color=:red, label="Cerjan sponge")

p_log = plot(t_axis, max.(e_h_n, 1e-8), lw=2, color=:blue, label="HABC",
    yscale=:log10, xlabel="Time (s)", ylabel="∑ vz²  (log)",
    title="Same, log scale — lower late-time floor = less boundary leakage",
    legend=:topright, titlefontsize=10)
plot!(p_log, t_axis, max.(e_s_n, 1e-8), lw=2, color=:red, label="Cerjan sponge")

savefig(plot(p_lin, p_log; layout=(2, 1), size=(900, 800)),
    out("energy_curve_elastic.png"))

# ── (3) 末段能量比 + 报告 ─────────────────────────────────────────────
tail = (3*nframes÷4):nframes
ratio = mean(e_s_n[tail]) / max(mean(e_h_n[tail]), 1e-12)

report = """
HABC vs Cerjan sponge — elastic2d, $(NX)×$(NZ), nbc=$NBC, fd_order=$FDORD
==================================================================
Model:    homogeneous vp=$(VP) m/s, vs=$(VS) m/s, rho=$(RHO) kg/m³
Source:   center explosive (τxx + τzz), Ricker $(F0) Hz
Time:     nt=$NT, dt=$(DT*1000) ms  (T=$(NT*DT) s)

Wall-clock (full run, includes warmup):
  HABC    : $(@sprintf("%.3f", t_habc)) s
  Sponge  : $(@sprintf("%.3f", t_sp))   s
  Δ       : $(@sprintf("%+.1f %%", 100*(t_habc - t_sp)/t_sp))

Late-time interior energy (last 25 %, normalized to each run's peak):
  HABC    : $(@sprintf("%.3e", mean(e_h_n[tail])))
  Sponge  : $(@sprintf("%.3e", mean(e_s_n[tail])))
  Ratio   : $(@sprintf("%.1fx", ratio))   (>1 → sponge leaks more)

Outputs in $OUTDIR :
  - snapshots_grid_elastic.png
  - energy_curve_elastic.png
"""

write(out("report_elastic.txt"), report)
println("\n" * report)