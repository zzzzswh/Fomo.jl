# example/comparison_experiment.jl
#
# ═══════════════════════════════════════════════════════════════════
# 实验：弹性波 vs 耦合 P-S 势场求解器对比 + 波场分离重建
#
# 核心验证：
#   1. P/S 势场的模式分离（界面选择性）
#   2. 从 P+S 势场精确重建粒子速度 vz（Helmholtz 分解 + FFT）
#   3. 波场分离: vz = vz_P + vz_S
# ═══════════════════════════════════════════════════════════════════

using CUDA
using Plots
using Printf
using Statistics
using FFTW          # FFT: Pkg.add("FFTW")

using Fomo

ENV["GKSwstype"] = "100"
gr()

# ══════════════════════════════════════════════════════════════
# 0. 输出目录 & 辅助函数
# ══════════════════════════════════════════════════════════════

const OUTDIR = "output_comparison"
mkpath(OUTDIR)
outpath(name) = joinpath(OUTDIR, name)

println("="^60)
println("  弹性波 vs 耦合 P-S 势场 对比 + 波场分离重建")
println("  输出目录: $(OUTDIR)/")
println("="^60)

function safe_scale(data, pct=0.98)
    valid = filter(x -> isfinite(x) && x != 0, data[:])
    isempty(valid) ? 1.0f0 : max(Float32(quantile(abs.(valid), pct)), eps(Float32))
end

safe_norm1(x) = (m = maximum(abs.(x)); m > eps(Float32) ? x ./ m : zero(x))

function clean_nan!(d)
    d[isnan.(d)] .= 0.0f0
    return d
end

function record_video(snaps, snap_interval, filepath, title_prefix;
    fps=10, adaptive_clims=true)
    isempty(snaps) && (println("  ⚠ 无快照，跳过 $filepath");
    return)
    global_max = adaptive_clims ? 0.0f0 : maximum(maximum.(abs, snaps))
    global_scale = max(global_max * 0.5f0, 1f-10)
    global_clims = (-global_scale, global_scale)
    anim = @animate for (i, snap) in enumerate(snaps)
        s = clean_nan!(copy(Float32.(snap)))
        cl = if adaptive_clims
            sc = max(maximum(abs, s) * 0.5f0, 1f-10)
            (-sc, sc)
        else
            global_clims
        end
        heatmap(s', title="$title_prefix (Step $(i * snap_interval))",
            xlabel="X (grid)", ylabel="Z (grid)",
            color=:seismic, clims=cl, aspect_ratio=:equal, yflip=true)
    end
    mp4(anim, filepath, fps=fps)
    println("  ✓ $filepath")
end

# ──────────────────────────────────────────────────────────────
# ⭐ 核心公式: 从 P/S 势场重建粒子速度（Helmholtz + FFT）
#
#   Ṗ = ∇·v,  Ṡ = (∇×v)_y
#
#   v = ∇φ̇ + ∇×ψ̇  其中  Ṗ = ∇²φ̇,  Ṡ = -∇²ψ̇
#
#   Fourier 域:
#     φ̇ˆ = -Ṗˆ / k²
#     ψ̇ˆ =  Ṡˆ / k²
#     v̂z = ikz·φ̇ˆ + ikx·ψ̇ˆ
#        = i·(kx·Ṡˆ - kz·Ṗˆ) / k²
#
#   分离:
#     v̂z_P = -i·kz·Ṗˆ / k²    (P波对vz的贡献)
#     v̂z_S =  i·kx·Ṡˆ / k²    (S波对vz的贡献)
#     vz = vz_P + vz_S           (精确)
# ──────────────────────────────────────────────────────────────

function reconstruct_vz(Pdot::Matrix, Sdot::Matrix, dh::Real)
    nx, nz = size(Pdot)

    # 余弦锥削（边缘 5% 渐变到零，消除 FFT 周期性假设的边界跳变）
    taper_x = ones(Float64, nx)
    taper_z = ones(Float64, nz)
    tw = max(1, round(Int, 0.05 * nx))
    for i in 1:tw
        w = 0.5 * (1 - cos(π * (i - 1) / tw))
        taper_x[i] = w
        taper_x[nx-i+1] = w
    end
    tw = max(1, round(Int, 0.05 * nz))
    for j in 1:tw
        w = 0.5 * (1 - cos(π * (j - 1) / tw))
        taper_z[j] = w
        taper_z[nz-j+1] = w
    end
    taper2d = [taper_x[i] * taper_z[j] for i in 1:nx, j in 1:nz]

    Pdot_t = Float64.(Pdot) .* taper2d
    Sdot_t = Float64.(Sdot) .* taper2d

    # 波数网格
    kx = 2π * fftfreq(nx, 1.0 / dh)
    kz = 2π * fftfreq(nz, 1.0 / dh)
    KX = [kx[i] for i in 1:nx, j in 1:nz]
    KZ = [kz[j] for i in 1:nx, j in 1:nz]
    K2 = KX .^ 2 .+ KZ .^ 2

    # 正则化: 用 k²/(k⁴+ε) 代替 1/k²，压制低波数放大
    k_min = 2π / (max(nx, nz) * dh)   # 最低有效波数
    eps_reg = (2.0 * k_min)^4          # 正则化参数
    inv_K2 = K2 ./ (K2 .^ 2 .+ eps_reg)

    # FFT + Helmholtz 重建
    Pdot_hat = fft(Pdot_t)
    Sdot_hat = fft(Sdot_t)

    vz_P_hat = -im .* KZ .* Pdot_hat .* inv_K2
    vz_S_hat = im .* KX .* Sdot_hat .* inv_K2

    vz_P = Float32.(real(ifft(vz_P_hat)))
    vz_S = Float32.(real(ifft(vz_S_hat)))
    vz = vz_P .+ vz_S

    return vz, vz_P, vz_S
end

"""
从连续三个快照估计时间导数: ḟ ≈ (f_next - f_prev) / (2Δt)
"""
function estimate_time_deriv(snap_prev, snap_next, dt_snap)
    return Float32.((snap_next .- snap_prev) ./ (2.0 * dt_snap))
end

# ══════════════════════════════════════════════════════════════
# 1. 模型
# ══════════════════════════════════════════════════════════════

nx, nz = 600, 400
dh = 5.0f0
dt = 0.0003f0
nt = 8000
f0 = 15.0f0
nbc = 100

println("\n[1/7] 构建模型...")

vp = fill(3000.0f0, nx, nz)
vs = fill(1500.0f0, nx, nz)
rho = fill(1.0f0, nx, nz)

z1 = 120;
vp[:, z1:end] .= 3600.0f0;
z2 = 220;
vs[:, z2:end] .= 1900.0f0;
for ix in 1:nx
    z3 = 300 + round(Int, 40.0 * (ix - 1) / (nx - 1))
    vp[ix, z3:end] .= 4500.0f0
    vs[ix, z3:end] .= 2400.0f0
end

println("  界面1: 纯VP 3000→3600 | 界面2: 纯VS 1500→1900 | 界面3: VP+VS 倾斜")

# ══════════════════════════════════════════════════════════════
# 2. 观测系统
# ══════════════════════════════════════════════════════════════

sx = [nx ÷ 2];
sz = [15];
rx = collect(1:3:nx);
rz = fill(15, length(rx));
snap_interval = 50

# ══════════════════════════════════════════════════════════════
# 3. 运行 elastic2d
# ══════════════════════════════════════════════════════════════

println("\n[2/7] 运行 elastic2d...")
seis_vx, seis_vz, snaps_el = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=nbc, fd_order=8, snap_interval=snap_interval)

# ══════════════════════════════════════════════════════════════
# 4. 运行 coupled2d
# ══════════════════════════════════════════════════════════════

println("\n[3/7] 运行 coupled2d...")
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=nbc, fd_order=8, snap_interval=snap_interval)

# ══════════════════════════════════════════════════════════════
# 5. 视频
# ══════════════════════════════════════════════════════════════

println("\n[4/7] 录制视频...")
record_video(snaps_el, snap_interval, outpath("wavefield_elastic_vz.mp4"),
    "Elastic Vz"; fps=15, adaptive_clims=true)
record_video(snaps_P, snap_interval, outpath("wavefield_P_potential.mp4"),
    "P Potential"; fps=15, adaptive_clims=true)
record_video(snaps_S, snap_interval, outpath("wavefield_S_potential.mp4"),
    "S Potential"; fps=15, adaptive_clims=true)

# ══════════════════════════════════════════════════════════════
# 6. ⭐ 波场重建与分离
# ══════════════════════════════════════════════════════════════

println("\n[5/7] 波场重建: vz = vz_P + vz_S (Helmholtz + FFT)...")

n_snaps = min(length(snaps_el), length(snaps_P), length(snaps_S))
dt_snap = Float64(snap_interval * dt)

# 选3个代表性时刻做重建对比
recon_indices = [
    max(2, n_snaps ÷ 5),
    max(2, 2 * n_snaps ÷ 5),
    max(2, 3 * n_snaps ÷ 5),
]

function snap_heatmap(snap, title_str; cmap=:seismic)
    s = clean_nan!(copy(Float32.(snap)))
    scale = safe_scale(s, 0.99)
    if cmap == :seismic
        heatmap(s', title=title_str, c=cmap, clims=(-scale, scale),
            xlabel="X", ylabel="Z", yflip=true, aspect_ratio=:auto, titlefontsize=9)
    else
        heatmap(s', title=title_str, c=cmap, clims=(0, scale),
            xlabel="X", ylabel="Z", yflip=true, aspect_ratio=:auto, titlefontsize=9)
    end
end

for (k, si) in enumerate(recon_indices)
    si_prev = si - 1
    si_next = si + 1
    if si_next > n_snaps
        continue
    end

    t_sec = si * snap_interval * dt
    t_str = @sprintf("t=%.2fs", t_sec)

    # 时间导数
    Pdot = estimate_time_deriv(snaps_P[si_prev], snaps_P[si_next], dt_snap)
    Sdot = estimate_time_deriv(snaps_S[si_prev], snaps_S[si_next], dt_snap)

    # Helmholtz 重建
    vz_recon, vz_P, vz_S = reconstruct_vz(Pdot, Sdot, Float64(dh))

    # 弹性波参考
    vz_ref = Float32.(snaps_el[si])

    # 残差
    residual = vz_recon .- vz_ref
    ref_max = maximum(abs.(vz_ref))
    res_max = maximum(abs.(residual))
    rel_err = ref_max > 0 ? res_max / ref_max : 0.0
    println("  Snapshot #$si ($t_str): max|residual|/max|vz| = $(round(rel_err*100, digits=2))%")

    # ── 图: 重建对比（6 panel）──
    pa = snap_heatmap(vz_ref, "Elastic Vz ($t_str)")
    pb = snap_heatmap(vz_recon, "Reconstructed vz ($t_str)")
    pc = snap_heatmap(residual, "Residual ($(round(rel_err*100,digits=1))%)")
    pd = snap_heatmap(vz_P, "vz_P (P-wave part)")
    pe = snap_heatmap(vz_S, "vz_S (S-wave part)")
    pf = snap_heatmap(abs.(vz_P) .+ abs.(vz_S), "|vz_P|+|vz_S|"; cmap=:hot)

    fig = plot(pa, pb, pc, pd, pe, pf, layout=(2, 3), size=(1800, 900),
        plot_title="Wavefield Reconstruction & Separation (#$si, $t_str)")
    savefig(fig, outpath("reconstruction_$(k).png"))
    println("  ✓ reconstruction_$(k).png")
end

# 录制分离视频
println("\n  录制波场分离视频...")
recon_vz_P_snaps = Vector{Matrix{Float32}}()
recon_vz_S_snaps = Vector{Matrix{Float32}}()

for si in 2:n_snaps-1
    Pdot = estimate_time_deriv(snaps_P[si-1], snaps_P[si+1], dt_snap)
    Sdot = estimate_time_deriv(snaps_S[si-1], snaps_S[si+1], dt_snap)
    _, vz_P, vz_S = reconstruct_vz(Pdot, Sdot, Float64(dh))
    push!(recon_vz_P_snaps, vz_P)
    push!(recon_vz_S_snaps, vz_S)
end

record_video(recon_vz_P_snaps, snap_interval, outpath("wavefield_vz_P_separated.mp4"),
    "vz_P (P-wave)"; fps=15, adaptive_clims=true)
record_video(recon_vz_S_snaps, snap_interval, outpath("wavefield_vz_S_separated.mp4"),
    "vz_S (S-wave)"; fps=15, adaptive_clims=true)

# ══════════════════════════════════════════════════════════════
# 7. 静态对比图
# ══════════════════════════════════════════════════════════════

println("\n[6/7] 绘制对比图...")

function shot_heatmap(data, title_str; clim_pct=0.98)
    d = clean_nan!(trace_norm(data, dims=2))
    scale = safe_scale(d, clim_pct)
    heatmap(d', title=title_str, c=:seismic, clims=(-scale, scale),
        xlabel="Receiver", ylabel="Time Sample",
        yflip=true, size=(600, 500), titlefontsize=11)
end

# 速度模型
pv1 = heatmap(vp', title="VP (m/s)", c=:viridis, xlabel="X", ylabel="Z", yflip=true)
pv2 = heatmap(vs', title="VS (m/s)", c=:viridis, xlabel="X", ylabel="Z", yflip=true)
savefig(plot(pv1, pv2, layout=(1, 2), size=(1200, 400),
        plot_title="Velocity Model"), outpath("model.png"))
println("  ✓ model.png")

# 三道集
p1 = shot_heatmap(seis_vz, "Elastic Vz (P+S mixed)")
p2 = shot_heatmap(seis_P, "P Potential")
p3 = shot_heatmap(seis_S, "S Potential")
savefig(plot(p1, p2, p3, layout=(1, 3), size=(1800, 550),
        plot_title="Seismogram Comparison"), outpath("seismograms.png"))
println("  ✓ seismograms.png")

# P/S 分离
P_n = clean_nan!(trace_norm(seis_P, dims=2))
S_n = clean_nan!(trace_norm(seis_S, dims=2))
p4 = heatmap(P_n', title="P Potential — PP Reflections",
    c=:seismic, clims=(-safe_scale(P_n), safe_scale(P_n)),
    xlabel="Receiver", ylabel="Time", yflip=true, titlefontsize=11)
p5 = heatmap(S_n', title="S Potential — PS Conversions",
    c=:seismic, clims=(-safe_scale(S_n), safe_scale(S_n)),
    xlabel="Receiver", ylabel="Time", yflip=true, titlefontsize=11)
savefig(plot(p4, p5, layout=(1, 2), size=(1300, 600),
        plot_title="Mode Separation: P vs S"), outpath("mode_separation.png"))
println("  ✓ mode_separation.png")

# 单道波形
mid = length(rx) ÷ 2
t = (1:nt) .* dt
pt = plot(t, safe_norm1(seis_vz[mid, :]), label="Elastic Vz",
    color=:black, lw=1.2, xlabel="Time (s)", ylabel="Amp.",
    title="Single Trace", legend=:topright, size=(900, 400))
plot!(pt, t, safe_norm1(seis_P[mid, :]), label="P Potential", color=:blue, lw=1.0, alpha=0.8)
plot!(pt, t, safe_norm1(seis_S[mid, :]), label="S Potential", color=:red, lw=1.0, alpha=0.8)
savefig(pt, outpath("single_trace.png"))
println("  ✓ single_trace.png")

# ══════════════════════════════════════════════════════════════
# 8. 完成
# ══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  实验完成！")
println("="^60)
println("""
$(OUTDIR)/ 内容:

  模型 & 道集:
    model.png, seismograms.png, mode_separation.png, single_trace.png

  ⭐ 波场重建 & 分离:
    reconstruction_1~3.png   6-panel: Elastic | Reconstructed | Residual
                                      vz_P    | vz_S          | |vz_P|+|vz_S|

  视频:
    wavefield_elastic_vz.mp4      弹性波 Vz（参考）
    wavefield_P_potential.mp4     P 势场
    wavefield_S_potential.mp4     S 势场
    wavefield_vz_P_separated.mp4  ⭐ 分离出的 vz_P（纯 P 波粒子速度）
    wavefield_vz_S_separated.mp4  ⭐ 分离出的 vz_S（纯 S 波粒子速度）

  公式:
    vz = vz_P + vz_S  (精确, 通过 Helmholtz 分解 + FFT Poisson 求解)
    v̂z_P = -i·kz·Ṗ / k²     (P 波对 vz 的贡献)
    v̂z_S =  i·kx·Ṡ / k²     (S 波对 vz 的贡献)
""")