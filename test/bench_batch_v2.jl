# test/bench_batch_v2.jl — 批处理吞吐基准 v2(需 GPU)
#
# 修正 verify_batch.jl 第 3 段的测量缺陷:
#   - 每个测点重复 5 次取【中位数】,并报告波动幅度 (max−min)/median;
#   - 单炮基线每个 (方程, 网格) 只测一组(det 与 graph 各一),不逐行重测;
#   - 加入 graph 单炮作为"最强单炮"基线,批吞吐与它对比才公平;
#   - Windows/WDDM 下桌面合成与计算共享 GPU,若波动仍 >15%,
#     建议关闭其他占卡程序或多跑几遍取稳定值。
#
# 用法:julia --project=. test/bench_batch_v2.jl
#
using CUDA
using Printf
using Statistics
using StaticArrays
using Fomo
using Fomo: init_acoustic_medium, AcousticWavefield, init_medium, Wavefield,
    init_habc, init_source, init_receiver, get_fd_coefficients, ricker_wavelet, reset!,
    _acoustic2d_loop_fused!, _elastic2d_loop_fused!,
    _acoustic2d_loop_graph!, _elastic2d_loop_graph!,
    BatchedAcousticWavefield, BatchedWavefield, init_batched_source,
    _acoustic2d_loop_batch!, _elastic2d_loop_batch!

const REPS = 5

"重复 REPS 次计时,返回 (median, spread%)。f 接受步数。"
function timed(f, nt)
    f(3)                      # 预热(JIT + 频率拉升)
    ts = Float64[]
    for _ in 1:REPS
        e = CUDA.@elapsed begin
            f(nt)
        end
        push!(ts, Float64(e))
    end
    med = median(ts)
    spread = 100 * (maximum(ts) - minimum(ts)) / med
    return med, spread
end

function setup_case(; nx, nz, nt, ns, elastic::Bool)
    dh = 10.0f0; dt = 0.001f0; f0 = 15.0f0
    nbc = 50; fd_order = 8
    vp = fill(3000.0f0, nx, nz); vp[:, nz÷2:end] .= 3800.0f0
    rho = fill(2200.0f0, nx, nz)
    a = get_fd_coefficients(fd_order)
    w = ricker_wavelet(f0, dt, nt)
    wm = repeat(reshape(w, 1, nt), 1, 1)
    if elastic
        med = init_medium(vp, vp ./ 1.8f0, rho, dh, nbc, fd_order)
    else
        med = init_acoustic_medium(vp, rho, dh, nbc, fd_order)
    end
    bc = init_habc(nx, nz, med.pad, dt, dh, minimum(vp))
    rxs = Int32.(collect(10:5:nx-10))
    rec = init_receiver(med.pad, rxs, fill(Int32(6), length(rxs)), elastic ? :vz : :p)
    xs = round.(Int, range(nx ÷ (ns + 1), nx - nx ÷ (ns + 1); length=ns))
    Sb = init_batched_source(med.pad, reshape(Int.(xs), 1, ns), fill(8, 1, ns), wm)
    S1 = init_source(med.pad, Int32[xs[1]], Int32[8], wm)
    return med, bc, rec, a, dt, wm, Sb, S1
end

function run_grid(; nx, nz, nt, elastic::Bool)
    nosnap = Matrix{Float32}[]
    name = elastic ? "elastic" : "acoustic"

    # ── 单炮基线(det / graph),每网格只测一次 ──
    med, bc, rec, a, dt, _, _, S1 = setup_case(nx=nx, nz=nz, nt=nt, ns=1, elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc); nxi = Int32(nxp); nzi = Int32(nzp)
    nrec = length(rec.rx)
    if elastic
        W1 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    else
        W1 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    end
    b1 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)

    function det1!(n)
        if elastic
            _elastic2d_loop_fused!(W1, med, S1, rec, bc, a, dt, n, med.pad,
                b1[1], b1[2], nosnap, 0)
        else
            _acoustic2d_loop_fused!(W1, med, S1, rec, bc, a, dt, n, med.pad,
                nbc, nxi, nzi, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                b1[1], b1[2], b1[3], nosnap, 0)
        end
        return nothing
    end
    function graph1!(n)
        if elastic
            _elastic2d_loop_graph!(W1, med, S1, rec, bc, a, dt, n, med.pad,
                b1[1], b1[2])
        else
            _acoustic2d_loop_graph!(W1, med, S1, rec, bc, a, dt, n, med.pad,
                nbc, nxi, nzi, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                b1[1], b1[2], b1[3])
        end
        return nothing
    end

    t_det, s_det = timed(det1!, nt)
    t_gra, s_gra = timed(graph1!, nt)
    best_single = min(t_det, t_gra)
    @printf("%-8s %-10s 单炮 det   %9.1f st/s (±%4.1f%%)\n", name, "$(nx)x$(nz)", nt / t_det, s_det)
    @printf("%-8s %-10s 单炮 graph %9.1f st/s (±%4.1f%%)\n", name, "$(nx)x$(nz)", nt / t_gra, s_gra)

    # ── 批处理 ns ∈ (2, 4, 8, 16) ──
    for ns in (2, 4, 8, 16)
        med2, bc2, rec2, a2, dt2, _, Sb, _ = setup_case(nx=nx, nz=nz, nt=nt, ns=ns, elastic=elastic)
        nbc2 = Int32(bc2.nbc); nxi2 = Int32(med2.nx); nzi2 = Int32(med2.nz)
        nrec2 = length(rec2.rx)
        if elastic
            Wb = BatchedWavefield(med2.nx - 2med2.pad, med2.nz - 2med2.pad, med2.pad, ns)
        else
            Wb = BatchedAcousticWavefield(med2.nx - 2med2.pad, med2.nz - 2med2.pad, med2.pad, ns)
        end
        sb = Tuple(CUDA.zeros(Float32, nrec2, nt, ns) for _ in 1:3)
        function batch!(n)
            if elastic
                _elastic2d_loop_batch!(Wb, med2, Sb, rec2, bc2, a2, dt2, n, sb[1], sb[2], ns)
            else
                _acoustic2d_loop_batch!(Wb, med2, Sb, rec2, bc2, a2, dt2, n,
                    nbc2, nxi2, nzi2, bc2.qx, bc2.qz, bc2.qt_x, bc2.qt_z, bc2.qxt,
                    sb[1], sb[2], sb[3], ns)
            end
            return nothing
        end
        t_b, s_b = timed(batch!, nt)
        shot_thru = ns * nt / t_b
        @printf("%-8s %-10s 批 ns=%-3d  %9.1f 炮·st/s (±%4.1f%%) | vs 最强单炮 %5.2fx\n",
            name, "$(nx)x$(nz)", ns, shot_thru, s_b, shot_thru / (nt / best_single))
        Wb = nothing
        sb = nothing
        GC.gc()
        CUDA.reclaim()
    end
    println("-"^78)
end

println("GPU: ", CUDA.name(CUDA.device()), "   (每点 $(REPS) 次取中位数)")
println("═"^78)
for elastic in (false, true), (nx, nz, nt) in ((240, 200, 600), (500, 400, 400), (1000, 800, 200))
    run_grid(nx=nx, nz=nz, nt=nt, elastic=elastic)
end
println("提示:若 ±波动 普遍 >15%,请关闭占用 GPU 的其他程序(浏览器硬件加速、")
println("      桌面特效)后重跑;Windows WDDM 下建议连续跑 2-3 遍取稳定一遍。")
