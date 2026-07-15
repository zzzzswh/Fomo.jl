# test/verify_batch.jl — 补丁三(多炮批处理)验收(需 GPU)
#
# 核心判据:各炮之间零数据交互,批处理只是同一套算术在第三维摆 n 层,
# 因此【批处理中每一炮 ↔ 单炮 det 生产路径】必须逐位相等(精确 0.0)。
#
# 三段:
#   1. 循环级:批处理循环 vs 逐炮单跑 det 融合循环,比最终全场 + 记录 → 0.0
#   2. API 级:acoustic2d_batch / elastic2d_batch vs 逐炮调用单炮 API → 0.0
#   3. 吞吐:batch(ns) 相对 ns× 顺序单炮 的炮·步吞吐提升
#
# 用法:julia --project=. test/verify_batch.jl
#
using CUDA
using Printf
using StaticArrays
using Fomo
using Fomo: init_acoustic_medium, AcousticWavefield, init_medium, Wavefield,
    init_habc, init_source, init_receiver, get_fd_coefficients, ricker_wavelet, reset!,
    _acoustic2d_loop_fused!, _elastic2d_loop_fused!,
    BatchedAcousticWavefield, BatchedWavefield, init_batched_source,
    _acoustic2d_loop_batch!, _elastic2d_loop_batch!

maxabs(a, b) = maximum(abs.(Array(a) .- Array(b)))
verdict(x, thr) = x <= thr ? "OK " : "FAIL"

const SHOT_XS = [60, 100, 140, 180]   # 4 炮的震源 x 位置(z 均为 8)

function make_setup(; nx=240, nz=200, nt=400, nbc=50, fd_order=8, elastic=false)
    dh = 10.0f0
    dt = 0.001f0
    f0 = 15.0f0
    vp = fill(3000.0f0, nx, nz); vp[:, nz÷2:end] .= 3800.0f0
    rho = fill(2200.0f0, nx, nz)
    a = get_fd_coefficients(fd_order)
    w = ricker_wavelet(f0, dt, nt)
    wm = repeat(reshape(w, 1, nt), 1, 1)
    if elastic
        vs = vp ./ 1.8f0
        med = init_medium(vp, vs, rho, dh, nbc, fd_order)
    else
        med = init_acoustic_medium(vp, rho, dh, nbc, fd_order)
    end
    bc = init_habc(nx, nz, med.pad, dt, dh, minimum(vp))
    rxs = Int32.(collect(10:5:nx-10))
    rec = init_receiver(med.pad, rxs, fill(Int32(6), length(rxs)), elastic ? :vz : :p)
    return med, bc, rec, a, dt, nt, wm, vp, rho
end

# ══════════════ 1. 循环级:批 vs 逐炮单跑(期望精确 0)══════════════
function sec1(; elastic::Bool)
    med, bc, rec, a, dt, nt, wm, _, _ = make_setup(elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    nrec = length(rec.rx)
    ns = length(SHOT_XS)
    nosnap = Matrix{Float32}[]

    # 批处理:炮沿第 3 维
    sxm = reshape(Int.(SHOT_XS), 1, ns)          # (n_src=1, ns)
    szm = fill(8, 1, ns)
    Sb = init_batched_source(med.pad, sxm, szm, wm)
    if elastic
        Wb = BatchedWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad, ns)
    else
        Wb = BatchedAcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad, ns)
    end
    sb = Tuple(CUDA.zeros(Float32, nrec, nt, ns) for _ in 1:3)
    if elastic
        _elastic2d_loop_batch!(Wb, med, Sb, rec, bc, a, dt, nt, sb[1], sb[2], ns)
    else
        _acoustic2d_loop_batch!(Wb, med, Sb, rec, bc, a, dt, nt,
            nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
            sb[1], sb[2], sb[3], ns)
    end
    CUDA.synchronize()

    # 逐炮单跑(det 生产路径)并逐位比对
    flds = elastic ? (:vx, :vz, :txx, :tzz, :txz) : (:p, :vx, :vz)
    worst_f = 0.0f0
    worst_s = 0.0f0
    for (s, x) in enumerate(SHOT_XS)
        S1 = init_source(med.pad, Int32[x], Int32[8], wm)
        if elastic
            W1 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
        else
            W1 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
        end
        b1 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
        if elastic
            _elastic2d_loop_fused!(W1, med, S1, rec, bc, a, dt, nt, med.pad,
                b1[1], b1[2], nosnap, 0)
        else
            _acoustic2d_loop_fused!(W1, med, S1, rec, bc, a, dt, nt, med.pad,
                nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                b1[1], b1[2], b1[3], nosnap, 0)
        end
        CUDA.synchronize()
        for f in flds
            fb = Array(getfield(Wb, f))
            worst_f = max(worst_f, maximum(abs.(Array(getfield(W1, f)) .- fb[:, :, s])))
        end
        sb1 = Array(sb[1])
        worst_s = max(worst_s, maximum(abs.(Array(b1[1]) .- sb1[:, :, s])))
        sb2 = Array(sb[2])
        worst_s = max(worst_s, maximum(abs.(Array(b1[2]) .- sb2[:, :, s])))
    end

    name = elastic ? "弹性" : "声波"
    println("── 1. $name 循环级:批(ns=$ns) vs 逐炮单跑(期望精确 0)──")
    @printf("  全部场×全部炮  max|Δ| = %.3e   %s\n", worst_f, verdict(worst_f, 0.0))
    @printf("  地震记录       max|Δ| = %.3e   %s\n", worst_s, verdict(worst_s, 0.0))
    return worst_f == 0.0 && worst_s == 0.0
end

# ══════════════ 2. API 级:批 API vs 逐炮单炮 API(期望精确 0)══════════════
function sec2(; elastic::Bool)
    nx, nz, nt = 240, 200, 400
    dh = 10.0f0; dt = 0.001f0; f0 = 15.0f0
    vp = fill(3000.0f0, nx, nz); vp[:, nz÷2:end] .= 3800.0f0
    rho = fill(2200.0f0, nx, nz)
    vs = vp ./ 1.8f0
    rx = collect(10:5:nx-10)
    rz = fill(6, length(rx))
    ns = length(SHOT_XS)

    worst = 0.0f0
    if elastic
        rb = elastic2d_batch(vp, vs, rho, dh, dt, nt, f0;
            sx=SHOT_XS, sz=fill(8, ns), rx=rx, rz=rz, verbose=false)
        for (s, x) in enumerate(SHOT_XS)
            r1 = elastic2d(vp, vs, rho, dh, dt, nt, f0;
                sx=[x], sz=[8], rx=rx, rz=rz, verbose=false)
            worst = max(worst, maximum(abs.(r1.seis_vx .- rb.seis_vx[:, :, s])))
            worst = max(worst, maximum(abs.(r1.seis_vz .- rb.seis_vz[:, :, s])))
        end
    else
        rb = acoustic2d_batch(vp, rho, dh, dt, nt, f0;
            sx=SHOT_XS, sz=fill(8, ns), rx=rx, rz=rz, verbose=false)
        for (s, x) in enumerate(SHOT_XS)
            r1 = acoustic2d(vp, rho, dh, dt, nt, f0;
                sx=[x], sz=[8], rx=rx, rz=rz, verbose=false)
            worst = max(worst, maximum(abs.(r1.seis_p .- rb.seis_p[:, :, s])))
            worst = max(worst, maximum(abs.(r1.seis_vx .- rb.seis_vx[:, :, s])))
            worst = max(worst, maximum(abs.(r1.seis_vz .- rb.seis_vz[:, :, s])))
        end
    end
    name = elastic ? "弹性" : "声波"
    println("── 2. $name API 级:批 API vs 逐炮单炮 API(期望精确 0)──")
    @printf("  地震记录 max|Δ| = %.3e   %s\n", worst, verdict(worst, 0.0))
    return worst == 0.0
end

# ══════════════ 3. 吞吐缩放 ══════════════
function bench_batch(; nx, nz, nt, ns, elastic::Bool)
    med, bc, rec, a, dt, _, wm, _, _ = make_setup(nx=nx, nz=nz, nt=nt, elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc); nxi = Int32(nxp); nzi = Int32(nzp)
    nrec = length(rec.rx)
    nosnap = Matrix{Float32}[]

    xs = round.(Int, range(nx ÷ (ns + 1), nx - nx ÷ (ns + 1); length=ns))
    sxm = reshape(Int.(xs), 1, ns)
    szm = fill(8, 1, ns)
    Sb = init_batched_source(med.pad, sxm, szm, wm)

    # 单炮基线(det 生产路径,取第 1 炮位置)
    S1 = init_source(med.pad, Int32[xs[1]], Int32[8], wm)
    if elastic
        W1 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
        Wb = BatchedWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad, ns)
    else
        W1 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
        Wb = BatchedAcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad, ns)
    end
    b1 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    sb = Tuple(CUDA.zeros(Float32, nrec, nt, ns) for _ in 1:3)

    function single_run!(n)
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
    function batch_run!(n)
        if elastic
            _elastic2d_loop_batch!(Wb, med, Sb, rec, bc, a, dt, n, sb[1], sb[2], ns)
        else
            _acoustic2d_loop_batch!(Wb, med, Sb, rec, bc, a, dt, n,
                nbc, nxi, nzi, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                sb[1], sb[2], sb[3], ns)
        end
        return nothing
    end

    single_run!(5)
    reset!(W1)
    e1 = CUDA.@elapsed begin
        single_run!(nt)
    end
    batch_run!(5)
    reset!(Wb)
    eb = CUDA.@elapsed begin
        batch_run!(nt)
    end
    return Float64(e1), Float64(eb)
end

function sec3()
    println("── 3. 吞吐缩放:batch(ns) vs ns× 顺序单炮(炮·步/秒)──")
    @printf("%-8s %-10s %4s | %12s %12s | %10s\n",
        "eq", "grid", "ns", "顺序 炮·st/s", "批 炮·st/s", "批/顺序")
    println("-"^72)
    for elastic in (false, true), (nx, nz, nt) in ((240, 200, 600), (500, 400, 400))
        for ns in (1, 4, 8)
            t1, tb = bench_batch(nx=nx, nz=nz, nt=nt, ns=ns, elastic=elastic)
            seq = nt / t1                    # 顺序:每炮 t1,炮·步/s = nt/t1
            bat = ns * nt / tb
            @printf("%-8s %-10s %4d | %12.1f %12.1f | %9.2fx\n",
                elastic ? "elastic" : "acoustic", "$(nx)x$(nz)", ns, seq, bat, bat / seq)
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
println("═"^70)
ok1a = sec1(elastic=false)
ok1e = sec1(elastic=true); println()
ok2a = sec2(elastic=false)
ok2e = sec2(elastic=true); println()
sec3()
println("═"^70)
if ok1a && ok1e && ok2a && ok2e
    println("全部通过:批处理每一炮与单炮生产路径逐位相等。可合入。")
else
    println("存在非零差异 —— 批处理实现有 bug,请把完整输出发回。")
end
