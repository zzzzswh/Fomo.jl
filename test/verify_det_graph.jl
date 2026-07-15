# test/verify_det_graph.jl — 补丁二(确定性 HABC + CUDA Graphs)验收(需 GPU)
#
# 五段验收:
#   1. det-HABC 单步 ↔ 无 race CPU 双遍参考:偏差应仅为 FMA 舍入(~1e-7 相对),
#      且 det 自身重跑逐位为 0(racy 版做不到的正是这一点)。
#   2. 生产路径(det 融合循环)完整 400 步跑两遍:全场+记录逐位 0.0
#      —— 这是"逐位可复现"的直接验收。
#   3. det ↔ racy 物理一致性:分歧仍应限于边界帧,深内区 ~1e-6·peak
#      (确定性化不改变物理,只固定 race 的解)。
#   4. graph 循环 ↔ det 融合循环:同 kernel 同序同参,期望逐位 0.0。
#      (若 capture 失败会打印警告并回退,此时比较仍应为 0。)
#   5. 三方吞吐:racy 融合(补丁一) / det 融合 / CUDA Graph。
#
# 用法:julia --project=. test/verify_det_graph.jl
#
using CUDA
using Printf
using StaticArrays
using Fomo
using Fomo: init_acoustic_medium, AcousticWavefield, init_medium, Wavefield,
    init_habc, init_source, init_receiver, get_fd_coefficients, ricker_wavelet, reset!,
    backup_single_field!, update_velocity_acoustic!, update_pressure_acoustic!,
    apply_habc_single_field!, inject_source!, record_receivers!,
    fused_update_velocity_acoustic!, fused_update_pressure_acoustic!,
    fused_update_velocity_elastic!, fused_update_stress_elastic!,
    apply_habc_frame_1!, apply_habc_frame_2!, apply_habc_frame_3!,
    record_receivers3!, record_receivers2!, inject_source_pair!,
    apply_habc_det_1!, apply_habc_det_2!, apply_habc_det_3!, _habc_frame_total,
    _acoustic2d_loop_fused!, _elastic2d_loop_fused!,
    _acoustic2d_loop_graph!, _elastic2d_loop_graph!

maxabs(a, b) = maximum(abs.(Array(a) .- Array(b)))
verdict(x, thr) = x <= thr ? "OK " : "FAIL"

function make_setup(; nx=240, nz=200, nt=400, nbc=50, fd_order=8, elastic=false)
    dh = 10.0f0
    dt = 0.001f0
    f0 = 15.0f0
    vp = fill(3000.0f0, nx, nz)
    vp[:, nz÷2:end] .= 3800.0f0
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
    src = init_source(med.pad, Int32[nx÷2], Int32[8], wm)
    rxs = Int32.(collect(10:5:nx-10))
    rec = init_receiver(med.pad, rxs, fill(Int32(6), length(rxs)), elastic ? :vz : :p)
    return med, bc, src, rec, a, dt, nt
end

"CPU 双遍 Higdon 参考(与 v2 相同)。"
function habc_cpu_ref(fpre::Matrix{Float32}, fold::Matrix{Float32}, w::Matrix{Float32},
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32, nbc::Int)
    nx, nz = size(fpre)
    fref = copy(fpre)
    for j in 2:nz-1, i in 2:nx-1
        is_left = i <= nbc + 1
        is_right = i >= nx - nbc
        is_top = j <= nbc + 1
        is_bottom = j >= nz - nbc
        in_x = is_left || is_right
        in_z = is_top || is_bottom
        (in_x || in_z) || continue
        wt = w[i, j]
        f_cur = fpre[i, j]
        if in_x && in_z
            sum_x = is_left ?
                    (-qx * fpre[i+1, j] - qt_x * fold[i, j] - qxt * fold[i+1, j]) :
                    (-qx * fpre[i-1, j] - qt_x * fold[i, j] - qxt * fold[i-1, j])
            sum_z = is_top ?
                    (-qz * fpre[i, j+1] - qt_z * fold[i, j] - qxt * fold[i, j+1]) :
                    (-qz * fpre[i, j-1] - qt_z * fold[i, j] - qxt * fold[i, j-1])
            fref[i, j] = wt * f_cur + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        elseif in_x
            sum_x = is_left ?
                    (-qx * fpre[i+1, j] - qt_x * fold[i, j] - qxt * fold[i+1, j]) :
                    (-qx * fpre[i-1, j] - qt_x * fold[i, j] - qxt * fold[i-1, j])
            fref[i, j] = wt * f_cur + (1.0f0 - wt) * sum_x
        else
            sum_z = is_top ?
                    (-qz * fpre[i, j+1] - qt_z * fold[i, j] - qxt * fold[i, j+1]) :
                    (-qz * fpre[i, j-1] - qt_z * fold[i, j] - qxt * fold[i, j-1])
            fref[i, j] = wt * f_cur + (1.0f0 - wt) * sum_z
        end
    end
    return fref
end

# ══════════════════ 1. det-HABC 单步 vs CPU 参考 + 自复现 ══════════════════
function sec1()
    med, bc, src, rec, a, dt, _ = make_setup()
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc)
    nx = Int32(nxp)
    nz = Int32(nzp)
    W = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)

    for it in 1:150
        backup_single_field!(W.vx_old, W.vx, nbc, nx, nz)
        backup_single_field!(W.vz_old, W.vz, nbc, nx, nz)
        update_velocity_acoustic!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W.vx, W.vx_old, bc.w_vx, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        apply_habc_single_field!(W.vz, W.vz_old, bc.w_vz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        backup_single_field!(W.p_old, W.p, nbc, nx, nz)
        update_pressure_acoustic!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W.p, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        inject_source!(W.p, src, it, dt)
    end
    backup_single_field!(W.vx_old, W.vx, nbc, nx, nz)
    backup_single_field!(W.vz_old, W.vz, nbc, nx, nz)
    update_velocity_acoustic!(W, med, a, dt, inner_nx, inner_nz)
    apply_habc_single_field!(W.vx, W.vx_old, bc.w_vx, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    apply_habc_single_field!(W.vz, W.vz_old, bc.w_vz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    backup_single_field!(W.p_old, W.p, nbc, nx, nz)
    update_pressure_acoustic!(W, med, a, dt, inner_nx, inner_nz)
    CUDA.synchronize()

    p_pre = Array(W.p)
    ref = habc_cpu_ref(p_pre, Array(W.p_old), Array(bc.w_tau),
        bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, Int(bc.nbc))
    scale = maximum(abs.(ref))

    scr = CUDA.zeros(Float32, _habc_frame_total(nx, nz, nbc))
    g1 = CuArray(p_pre)
    apply_habc_det_1!(g1, W.p_old, bc.w_tau, scr, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    g2 = CuArray(p_pre)
    apply_habc_det_1!(g2, W.p_old, bc.w_tau, scr, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    g3 = CuArray(p_pre)   # racy 帧内核对照
    apply_habc_frame_1!(g3, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    CUDA.synchronize()

    d_ref = maximum(abs.(Array(g1) .- ref)) / scale
    d_rep = maxabs(g1, g2)
    d_racy = maxabs(g1, g3) / scale
    println("── 1. det-HABC 单步(相对全场峰值)──")
    @printf("  det ↔ CPU双遍参考   %.3e   %s   (期望仅 FMA 舍入,~1e-7)\n", d_ref, verdict(d_ref, 1e-6))
    @printf("  det ↔ det 重跑      %.3e   %s   (期望精确 0)\n", d_rep, verdict(d_rep, 0.0))
    @printf("  det ↔ racy帧内核    %.3e        (信息值;本状态若无 race 事件则为 0)\n", d_racy)
    return d_ref <= 1e-6 && d_rep == 0.0
end

# ══════════════════ 2. 生产路径跑两遍:逐位可复现 ══════════════════
function sec2(; elastic::Bool)
    med, bc, src, rec, a, dt, nt = make_setup(elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc)
    nx = Int32(nxp)
    nz = Int32(nzp)
    nrec = length(rec.rx)
    WF = elastic ? Wavefield : AcousticWavefield
    nosnap = Matrix{Float32}[]

    run!(W, bufs) = elastic ?
                    _elastic2d_loop_fused!(W, med, src, rec, bc, a, dt, nt, med.pad,
        bufs[1], bufs[2], nosnap, 0) :
                    _acoustic2d_loop_fused!(W, med, src, rec, bc, a, dt, nt, med.pad,
        nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
        bufs[1], bufs[2], bufs[3], nosnap, 0)

    W1 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    b1 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    b2 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    run!(W1, b1)
    run!(W2, b2)
    CUDA.synchronize()

    flds = elastic ? (:vx, :vz, :txx, :tzz, :txz) : (:p, :vx, :vz)
    d_f = maximum([maxabs(getfield(W1, f), getfield(W2, f)) for f in flds])
    d_s = max(maxabs(b1[1], b2[1]), maxabs(b1[2], b2[2]))
    name = elastic ? "弹性" : "声波"
    println("── 2. $name 生产路径(det 融合循环)跑两遍(期望精确 0)──")
    @printf("  全部场    max|Δ| = %.3e   %s\n", d_f, verdict(d_f, 0.0))
    @printf("  地震记录  max|Δ| = %.3e   %s\n", d_s, verdict(d_s, 0.0))
    return d_f == 0.0 && d_s == 0.0
end

# ══════════════════ 3. det ↔ racy 物理一致性 ══════════════════
function sec3(; elastic::Bool)
    med, bc, src, rec, a, dt, nt = make_setup(elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc)
    nx = Int32(nxp)
    nz = Int32(nzp)
    nrec = length(rec.rx)
    m = Int(bc.nbc) + 2 + 8
    WF = elastic ? Wavefield : AcousticWavefield
    nosnap = Matrix{Float32}[]

    # racy 序列(补丁一行为),手工循环以便追踪峰值
    W1 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    s1 = CUDA.zeros(Float32, nrec, nt)
    d1 = CUDA.zeros(Float32, nrec, nt)
    d2 = CUDA.zeros(Float32, nrec, nt)
    peak = 0.0f0
    mainf(W) = elastic ? W.vz : W.p
    for it in 1:nt
        if elastic
            fused_update_velocity_elastic!(W1, med, a, dt, nbc)
            apply_habc_frame_2!(W1.vx, W1.vx_old, bc.w_vx, W1.vz, W1.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            fused_update_stress_elastic!(W1, med, a, dt, nbc)
            apply_habc_frame_3!(W1.txx, W1.txx_old, W1.tzz, W1.tzz_old, W1.txz, W1.txz_old,
                bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            inject_source_pair!(W1.txx, W1.tzz, src, it)
            record_receivers2!(d1, s1, W1, rec, it)
        else
            fused_update_velocity_acoustic!(W1, med, a, dt, nbc)
            apply_habc_frame_2!(W1.vx, W1.vx_old, bc.w_vx, W1.vz, W1.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            fused_update_pressure_acoustic!(W1, med, a, dt, nbc)
            apply_habc_frame_1!(W1.p, W1.p_old, bc.w_tau,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            inject_source!(W1.p, src, it, dt)
            record_receivers3!(s1, d1, d2, W1, rec, it)
        end
        if it % 20 == 0
            peak = max(peak, Float32(maximum(abs, mainf(W1))))
        end
    end
    CUDA.synchronize()
    peak = max(peak, Float32(maximum(abs, mainf(W1))), 1.0f-30)

    # det 生产路径
    W2 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    b2 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    if elastic
        _elastic2d_loop_fused!(W2, med, src, rec, bc, a, dt, nt, med.pad,
            b2[1], b2[2], nosnap, 0)
        s2 = b2[2]
    else
        _acoustic2d_loop_fused!(W2, med, src, rec, bc, a, dt, nt, med.pad,
            nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
            b2[1], b2[2], b2[3], nosnap, 0)
        s2 = b2[1]
    end
    CUDA.synchronize()

    d = abs.(Array(mainf(W1)) .- Array(mainf(W2)))
    d_int = maximum(@view d[m+1:end-m, m+1:end-m]) / peak
    d_all = maximum(d) / peak
    seisd = maxabs(s1, s2) / max(maximum(abs.(Array(s1))), 1.0f-30)
    name = elastic ? "弹性(vz)" : "声波(p)"
    println("── 3. $name det ↔ racy(补丁一)物理一致性,以全程峰值归一 ──")
    @printf("  深内区   max|Δ|/peak = %.3e   %s\n", d_int, verdict(d_int, 5e-4))
    @printf("  含边界帧 max|Δ|/peak = %.3e   (信息值)\n", d_all)
    @printf("  地震记录 rel-max      = %.3e   %s\n", seisd, verdict(seisd, 1e-4))
    return d_int <= 5e-4 && seisd <= 1e-4
end

# ══════════════════ 4. graph 循环 ↔ det 融合循环:逐位等价 ══════════════════
function sec4(; elastic::Bool)
    med, bc, src, rec, a, dt, nt = make_setup(elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc)
    nx = Int32(nxp)
    nz = Int32(nzp)
    nrec = length(rec.rx)
    WF = elastic ? Wavefield : AcousticWavefield
    nosnap = Matrix{Float32}[]

    W1 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    b1 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    b2 = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    if elastic
        _elastic2d_loop_fused!(W1, med, src, rec, bc, a, dt, nt, med.pad, b1[1], b1[2], nosnap, 0)
        _elastic2d_loop_graph!(W2, med, src, rec, bc, a, dt, nt, med.pad, b2[1], b2[2])
    else
        _acoustic2d_loop_fused!(W1, med, src, rec, bc, a, dt, nt, med.pad,
            nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
            b1[1], b1[2], b1[3], nosnap, 0)
        _acoustic2d_loop_graph!(W2, med, src, rec, bc, a, dt, nt, med.pad,
            nbc, nx, nz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
            b2[1], b2[2], b2[3])
    end
    CUDA.synchronize()

    flds = elastic ? (:vx, :vz, :txx, :tzz, :txz) : (:p, :vx, :vz)
    d_f = maximum([maxabs(getfield(W1, f), getfield(W2, f)) for f in flds])
    d_s = max(maxabs(b1[1], b2[1]), maxabs(b1[2], b2[2]))
    name = elastic ? "弹性" : "声波"
    println("── 4. $name graph 循环 ↔ det 融合循环(期望精确 0)──")
    @printf("  全部场    max|Δ| = %.3e   %s\n", d_f, verdict(d_f, 0.0))
    @printf("  地震记录  max|Δ| = %.3e   %s\n", d_s, verdict(d_s, 0.0))
    return d_f == 0.0 && d_s == 0.0
end

# ══════════════════ 5. 三方吞吐 ══════════════════
function bench_one(; nx, nz, nt, elastic::Bool)
    med, bc, src, rec, a, dt, _ = make_setup(nx=nx, nz=nz, nt=nt, elastic=elastic)
    nxp, nzp = med.nx, med.nz
    nbc = Int32(bc.nbc)
    nxi = Int32(nxp)
    nzi = Int32(nzp)
    nrec = length(rec.rx)
    WF = elastic ? Wavefield : AcousticWavefield
    W = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    bufs = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    nosnap = Matrix{Float32}[]

    function racy_step!(it)
        if elastic
            fused_update_velocity_elastic!(W, med, a, dt, nbc)
            apply_habc_frame_2!(W.vx, W.vx_old, bc.w_vx, W.vz, W.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
            fused_update_stress_elastic!(W, med, a, dt, nbc)
            apply_habc_frame_3!(W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
                bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
            inject_source_pair!(W.txx, W.tzz, src, it)
            record_receivers2!(bufs[1], bufs[2], W, rec, it)
        else
            fused_update_velocity_acoustic!(W, med, a, dt, nbc)
            apply_habc_frame_2!(W.vx, W.vx_old, bc.w_vx, W.vz, W.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
            fused_update_pressure_acoustic!(W, med, a, dt, nbc)
            apply_habc_frame_1!(W.p, W.p_old, bc.w_tau,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
            inject_source!(W.p, src, it, dt)
            record_receivers3!(bufs[1], bufs[2], bufs[3], W, rec, it)
        end
        return nothing
    end

    function det_run!(n)
        if elastic
            _elastic2d_loop_fused!(W, med, src, rec, bc, a, dt, n, med.pad,
                bufs[1], bufs[2], nosnap, 0)
        else
            _acoustic2d_loop_fused!(W, med, src, rec, bc, a, dt, n, med.pad,
                nbc, nxi, nzi, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                bufs[1], bufs[2], bufs[3], nosnap, 0)
        end
        return nothing
    end

    function graph_run!(n)
        if elastic
            _elastic2d_loop_graph!(W, med, src, rec, bc, a, dt, n, med.pad,
                bufs[1], bufs[2])
        else
            _acoustic2d_loop_graph!(W, med, src, rec, bc, a, dt, n, med.pad,
                nbc, nxi, nzi, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt,
                bufs[1], bufs[2], bufs[3])
        end
        return nothing
    end

    # racy
    for it in 1:5
        racy_step!(it)
    end
    CUDA.synchronize()
    reset!(W)
    e_racy = CUDA.@elapsed begin
        for it in 1:nt
            racy_step!(it)
        end
    end
    reset!(W)
    # det(整段调用,含一次性 scratch 分配)
    det_run!(5)
    reset!(W)
    e_det = CUDA.@elapsed begin
        det_run!(nt)
    end
    reset!(W)
    # graph(整段调用,含预热步 + capture ~1ms,按生产路径的真实成本计)
    graph_run!(5)
    reset!(W)
    e_graph = CUDA.@elapsed begin
        graph_run!(nt)
    end
    return Float64(e_racy), Float64(e_det), Float64(e_graph)
end

function sec5()
    println("── 5. 吞吐:racy 融合(补丁一) / det 融合 / CUDA Graph ──")
    @printf("%-8s %-10s %5s | %10s %10s %10s | %8s %8s\n",
        "eq", "grid", "nt", "racy st/s", "det st/s", "graph st/s", "det/racy", "graph/racy")
    println("-"^92)
    for elastic in (false, true), (nx, nz, nt) in ((240, 200, 1000), (500, 400, 600), (1000, 800, 300))
        t_r, t_d, t_g = bench_one(nx=nx, nz=nz, nt=nt, elastic=elastic)
        @printf("%-8s %-10s %5d | %10.1f %10.1f %10.1f | %7.2fx %7.2fx\n",
            elastic ? "elastic" : "acoustic", "$(nx)x$(nz)", nt,
            nt / t_r, nt / t_d, nt / t_g, t_r / t_d, t_r / t_g)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
println("═"^70)
ok1 = sec1();
println();
ok2a = sec2(elastic=false)
ok2e = sec2(elastic=true);
println();
ok3a = sec3(elastic=false)
ok3e = sec3(elastic=true);
println();
ok4a = sec4(elastic=false)
ok4e = sec4(elastic=true);
println();
sec5()
println("═"^70)
all_ok = ok1 && ok2a && ok2e && ok3a && ok3e && ok4a && ok4e
if all_ok
    println("全部通过:det-HABC 数学正确且逐位可复现;graph 与生产路径逐位等价;")
    println("          物理与 racy 版一致(分歧限于边界帧)。可合入。")
else
    println("存在未通过项,请把完整输出发回。")
end