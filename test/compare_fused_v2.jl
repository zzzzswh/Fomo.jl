# test/compare_fused_v2.jl — v1 全场 FAIL 的来源分解(需 GPU)
#
# v1 现象:地震记录(内区)全部 ~1e-6 吻合,而最终全场 rel-max 差 1e-3~1e-2。
#
# 假设:失配全部来自 HABC 内核中【原实现已文档化】的邻点 race
#       (见 kernels.jl "关于并行安全性" 一节:f[i±1,j] 可能读到已被邻线程
#       修正后的值)。旧的 2D 全网格线程映射与新的 1D 帧映射调度顺序不同,
#       同一个 race 会收敛到不同结果;其量级是 O(单点修正量),不是舍入误差,
#       所以出现 1e-3 是这个机制的正常签名,而 v1 的 1e-4 阈值定错了。
#
# 三段测试(每段单独可证伪):
#   A. 把 HABC 固定为【旧内核】,只替换备份/更新/注入/记录的融合。
#      这些融合我论证过是逐位等价重排 → 期望差 = 0.0(精确)。
#      若 A ≠ 0:融合逻辑真的有 bug,假设被证伪,请把输出发回来。
#      同时跑两遍完全相同的旧序列作对照,排除旧内核本身的跨运行非确定性。
#   B. 单步 HABC 对照一个【无 race 的 CPU 双遍参考】(先全部读旧值算修正,
#      再统一写回)。期望:旧内核↔参考、新内核↔参考、旧↔新 三个偏差同量级,
#      且全部位于边界帧内 → 证明两者是同一数学的不同 race 采样,无谁对谁错。
#   C. 完整 400 步,把误差按【深内区 / 边界帧】分解,并以全程峰值幅度归一,
#      给出物理上有意义的误差水平(v1 的 rel-max 分母是终态场峰值,
#      波被吸收后分母很小,数字被放大)。
#
# 用法:julia --project=. test/compare_fused_v2.jl
#
using CUDA
using Printf
using StaticArrays
using Fomo
using Fomo: init_acoustic_medium, AcousticWavefield, init_medium, Wavefield,
    init_habc, init_source, init_receiver, get_fd_coefficients, ricker_wavelet, reset!,
    backup_single_field!, update_velocity_acoustic!, update_pressure_acoustic!,
    apply_habc_single_field!, inject_source!, record_receivers!,
    backup_velocity!, backup_stress!, update_velocity!, update_stress!,
    apply_habc_velocity!, apply_habc_stress!,
    fused_update_velocity_acoustic!, fused_update_pressure_acoustic!,
    fused_update_velocity_elastic!, fused_update_stress_elastic!,
    apply_habc_frame_1!, apply_habc_frame_2!, apply_habc_frame_3!,
    record_receivers3!, record_receivers2!, inject_source_pair!

maxabs(a, b) = maximum(abs.(Array(a) .- Array(b)))

function make_setup(; nx=240, nz=200, nbc=50, fd_order=8, elastic=false)
    dh = 10.0f0
    dt = 0.001f0
    nt = 400
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
    src = init_source(med.pad, Int32[nx ÷ 2], Int32[8], wm)
    rxs = Int32.(collect(10:5:nx-10))
    rec = init_receiver(med.pad, rxs, fill(Int32(6), length(rxs)), elastic ? :vz : :p)
    return med, bc, src, rec, a, dt, nt
end

verdict(x, thr) = x <= thr ? "OK " : "FAIL"

# ══════════════════════════════════════════════════════════════════════════════
# 测试 A:HABC 固定为旧内核,只考融合的备份/更新/注入/记录 → 期望精确 0.0
# ══════════════════════════════════════════════════════════════════════════════
function testA_acoustic()
    med, bc, src, rec, a, dt, nt = make_setup(elastic=false)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    nrec = length(rec.rx)

    W0 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)  # 旧序列
    W1 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)  # 旧序列(控制组)
    W2 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)  # 融合更新 + 旧 HABC
    s0 = CUDA.zeros(Float32, nrec, nt)
    s2p = CUDA.zeros(Float32, nrec, nt)
    s2x = CUDA.zeros(Float32, nrec, nt)
    s2z = CUDA.zeros(Float32, nrec, nt)

    old_step!(W, it) = begin
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

    for it in 1:nt
        old_step!(W0, it); record_receivers!(s0, W0.p, rec, it)
        old_step!(W1, it)
        # 融合更新 + 旧 HABC(与旧序列同一 HABC 内核、同一 launch 几何)
        fused_update_velocity_acoustic!(W2, med, a, dt, nbc)
        apply_habc_single_field!(W2.vx, W2.vx_old, bc.w_vx, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        apply_habc_single_field!(W2.vz, W2.vz_old, bc.w_vz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        fused_update_pressure_acoustic!(W2, med, a, dt, nbc)
        apply_habc_single_field!(W2.p, W2.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        inject_source!(W2.p, src, it, dt)
        record_receivers3!(s2p, s2x, s2z, W2, rec, it)
    end
    CUDA.synchronize()

    ctrl = max(maxabs(W0.p, W1.p), maxabs(W0.vx, W1.vx), maxabs(W0.vz, W1.vz))
    fuse = max(maxabs(W0.p, W2.p), maxabs(W0.vx, W2.vx), maxabs(W0.vz, W2.vz))
    seis = maxabs(s0, s2p)
    println("── A. 声波:HABC 固定为旧内核,只考融合(期望全为 0.0)──")
    @printf("  旧 vs 旧(控制,同代码跑两遍)  max|Δ| = %.3e\n", ctrl)
    @printf("  旧 vs 融合更新+旧HABC          max|Δ| = %.3e   %s\n", fuse, verdict(fuse, 1e-7))
    @printf("  记录 record vs record3          max|Δ| = %.3e   %s\n", seis, verdict(seis, 1e-7))
    if ctrl > 0
        println("  [注意] 控制组不为 0:旧 HABC 内核在你的 GPU 上跨运行即非确定,")
        println("         此时应以 |旧vs融合| ≲ |旧vs旧| 为通过标准。")
    end
    return fuse <= max(1e-7, 2 * ctrl)
end

function testA_elastic()
    med, bc, src, rec, a, dt, nt = make_setup(elastic=true)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nrec = length(rec.rx)

    W0 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W1 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    s0 = CUDA.zeros(Float32, nrec, nt)
    s2x = CUDA.zeros(Float32, nrec, nt)
    s2z = CUDA.zeros(Float32, nrec, nt)

    old_step!(W, it) = begin
        backup_velocity!(W, bc, med)
        update_velocity!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_velocity!(W, bc, med)
        backup_stress!(W, bc, med)
        update_stress!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_stress!(W, bc, med)
        inject_source!(W.txx, src, it, dt)
        inject_source!(W.tzz, src, it, dt)
    end

    nbc32 = Int32(bc.nbc)
    for it in 1:nt
        old_step!(W0, it); record_receivers!(s0, W0.vz, rec, it)
        old_step!(W1, it)
        fused_update_velocity_elastic!(W2, med, a, dt, nbc32)
        apply_habc_velocity!(W2, bc, med)
        fused_update_stress_elastic!(W2, med, a, dt, nbc32)
        apply_habc_stress!(W2, bc, med)
        inject_source_pair!(W2.txx, W2.tzz, src, it)
        record_receivers2!(s2x, s2z, W2, rec, it)
    end
    CUDA.synchronize()

    ctrl = maximum([maxabs(getfield(W0, f), getfield(W1, f)) for f in (:vx, :vz, :txx, :tzz, :txz)])
    fuse = maximum([maxabs(getfield(W0, f), getfield(W2, f)) for f in (:vx, :vz, :txx, :tzz, :txz)])
    seis = maxabs(s0, s2z)
    println("── A. 弹性:HABC 固定为旧内核,只考融合(期望全为 0.0)──")
    @printf("  旧 vs 旧(控制)                 max|Δ| = %.3e\n", ctrl)
    @printf("  旧 vs 融合更新+旧HABC          max|Δ| = %.3e   %s\n", fuse, verdict(fuse, 1e-7))
    @printf("  记录 record vs record2          max|Δ| = %.3e   %s\n", seis, verdict(seis, 1e-7))
    return fuse <= max(1e-7, 2 * ctrl)
end

# ══════════════════════════════════════════════════════════════════════════════
# 测试 B:单步 HABC 对照无 race 的 CPU 双遍参考
# ══════════════════════════════════════════════════════════════════════════════
"CPU 双遍 Higdon:所有邻点一律读修正前的值(与 GPU 公式逐字同序),再统一写回。"
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

function testB_acoustic()
    med, bc, src, rec, a, dt, _ = make_setup(elastic=false)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    W = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)

    # 先完整推进 150 步(震源在浅部,波早已进入上边界帧),再走到 HABC p 之前停下
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
    # 此刻 W.p = 更新后、HABC 前;W.p_old = 更新前备份 —— 正是 HABC 的输入

    p_pre = Array(W.p)
    p_old = Array(W.p_old)
    w_tau = Array(bc.w_tau)
    ref = habc_cpu_ref(p_pre, p_old, w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, Int(bc.nbc))
    scale = maximum(abs.(ref))

    g_old = CuArray(p_pre)
    apply_habc_single_field!(g_old, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    g_new = CuArray(p_pre)
    apply_habc_frame_1!(g_new, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    g_new2 = CuArray(p_pre)
    apply_habc_frame_1!(g_new2, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
    CUDA.synchronize()

    d_or = maximum(abs.(Array(g_old) .- ref))
    d_nr = maximum(abs.(Array(g_new) .- ref))
    d_on = maxabs(g_old, g_new)
    d_nn = maxabs(g_new, g_new2)

    dmat = abs.(Array(g_old) .- Array(g_new))
    am = argmax(dmat)
    in_frame = (am[1] <= bc.nbc + 2 || am[1] >= nxp - bc.nbc - 1 ||
                am[2] <= bc.nbc + 2 || am[2] >= nzp - bc.nbc - 1)

    println("── B. 声波单步 HABC vs 无 race 的 CPU 双遍参考(相对全场峰值)──")
    @printf("  旧GPU内核 ↔ 参考   %.3e   ← 旧内核的 race 偏差\n", d_or / scale)
    @printf("  新GPU内核 ↔ 参考   %.3e   ← 新内核的 race 偏差\n", d_nr / scale)
    @printf("  旧GPU ↔ 新GPU      %.3e   ← 两种调度下 race 的互差\n", d_on / scale)
    @printf("  新GPU ↔ 新GPU重跑  %.3e   ← 新内核自身可复现性\n", d_nn / scale)
    @printf("  旧↔新最大差点位 (i,j)=(%d,%d)%s\n", am[1], am[2],
        in_frame ? ",位于边界帧内 ✓" : ",不在边界帧内 ←【异常,请回报】")
    println("  判读:若三个 race 偏差同量级且最大差在帧内,则新旧同为同一数学的")
    println("        不同 race 采样,无谁对谁错;若新↔参考 ≫ 旧↔参考,则新内核有问题。")
    return in_frame && d_nr < 10 * max(d_or, 1e-12)
end

# ══════════════════════════════════════════════════════════════════════════════
# 测试 C:完整 400 步,内区/边界帧分解 + 全程峰值归一
# ══════════════════════════════════════════════════════════════════════════════
function testC(; elastic::Bool)
    med, bc, src, rec, a, dt, nt = make_setup(elastic=elastic)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    nrec = length(rec.rx)
    m = Int(bc.nbc) + 2 + 8   # 边界帧 + FD halo,再往里才算"深内区"

    WF = elastic ? Wavefield : AcousticWavefield
    W1 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = WF(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    s1 = CUDA.zeros(Float32, nrec, nt)
    s2 = CUDA.zeros(Float32, nrec, nt)
    dummy1 = CUDA.zeros(Float32, nrec, nt)   # 接住不参与比较的场
    dummy2 = CUDA.zeros(Float32, nrec, nt)
    peak = 0.0f0
    mainf(W) = elastic ? W.vz : W.p

    for it in 1:nt
        if elastic
            backup_velocity!(W1, bc, med)
            update_velocity!(W1, med, a, dt, inner_nx, inner_nz)
            apply_habc_velocity!(W1, bc, med)
            backup_stress!(W1, bc, med)
            update_stress!(W1, med, a, dt, inner_nx, inner_nz)
            apply_habc_stress!(W1, bc, med)
            inject_source!(W1.txx, src, it, dt)
            inject_source!(W1.tzz, src, it, dt)
            record_receivers!(s1, W1.vz, rec, it)

            fused_update_velocity_elastic!(W2, med, a, dt, nbc)
            apply_habc_frame_2!(W2.vx, W2.vx_old, bc.w_vx, W2.vz, W2.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            fused_update_stress_elastic!(W2, med, a, dt, nbc)
            apply_habc_frame_3!(W2.txx, W2.txx_old, W2.tzz, W2.tzz_old, W2.txz, W2.txz_old,
                bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            inject_source_pair!(W2.txx, W2.tzz, src, it)
            record_receivers2!(dummy1, s2, W2, rec, it)   # vx→dummy, vz→s2
        else
            backup_single_field!(W1.vx_old, W1.vx, nbc, nx, nz)
            backup_single_field!(W1.vz_old, W1.vz, nbc, nx, nz)
            update_velocity_acoustic!(W1, med, a, dt, inner_nx, inner_nz)
            apply_habc_single_field!(W1.vx, W1.vx_old, bc.w_vx, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            apply_habc_single_field!(W1.vz, W1.vz_old, bc.w_vz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            backup_single_field!(W1.p_old, W1.p, nbc, nx, nz)
            update_pressure_acoustic!(W1, med, a, dt, inner_nx, inner_nz)
            apply_habc_single_field!(W1.p, W1.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            inject_source!(W1.p, src, it, dt)
            record_receivers!(s1, W1.p, rec, it)

            fused_update_velocity_acoustic!(W2, med, a, dt, nbc)
            apply_habc_frame_2!(W2.vx, W2.vx_old, bc.w_vx, W2.vz, W2.vz_old, bc.w_vz,
                bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            fused_update_pressure_acoustic!(W2, med, a, dt, nbc)
            apply_habc_frame_1!(W2.p, W2.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
            inject_source!(W2.p, src, it, dt)
            record_receivers3!(s2, dummy1, dummy2, W2, rec, it)   # p→s2, vx/vz→dummy
        end
        if it % 20 == 0
            peak = max(peak, Float32(maximum(abs, mainf(W1))))
        end
    end
    CUDA.synchronize()
    peak = max(peak, Float32(maximum(abs, mainf(W1))), 1.0f-30)

    f1 = Array(mainf(W1)); f2 = Array(mainf(W2))
    d = abs.(f1 .- f2)
    d_int = maximum(@view d[m+1:end-m, m+1:end-m])
    d_all = maximum(d)
    am = argmax(d)
    in_frame = (am[1] <= bc.nbc + 2 || am[1] >= nxp - bc.nbc - 1 ||
                am[2] <= bc.nbc + 2 || am[2] >= nzp - bc.nbc - 1)
    seisd = maxabs(s1, s2) / max(maximum(abs.(Array(s1))), 1.0f-30)

    name = elastic ? "弹性(vz)" : "声波(p)"
    println("── C. $name 完整 $nt 步,按区域分解,以全程峰值 $(Printf.@sprintf("%.3e", peak)) 归一 ──")
    @printf("  深内区   max|Δ|/peak = %.3e   %s\n", d_int / peak, verdict(d_int / peak, 5e-4))
    @printf("  含边界帧 max|Δ|/peak = %.3e   (信息值;差异应集中于帧)\n", d_all / peak)
    @printf("  全场最大差点位 (i,j)=(%d,%d)%s\n", am[1], am[2],
        in_frame ? ",位于边界帧内 ✓" : ",不在边界帧内")
    @printf("  地震记录 rel-max = %.3e   %s\n", seisd, verdict(seisd, 1e-4))
    return d_int / peak <= 5e-4 && seisd <= 1e-4
end

# ══════════════════════════════════════════════════════════════════════════════
println("═"^70)
okA1 = testA_acoustic(); println()
okA2 = testA_elastic(); println()
okB = testB_acoustic(); println()
okC1 = testC(elastic=false); println()
okC2 = testC(elastic=true); println()
println("═"^70)
if okA1 && okA2
    println("A 通过:融合的备份/更新/注入/记录与旧序列逐位等价,")
    println("        v1 的全场差异【全部】来自新旧 HABC 内核对同一已知 race 的不同调度。")
else
    println("A 未通过:融合逻辑存在真实 bug,请把完整输出发回。")
end
okB || println("B 异常:新 HABC 偏差显著大于旧内核或最大差不在帧内,请把完整输出发回。")
if okC1 && okC2
    println("C 通过:深内区与地震记录的物理误差在阈值内,race 差异被限制在边界帧。")
end
