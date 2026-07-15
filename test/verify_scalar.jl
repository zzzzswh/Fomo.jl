# test/verify_scalar.jl — scalar2d(二阶标量求解器)验收(需 GPU)
#
# 三段:
#   1. GPU ↔ CPU 同格式参考:纯 Julia 复刻同一离散格式(全网格携带语义 +
#      两遍 HABC + 同序注入/记录),150 步后比对终态场与记录。
#      期望仅 FMA 舍入的线性积累(~1e-6 相对),阈值 1e-4。
#   2. 网格自收敛:dh→dh/2(dt→dt/2, 源/检波点按 x=(i-1)dh 映射 i₂=2i−1),
#      粗细两套地震记录的相对 L2 差应明显小(阈值 <0.1,报告实际值)。
#      注意用 source_scale=:none:同时细化 dt、dh 时 (v·dt/dh)² 不变,
#      裸注入天然收敛;:v2dt2 是"固定 dh、细分 dt"的 deepwave 语义,
#      在本测试的联合细化下会人为造成 4× 振幅差。
#   3. 吞吐:scalar2d 循环 vs 一阶声波 det 循环 / graph 循环。
#      预期 memory-bound 网格上 scalar ≈ 一阶的 2.5~3 倍(4 vs ~12 次遍历)。
#
# 用法:julia --project=. test/verify_scalar.jl
#
using CUDA
using Printf
using StaticArrays
using Fomo
using Fomo: init_acoustic_medium, AcousticWavefield,
    init_habc, init_source, init_receiver, get_fd_coefficients, get_centered_d2,
    ricker_wavelet, reset!, _pad_array,
    _acoustic2d_loop_fused!, _acoustic2d_loop_graph!,
    _scalar2d_loop!, apply_habc_det_1!, _habc_frame_total,
    inject_source!, record_receivers!

maxabs(a, b) = maximum(abs.(Array(a) .- Array(b)))
verdict(x, thr) = x <= thr ? "OK " : "FAIL"

# ══════════════ 1. GPU ↔ CPU 同格式参考 ══════════════
"CPU 版单步:全网格携带语义的 leapfrog + 两遍 Higdon(全读旧值)。"
function cpu_scalar_step!(u_next::Matrix{Float32}, u::Matrix{Float32},
    u_prev::Matrix{Float32}, c2::Matrix{Float32},
    c0::Float32, d2, M::Int, w::Matrix{Float32},
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32, nbc::Int)
    nx, nz = size(u)
    L = length(d2)
    # a) leapfrog(内区)/ 携带(外圈)
    for j in 1:nz, i in 1:nx
        uc = u[i, j]
        un = uc
        if (i > M) && (i <= nx - M) && (j > M) && (j <= nz - M)
            lap = 2.0f0 * c0 * uc
            @inbounds for l in 1:L
                cl = d2[l]
                lap += cl * (u[i+l, j] + u[i-l, j] + u[i, j+l] + u[i, j-l])
            end
            un = 2.0f0 * uc - u_prev[i, j] + c2[i, j] * lap
        end
        u_next[i, j] = un
    end
    # b) 两遍 Higdon:f = u_next(全读修正前值),f_old = u
    fpre = copy(u_next)
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
                    (-qx * fpre[i+1, j] - qt_x * u[i, j] - qxt * u[i+1, j]) :
                    (-qx * fpre[i-1, j] - qt_x * u[i, j] - qxt * u[i-1, j])
            sum_z = is_top ?
                    (-qz * fpre[i, j+1] - qt_z * u[i, j] - qxt * u[i, j+1]) :
                    (-qz * fpre[i, j-1] - qt_z * u[i, j] - qxt * u[i, j-1])
            u_next[i, j] = wt * f_cur + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        elseif in_x
            sum_x = is_left ?
                    (-qx * fpre[i+1, j] - qt_x * u[i, j] - qxt * u[i+1, j]) :
                    (-qx * fpre[i-1, j] - qt_x * u[i, j] - qxt * u[i-1, j])
            u_next[i, j] = wt * f_cur + (1.0f0 - wt) * sum_x
        else
            sum_z = is_top ?
                    (-qz * fpre[i, j+1] - qt_z * u[i, j] - qxt * u[i, j+1]) :
                    (-qz * fpre[i, j-1] - qt_z * u[i, j] - qxt * u[i, j-1])
            u_next[i, j] = wt * f_cur + (1.0f0 - wt) * sum_z
        end
    end
    return nothing
end

function sec1()
    nx, nz, nt = 120, 100, 150
    dh = 10.0f0; dt = 0.001f0; f0 = 15.0f0
    nbc_user = 30; fd_order = 8
    vp = fill(3000.0f0, nx, nz); vp[:, nz÷2:end] .= 3600.0f0
    c0, d2 = get_centered_d2(fd_order)
    M = fd_order ÷ 2
    pad = nbc_user + M
    nxp, nzp = nx + 2pad, nz + 2pad

    c2h = _pad_array(vp, pad)
    c2h .= (c2h .* Float32(dt / dh)) .^ 2
    bc = init_habc(nx, nz, pad, dt, dh, minimum(vp))
    w_h = Array(bc.w_tau)
    nbc = Int(bc.nbc)

    wsig = ricker_wavelet(f0, dt, nt)
    sxr, szr = nx ÷ 2, 8
    scale = Float32(vp[sxr, szr]^2 * dt^2)
    rxs = collect(10:10:nx-10)
    rzs = fill(6, length(rxs))

    # ── GPU:直接调生产循环 ──
    wm = repeat(reshape(wsig .* scale, 1, nt), 1, 1)
    src = init_source(pad, Int32[sxr], Int32[szr], wm)
    rec = init_receiver(pad, Int32.(rxs), Int32.(rzs), :p)
    seis_g = CUDA.zeros(Float32, length(rxs), nt)
    ua = CUDA.zeros(Float32, nxp, nzp)
    ub = CUDA.zeros(Float32, nxp, nzp)
    c2 = CuArray(c2h)
    scr = CUDA.zeros(Float32, _habc_frame_total(Int32(nxp), Int32(nzp), Int32(nbc)))
    u_fin = _scalar2d_loop!(ua, ub, c2, src, rec, bc,
        c0, d2, dt, nt, Int32(M), Int32(nbc), Int32(nxp), Int32(nzp),
        Float32(bc.qx), Float32(bc.qz), Float32(bc.qt_x), Float32(bc.qt_z), Float32(bc.qxt),
        seis_g, scr)
    CUDA.synchronize()

    # ── CPU:同格式逐步复刻 ──
    u_old = zeros(Float32, nxp, nzp)
    u_cur = zeros(Float32, nxp, nzp)
    u_nxt = zeros(Float32, nxp, nzp)
    seis_c = zeros(Float32, length(rxs), nt)
    six, siz = sxr + pad, szr + pad
    for it in 1:nt
        cpu_scalar_step!(u_nxt, u_cur, u_old, c2h, c0, d2, M, w_h,
            Float32(bc.qx), Float32(bc.qz), Float32(bc.qt_x), Float32(bc.qt_z), Float32(bc.qxt), nbc)
        u_nxt[six, siz] += wsig[it] * scale
        for (r, x) in enumerate(rxs)
            seis_c[r, it] = u_nxt[x+pad, rzs[r]+pad]
        end
        u_old, u_cur, u_nxt = u_cur, u_nxt, u_old
    end

    peak = max(maximum(abs.(Array(u_fin))), 1.0f-30)
    d_f = maxabs(u_fin, u_cur) / peak
    d_s = maxabs(seis_g, seis_c) / max(maximum(abs.(seis_c)), 1.0f-30)
    println("── 1. scalar2d GPU ↔ CPU 同格式参考,$(nt) 步(相对峰值)──")
    @printf("  终态场   rel-max = %.3e   %s   (期望 FMA 积累,~1e-6)\n", d_f, verdict(d_f, 1e-4))
    @printf("  地震记录 rel-max = %.3e   %s\n", d_s, verdict(d_s, 1e-4))
    return d_f <= 1e-4 && d_s <= 1e-4
end

# ══════════════ 2. 网格自收敛 ══════════════
function sec2()
    nx, nz, nt = 200, 160, 500
    dh = 10.0f0; dt = 0.001f0; f0 = 12.0f0
    vp1 = fill(3000.0f0, nx, nz)
    sx1, sz1 = nx ÷ 2, 10
    rx1 = collect(30:20:nx-30)
    rz1 = fill(12, length(rx1))

    r_coarse = scalar2d(vp1, dh, dt, nt, f0;
        sx=[sx1], sz=[sz1], rx=rx1, rz=rz1,
        source_scale=:none, verbose=false)

    # 细网格:x=(i−1)dh 映射 i₂ = 2i − 1
    nx2, nz2 = 2nx - 1, 2nz - 1
    vp2 = fill(3000.0f0, nx2, nz2)
    r_fine = scalar2d(vp2, dh / 2, dt / 2, 2nt, f0;
        sx=[2sx1 - 1], sz=[2sz1 - 1],
        rx=2 .* rx1 .- 1, rz=2 .* rz1 .- 1,
        source_scale=:none, verbose=false)

    sc = r_coarse.seis_u
    sf = r_fine.seis_u[:, 2:2:end]     # 时刻 n·dt ↔ 细网格第 2n 步
    rel = sqrt(sum(abs2, sc .- sf) / max(sum(abs2, sf), 1.0e-30))
    println("── 2. 网格自收敛:dh=10 vs dh=5(同物理坐标源/检波)──")
    @printf("  地震记录 rel-L2(粗 vs 细) = %.3e   %s   (一致格式应明显小)\n",
        rel, verdict(rel, 0.1))
    return rel <= 0.1
end

# ══════════════ 3. 吞吐:scalar vs 一阶声波 ══════════════
function bench_scalar(; nx, nz, nt)
    dh = 10.0f0; dt = 0.001f0; f0 = 15.0f0
    nbc_user = 50; fd_order = 8
    vp = fill(3000.0f0, nx, nz); vp[:, nz÷2:end] .= 3800.0f0
    rho = fill(2200.0f0, nx, nz)
    rxs = Int32.(collect(10:5:nx-10))
    rzs = fill(Int32(6), length(rxs))
    wsig = ricker_wavelet(f0, dt, nt)
    wm = repeat(reshape(wsig, 1, nt), 1, 1)
    nosnap = Matrix{Float32}[]

    # 一阶声波(det 融合 + graph)
    a = get_fd_coefficients(fd_order)
    med = init_acoustic_medium(vp, rho, dh, nbc_user, fd_order)
    bc1 = init_habc(nx, nz, med.pad, dt, dh, minimum(vp))
    src1 = init_source(med.pad, Int32[nx ÷ 2], Int32[8], wm)
    rec1 = init_receiver(med.pad, rxs, rzs, :p)
    nrec = length(rec1.rx)
    W = AcousticWavefield(nx, nz, med.pad)
    b = Tuple(CUDA.zeros(Float32, nrec, nt) for _ in 1:3)
    nbc1 = Int32(bc1.nbc); nx1 = Int32(med.nx); nz1 = Int32(med.nz)

    function det_run!(n)
        _acoustic2d_loop_fused!(W, med, src1, rec1, bc1, a, dt, n, med.pad,
            nbc1, nx1, nz1, bc1.qx, bc1.qz, bc1.qt_x, bc1.qt_z, bc1.qxt,
            b[1], b[2], b[3], nosnap, 0)
        return nothing
    end
    function graph_run!(n)
        _acoustic2d_loop_graph!(W, med, src1, rec1, bc1, a, dt, n, med.pad,
            nbc1, nx1, nz1, bc1.qx, bc1.qz, bc1.qt_x, bc1.qt_z, bc1.qxt,
            b[1], b[2], b[3])
        return nothing
    end

    # scalar
    c0, d2 = get_centered_d2(fd_order)
    M = fd_order ÷ 2
    pad = nbc_user + M
    nxp, nzp = nx + 2pad, nz + 2pad
    c2h = _pad_array(vp, pad)
    c2h .= (c2h .* Float32(dt / dh)) .^ 2
    c2 = CuArray(c2h)
    bc2 = init_habc(nx, nz, pad, dt, dh, minimum(vp))
    src2 = init_source(pad, Int32[nx ÷ 2], Int32[8], wm)
    rec2 = init_receiver(pad, rxs, rzs, :p)
    seis_u = CUDA.zeros(Float32, nrec, nt)
    ua = CUDA.zeros(Float32, nxp, nzp)
    ub = CUDA.zeros(Float32, nxp, nzp)
    scr = CUDA.zeros(Float32, _habc_frame_total(Int32(nxp), Int32(nzp), Int32(bc2.nbc)))
    function scalar_run!(n)
        _scalar2d_loop!(ua, ub, c2, src2, rec2, bc2,
            c0, d2, dt, n, Int32(M), Int32(bc2.nbc), Int32(nxp), Int32(nzp),
            Float32(bc2.qx), Float32(bc2.qz), Float32(bc2.qt_x), Float32(bc2.qt_z), Float32(bc2.qxt),
            seis_u, scr)
        return nothing
    end

    det_run!(5)
    reset!(W)
    e_det = CUDA.@elapsed begin
        det_run!(nt)
    end
    graph_run!(5)
    reset!(W)
    e_graph = CUDA.@elapsed begin
        graph_run!(nt)
    end
    scalar_run!(5)
    fill!(ua, 0.0f0)
    fill!(ub, 0.0f0)
    e_sca = CUDA.@elapsed begin
        scalar_run!(nt)
    end
    return Float64(e_det), Float64(e_graph), Float64(e_sca)
end

function sec3()
    println("── 3. 吞吐:一阶声波(det / graph) vs 二阶标量 ──")
    @printf("%-12s %5s | %10s %10s %10s | %12s\n",
        "grid", "nt", "det st/s", "graph st/s", "scalar st/s", "scalar/graph")
    println("-"^72)
    for (nx, nz, nt) in ((240, 200, 1000), (500, 400, 600), (1000, 800, 300), (2000, 1600, 150))
        td, tg, ts = bench_scalar(nx=nx, nz=nz, nt=nt)
        @printf("%-12s %5d | %10.1f %10.1f %10.1f | %11.2fx\n",
            "$(nx)x$(nz)", nt, nt / td, nt / tg, nt / ts, tg / ts)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
println("═"^70)
ok1 = sec1(); println()
ok2 = sec2(); println()
sec3()
println("═"^70)
if ok1 && ok2
    println("通过:GPU 实现与预期离散格式一致,格式自收敛;可用于与 deepwave scalar")
    println("      的同公式正面对比(协议见报告附录 A,accuracy 同阶、v²dt² 源缩放)。")
else
    println("存在未通过项,请把完整输出发回。")
end
