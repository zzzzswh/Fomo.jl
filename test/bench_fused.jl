# test/bench_fused.jl — 旧内核序列 vs 融合内核序列 吞吐基准（需 GPU）
#
# 与 compare_fused.jl 相同的物理设置，但只计时不比数值。
# 每个 case 用同一块波场：先跑旧序列计时，reset 后跑新序列计时，
# 避免同时分配两套波场撑爆显存。
#
# 用法：julia --project=. test/bench_fused.jl
#
# 预期规律：
#   - 小/中网格（host-bound 区）：融合序列加速最明显（2×+），
#     因为每步 launch 从 12/13 次降到 6 次，且备份/HABC 不再空转全网格线程；
#   - 大网格（memory-bound 区）：加速收敛到备份+HABC 省下的带宽份额（~1.2-1.5×）。
#
using CUDA
using Printf
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

function make_setup(; nx, nz, nt, nbc=50, fd_order=8, elastic=false)
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
    src = init_source(med.pad, Int32[nx ÷ 2], Int32[8], wm)
    rxs = Int32.(collect(10:5:nx-10))
    rec = init_receiver(med.pad, rxs, fill(Int32(6), length(rxs)), elastic ? :vz : :p)
    return med, bc, src, rec, a, dt
end

# ─────────────────────────── 声波 ───────────────────────────
function bench_acoustic(; nx, nz, nt)
    med, bc, src, rec, a, dt = make_setup(nx=nx, nz=nz, nt=nt, elastic=false)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nxi = Int32(nxp); nzi = Int32(nzp)
    nrec = length(rec.rx)
    W = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    sp  = CUDA.zeros(Float32, nrec, nt)
    svx = CUDA.zeros(Float32, nrec, nt)
    svz = CUDA.zeros(Float32, nrec, nt)

    step_old!(it) = begin
        backup_single_field!(W.vx_old, W.vx, nbc, nxi, nzi)
        backup_single_field!(W.vz_old, W.vz, nbc, nxi, nzi)
        update_velocity_acoustic!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W.vx, W.vx_old, bc.w_vx, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        apply_habc_single_field!(W.vz, W.vz_old, bc.w_vz, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        backup_single_field!(W.p_old, W.p, nbc, nxi, nzi)
        update_pressure_acoustic!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W.p, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        inject_source!(W.p, src, it, dt)
        record_receivers!(sp, W.p, rec, it)
        record_receivers!(svx, W.vx, rec, it)
        record_receivers!(svz, W.vz, rec, it)
    end
    step_new!(it) = begin
        fused_update_velocity_acoustic!(W, med, a, dt, nbc)
        apply_habc_frame_2!(W.vx, W.vx_old, bc.w_vx, W.vz, W.vz_old, bc.w_vz,
            bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        fused_update_pressure_acoustic!(W, med, a, dt, nbc)
        apply_habc_frame_1!(W.p, W.p_old, bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        inject_source!(W.p, src, it, dt)
        record_receivers3!(sp, svx, svz, W, rec, it)
    end

    t_old = run_timed!(step_old!, W, nt)
    t_new = run_timed!(step_new!, W, nt)
    return t_old, t_new
end

# ─────────────────────────── 弹性波 ───────────────────────────
function bench_elastic(; nx, nz, nt)
    med, bc, src, rec, a, dt = make_setup(nx=nx, nz=nz, nt=nt, elastic=true)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nxi = Int32(nxp); nzi = Int32(nzp)
    nrec = length(rec.rx)
    W = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    svx = CUDA.zeros(Float32, nrec, nt)
    svz = CUDA.zeros(Float32, nrec, nt)

    step_old!(it) = begin
        backup_velocity!(W, bc, med)
        update_velocity!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_velocity!(W, bc, med)
        backup_stress!(W, bc, med)
        update_stress!(W, med, a, dt, inner_nx, inner_nz)
        apply_habc_stress!(W, bc, med)
        inject_source!(W.txx, src, it, dt)
        inject_source!(W.tzz, src, it, dt)
        record_receivers!(svx, W.vx, rec, it)
        record_receivers!(svz, W.vz, rec, it)
    end
    step_new!(it) = begin
        fused_update_velocity_elastic!(W, med, a, dt, nbc)
        apply_habc_frame_2!(W.vx, W.vx_old, bc.w_vx, W.vz, W.vz_old, bc.w_vz,
            bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        fused_update_stress_elastic!(W, med, a, dt, nbc)
        apply_habc_frame_3!(W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
            bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nxi, nzi, nbc)
        inject_source_pair!(W.txx, W.tzz, src, it)
        record_receivers2!(svx, svz, W, rec, it)
    end

    t_old = run_timed!(step_old!, W, nt)
    t_new = run_timed!(step_new!, W, nt)
    return t_old, t_new
end

"warmup 5 步 → reset → 计时 nt 步（CUDA.@elapsed 自带同步）"
function run_timed!(step!, W, nt)
    for it in 1:5
        step!(it)
    end
    CUDA.synchronize()
    reset!(W)
    t = CUDA.@elapsed begin
        for it in 1:nt
            step!(it)
        end
    end
    reset!(W)
    return Float64(t)
end

# ─────────────────────────── 主程序 ───────────────────────────
function main()
    dev = CUDA.device()
    println("GPU: ", CUDA.name(dev))
    println()
    @printf("%-10s %-12s %6s | %9s %9s | %11s %11s | %7s\n",
        "eq", "grid", "nt", "old(s)", "new(s)", "old st/s", "new st/s", "speedup")
    println("-"^96)
    cases = [(240, 200, 1000), (500, 400, 600), (1000, 800, 300), (2000, 1600, 150)]
    for (nx, nz, nt) in cases
        t_o, t_n = bench_acoustic(nx=nx, nz=nz, nt=nt)
        @printf("%-10s %-12s %6d | %9.4f %9.4f | %11.1f %11.1f | %6.2fx\n",
            "acoustic", "$(nx)x$(nz)", nt, t_o, t_n, nt / t_o, nt / t_n, t_o / t_n)
    end
    println("-"^96)
    for (nx, nz, nt) in cases
        t_o, t_n = bench_elastic(nx=nx, nz=nz, nt=nt)
        @printf("%-10s %-12s %6d | %9.4f %9.4f | %11.1f %11.1f | %6.2fx\n",
            "elastic", "$(nx)x$(nz)", nt, t_o, t_n, nt / t_o, nt / t_n, t_o / t_n)
    end
    println()
    println("提示：与 deepwave 对比请用相同 (grid, dt, nt, 单炮/批炮数) 并关闭 deepwave 的")
    println("      内部时间步细分差异影响（其 dt 会被 CFL 细分，实际 step 数可能更多）。")
end

main()
