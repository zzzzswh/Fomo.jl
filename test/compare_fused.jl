# test/compare_fused.jl
#
# 融合内核正确性验证：在同一进程内分别用「旧内核序列」与「新融合序列」
# 各推进 nt 步，逐场对比。
#
# 预期：
#   - 备份融合、检波融合、注入融合是逐位等价的重排；
#   - HABC 帧映射内核与旧全网格内核数学一致，但两者都保留了原实现
#     文档中说明的邻点 race，因此允许极小的非确定性差异。
#   - 判据：max|Δ| / max|field| < 1e-4（经验上应远小于此）。
#
# 用法：julia --project test/compare_fused.jl
#
using CUDA
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

relmax(a, b) = begin
    d = maximum(abs.(Array(a) .- Array(b)))
    m = max(maximum(abs.(Array(a))), 1.0f-30)
    d / m
end

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
    rec = init_receiver(med.pad, Int32.(collect(10:5:nx-10)), fill(Int32(6), length(10:5:nx-10)),
        elastic ? :vz : :p)
    return med, bc, src, rec, a, dt, nt
end

# ─────────────────────────── 声波 ───────────────────────────
function acoustic_compare()
    med, bc, src, rec, a, dt, nt = make_setup(elastic=false)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    qx = bc.qx; qz = bc.qz; qt_x = bc.qt_x; qt_z = bc.qt_z; qxt = bc.qxt
    nrec = length(rec.rx)

    W1 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = AcousticWavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    s1 = (CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt))
    s2 = (CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt))

    for it in 1:nt
        # ── 旧序列（12 launch）──
        backup_single_field!(W1.vx_old, W1.vx, nbc, nx, nz)
        backup_single_field!(W1.vz_old, W1.vz, nbc, nx, nz)
        update_velocity_acoustic!(W1, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W1.vx, W1.vx_old, bc.w_vx, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        apply_habc_single_field!(W1.vz, W1.vz_old, bc.w_vz, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        backup_single_field!(W1.p_old, W1.p, nbc, nx, nz)
        update_pressure_acoustic!(W1, med, a, dt, inner_nx, inner_nz)
        apply_habc_single_field!(W1.p, W1.p_old, bc.w_tau, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        inject_source!(W1.p, src, it, dt)
        record_receivers!(s1[1], W1.p, rec, it)
        record_receivers!(s1[2], W1.vx, rec, it)
        record_receivers!(s1[3], W1.vz, rec, it)

        # ── 新序列（6 launch）──
        fused_update_velocity_acoustic!(W2, med, a, dt, nbc)
        apply_habc_frame_2!(W2.vx, W2.vx_old, bc.w_vx, W2.vz, W2.vz_old, bc.w_vz,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        fused_update_pressure_acoustic!(W2, med, a, dt, nbc)
        apply_habc_frame_1!(W2.p, W2.p_old, bc.w_tau, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        inject_source!(W2.p, src, it, dt)
        record_receivers3!(s2[1], s2[2], s2[3], W2, rec, it)
    end
    CUDA.synchronize()

    println("── acoustic2d  (nt=$nt) ──")
    for (name, f1, f2) in (("p", W1.p, W2.p), ("vx", W1.vx, W2.vx), ("vz", W1.vz, W2.vz),
        ("seis_p", s1[1], s2[1]), ("seis_vx", s1[2], s2[2]), ("seis_vz", s1[3], s2[3]))
        r = relmax(f1, f2)
        ok = r < 1.0f-4 ? "OK " : "FAIL"
        println(rpad(name, 8), " rel-max-diff = ", r, "   ", ok)
    end
end

# ─────────────────────────── 弹性波 ───────────────────────────
function elastic_compare()
    med, bc, src, rec, a, dt, nt = make_setup(elastic=true)
    nxp, nzp = med.nx, med.nz
    inner_nx = nxp - 2 * med.M
    inner_nz = nzp - 2 * med.M
    nbc = Int32(bc.nbc); nx = Int32(nxp); nz = Int32(nzp)
    nrec = length(rec.rx)

    W1 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    W2 = Wavefield(nxp - 2med.pad, nzp - 2med.pad, med.pad)
    s1 = (CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt))
    s2 = (CUDA.zeros(Float32, nrec, nt), CUDA.zeros(Float32, nrec, nt))

    for it in 1:nt
        # ── 旧序列（13 launch）──
        backup_velocity!(W1, bc, med)
        update_velocity!(W1, med, a, dt, inner_nx, inner_nz)
        apply_habc_velocity!(W1, bc, med)
        backup_stress!(W1, bc, med)
        update_stress!(W1, med, a, dt, inner_nx, inner_nz)
        apply_habc_stress!(W1, bc, med)
        inject_source!(W1.txx, src, it, dt)
        inject_source!(W1.tzz, src, it, dt)
        record_receivers!(s1[1], W1.vx, rec, it)
        record_receivers!(s1[2], W1.vz, rec, it)

        # ── 新序列（6 launch）──
        fused_update_velocity_elastic!(W2, med, a, dt, nbc)
        apply_habc_frame_2!(W2.vx, W2.vx_old, bc.w_vx, W2.vz, W2.vz_old, bc.w_vz,
            bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        fused_update_stress_elastic!(W2, med, a, dt, nbc)
        apply_habc_frame_3!(W2.txx, W2.txx_old, W2.tzz, W2.tzz_old, W2.txz, W2.txz_old,
            bc.w_tau, bc.qx, bc.qz, bc.qt_x, bc.qt_z, bc.qxt, nx, nz, nbc)
        inject_source_pair!(W2.txx, W2.tzz, src, it)
        record_receivers2!(s2[1], s2[2], W2, rec, it)
    end
    CUDA.synchronize()

    println("── elastic2d  (nt=$nt) ──")
    for (name, f1, f2) in (("vx", W1.vx, W2.vx), ("vz", W1.vz, W2.vz),
        ("txx", W1.txx, W2.txx), ("tzz", W1.tzz, W2.tzz), ("txz", W1.txz, W2.txz),
        ("seis_vx", s1[1], s2[1]), ("seis_vz", s1[2], s2[2]))
        r = relmax(f1, f2)
        ok = r < 1.0f-4 ? "OK " : "FAIL"
        println(rpad(name, 8), " rel-max-diff = ", r, "   ", ok)
    end
end

acoustic_compare()
elastic_compare()
