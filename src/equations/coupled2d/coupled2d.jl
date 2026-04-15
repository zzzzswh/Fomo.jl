# src/equations/coupled2d/coupled2d.jl
#
# 耦合 P-S 势场 2D 模拟入口
#
# ═══════════════════════════════════════════════════════════════════
# 吸收边界：二阶等效 HABC
#
# 核心思想：将 leapfrog 拆成与 velocity-stress 同构的两个半步
#
#   半步 1（"速度"更新）：dPdt += dt·RHS  →  HABC on dPdt, dSdt
#   半步 2（"位置"更新）：P += dt·dPdt    →  HABC on P, S
#
# 每步两次 HABC → 等效二阶 Higdon ABC，与 elastic2d 吸收能力一致
# ═══════════════════════════════════════════════════════════════════

using CUDA
using StaticArrays

"""
    coupled2d(vp, vs, dh, dt, nt, f0; kwargs...) -> (seis_P, seis_S, snaps_P, snaps_S)

2D 耦合 P-S 势场模拟（常密度 ρ=1）。

# 参数
- `vp, vs`: [nx × nz] Float32 速度模型（单位 m/s）
- `dh`: 网格间距 (m)
- `dt`: 时间步长 (s)
- `nt`: 时间步数
- `f0`: 震源主频 (Hz)

# 关键字参数
- `sx, sz`: 震源坐标向量（网格索引）
- `rx, rz`: 接收器坐标向量（网格索引）
- `nbc`: 吸收边界层数 (default: 50)
- `fd_order`: 有限差分阶数 (default: 8)
- `snap_interval`: 快照间隔，0=不保存 (default: 0)
- `v_ref_p`: P 场 HABC 参考速度 (default: min(vp))
- `v_ref_s`: S 场 HABC 参考速度 (default: min(vs))

# 返回
- `seis_P`: [n_rec × nt] P 势场地震记录
- `seis_S`: [n_rec × nt] S 势场地震记录
- `snaps_P`: P 势场快照序列
- `snaps_S`: S 势场快照序列
"""
function coupled2d(
    vp::Matrix{Float32}, vs::Matrix{Float32},
    dh::Float32, dt::Float32, nt::Int, f0::Float32;
    sx::Vector{Int},
    sz::Vector{Int},
    rx::AbstractVector{Int},
    rz::AbstractVector{Int},
    nbc::Int=50,
    fd_order::Int=8,
    snap_interval::Int=0,
    v_ref_p::Float32=minimum(vp[vp.>0.0f0]),
    v_ref_s::Float32=minimum(vs[vs.>0.0f0])
)
    nx, nz = size(vp)

    # ── 有限差分系数（正则网格中心差分）──
    d1 = get_centered_d1(fd_order)
    d2_c0, d2 = get_centered_d2(fd_order)

    # ── 介质初始化 ──
    medium = init_coupled_medium(vp, vs, dh, nbc, fd_order)

    # ── 波场初始化（8 个标量场：4 物理 + 4 HABC 备份）──
    W = CoupledWavefield(nx, nz, medium.pad)

    # ── HABC 边界条件（P 和 S 各自的参考速度）──
    habc_p = init_habc(nx, nz, medium.pad, dt, dh, v_ref_p)
    habc_s = init_habc(nx, nz, medium.pad, dt, dh, v_ref_s)

    # ── 震源 ──
    wavelet_data = ricker_wavelet(f0, dt, nt)
    wavelet_matrix = reshape(wavelet_data, 1, nt)
    source = init_source(medium.pad, Int32.(sx), Int32.(sz), wavelet_matrix)

    # ── 接收器 ──
    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :p)

    # ── 分配地震记录 ──
    num_receivers = length(receiver.rx)
    seis_P = CUDA.zeros(Float32, num_receivers, nt)
    seis_S = CUDA.zeros(Float32, num_receivers, nt)

    # ── 分配快照 ──
    if snap_interval > 0
        num_snaps = nt ÷ snap_interval
        snaps_P = Vector{Matrix{Float32}}(undef, num_snaps)
        snaps_S = Vector{Matrix{Float32}}(undef, num_snaps)
    else
        snaps_P = Vector{Matrix{Float32}}()
        snaps_S = Vector{Matrix{Float32}}()
    end

    # ── 预计算 ──
    pad = medium.pad
    inner_nx = medium.nx - fd_order
    inner_nz = medium.nz - fd_order

    # HABC 标量参数（循环外提取，避免重复字段访问）
    nbc_i  = Int32(habc_p.nbc)
    nx_i   = Int32(medium.nx)
    nz_i   = Int32(medium.nz)

    qx_p   = Float32(habc_p.qx)
    qz_p   = Float32(habc_p.qz)
    qt_x_p = Float32(habc_p.qt_x)
    qt_z_p = Float32(habc_p.qt_z)
    qxt_p  = Float32(habc_p.qxt)

    qx_s   = Float32(habc_s.qx)
    qz_s   = Float32(habc_s.qz)
    qt_x_s = Float32(habc_s.qt_x)
    qt_z_s = Float32(habc_s.qt_z)
    qxt_s  = Float32(habc_s.qxt)

    # ── Warmup ──
    @info "Warming up coupled2d kernels..."
    _coupled2d_loop!(W, medium, source, receiver,
        d1, d2, d2_c0, dt, 1, inner_nx, inner_nz, pad,
        habc_p, habc_s,
        nbc_i, nx_i, nz_i,
        qx_p, qz_p, qt_x_p, qt_z_p, qxt_p,
        qx_s, qz_s, qt_x_s, qt_z_s, qxt_s,
        seis_P, seis_S, snaps_P, snaps_S, 0)
    CUDA.synchronize()

    # ── Reset ──
    reset!(W)
    fill!(seis_P, 0.0f0)
    fill!(seis_S, 0.0f0)

    # ── Run ──
    @info "Starting coupled2d simulation... (nx=$nx, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _coupled2d_loop!(W, medium, source, receiver,
            d1, d2, d2_c0, dt, nt, inner_nx, inner_nz, pad,
            habc_p, habc_s,
            nbc_i, nx_i, nz_i,
            qx_p, qz_p, qt_x_p, qt_z_p, qxt_p,
            qx_s, qz_s, qt_x_s, qt_z_s, qxt_s,
            seis_P, seis_S, snaps_P, snaps_S, snap_interval)
    end
    @info "Simulation complete! GPU time: $(round(elapsed, digits=3))s"

    return Array(seis_P), Array(seis_S), snaps_P, snaps_S
end

"""
内部时间步循环。

结构与 elastic2d 完全同构：

  elastic2d:                          coupled2d:
  ─────────────────────────           ─────────────────────────
  backup_velocity!                    backup dPdt, dSdt
  update_velocity!                    update_coupled_velocity!
  apply_habc_velocity!                HABC on dPdt, dSdt
  ─────────────────────────           ─────────────────────────
  backup_stress!                      backup P, S
  update_stress!                      update_coupled_position!
  apply_habc_stress!                  HABC on P, S
  ─────────────────────────           ─────────────────────────
  inject_source!(txx, tzz)            inject_source!(P)
  record_receivers!(vx, vz)           record_receivers!(P, S)
"""
function _coupled2d_loop!(
    W, M, S, R,
    d1, d2, d2_c0, dt, nt, inner_nx, inner_nz, pad,
    habc_p, habc_s,
    nbc, nx, nz,
    qx_p, qz_p, qt_x_p, qt_z_p, qxt_p,
    qx_s, qz_s, qt_x_s, qt_z_s, qxt_s,
    seis_P, seis_S, snaps_P, snaps_S, snap_interval
)
    snap_idx = 1

    for it in 1:nt
        # ══════════════════════════════════════════════════════════
        # A. "速度"半步：dPdt += dt·RHS, dSdt += dt·RHS
        #    对应 elastic2d 的 velocity phase
        # ══════════════════════════════════════════════════════════

        # A1. 边界备份（只拷贝边界区域，高效）
        backup_single_field!(W.dPdt_old, W.dPdt, nbc, nx, nz)
        backup_single_field!(W.dSdt_old, W.dSdt, nbc, nx, nz)

        # A2. 更新 dPdt, dSdt（计算量集中在这里）
        update_coupled_velocity!(W, M, d1, d2, d2_c0, dt, inner_nx, inner_nz)

        # A3. HABC on "速度"场（用 w_vx 权重，与弹性波一致）
        apply_habc_single_field!(W.dPdt, W.dPdt_old, habc_p.w_vx,
            qx_p, qz_p, qt_x_p, qt_z_p, qxt_p, nx, nz, nbc)
        apply_habc_single_field!(W.dSdt, W.dSdt_old, habc_s.w_vx,
            qx_s, qz_s, qt_x_s, qt_z_s, qxt_s, nx, nz, nbc)

        # ══════════════════════════════════════════════════════════
        # B. "位置"半步：P += dt·dPdt, S += dt·dSdt
        #    对应 elastic2d 的 stress phase
        # ══════════════════════════════════════════════════════════

        # B1. 边界备份
        backup_single_field!(W.P_old, W.P, nbc, nx, nz)
        backup_single_field!(W.S_old, W.S, nbc, nx, nz)

        # B2. 更新 P, S（简单逐点加法）
        update_coupled_position!(W, dt, M.M, inner_nx, inner_nz)

        # B3. HABC on "位置"场（用 w_tau 权重，与弹性波一致）
        apply_habc_single_field!(W.P, W.P_old, habc_p.w_tau,
            qx_p, qz_p, qt_x_p, qt_z_p, qxt_p, nx, nz, nbc)
        apply_habc_single_field!(W.S, W.S_old, habc_s.w_tau,
            qx_s, qz_s, qt_x_s, qt_z_s, qxt_s, nx, nz, nbc)

        # ══════════════════════════════════════════════════════════
        # C. 震源注入（爆炸源 → P 势场，与 elastic2d 注入 txx+tzz 对应）
        # ══════════════════════════════════════════════════════════
        inject_source!(W.P, S, it, dt)

        # ══════════════════════════════════════════════════════════
        # D. 记录接收器
        # ══════════════════════════════════════════════════════════
        record_receivers!(seis_P, W.P, R, it)
        record_receivers!(seis_S, W.S, R, it)

        # ══════════════════════════════════════════════════════════
        # E. 快照
        # ══════════════════════════════════════════════════════════
        if snap_interval > 0 && it % snap_interval == 0
            snaps_P[snap_idx] = Array(@view W.P[pad+1:end-pad, pad+1:end-pad])
            snaps_S[snap_idx] = Array(@view W.S[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end
