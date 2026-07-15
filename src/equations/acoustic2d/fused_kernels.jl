# src/equations/acoustic2d/fused_kernels.jl
#
# 声波融合内核（性能补丁核心之一）
#
# 旧 HABC 路径每个时间步 12 次 kernel launch：
#   backup(vx) + backup(vz) + update_v + habc(vx) + habc(vz)
#   + backup(p) + update_p + habc(p) + inject + record×3
#
# 新路径 6 次：
#   fused_update_v(备份+更新) + habc_frame_2(vx,vz)
#   + fused_update_p(备份+更新) + habc_frame_1(p)
#   + inject + record3
#
# 备份融合的正确性：
#   velocity 更新只读 p 的邻域、只写本点 vx/vz；备份只读写本点 vx/vz。
#   同一 kernel 内"先备份本点、再更新本点"与旧的"全场备份→全场更新"
#   逐位等价（更新读的是另一组场，不存在跨线程读写冲突）。
#   pressure / elastic 同理。
#
using CUDA
using StaticArrays

# ──────────────────────────────────────────────────────────────────────────────
# 融合 kernel：边界带备份 + 全场更新（velocity 相）
# 覆盖整个 padded 网格；strip 线程先备份，内区线程做 FD 更新
# ──────────────────────────────────────────────────────────────────────────────
function _fused_vel_acoustic_cuda!(
    vx, vz, p, vx_old, vz_old,
    bx, bz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        # 1) strip 备份（条件与 _backup_single_field_cuda! 一致）
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds begin
                vx_old[i, j] = vx[i, j]
                vz_old[i, j] = vz[i, j]
            end
        end

        # 2) 内区 FD 更新（区域与原 update_velocity_acoustic! 一致：[M+1, n-M]）
        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dpdx = 0.0f0
            dpdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dpdx += c * (p[i+l-1, j] - p[i-l, j])
                dpdz += c * (p[i, j+l] - p[i, j-l+1])
            end
            @inbounds begin
                vx[i, j] += bx[i, j] * dtx * dpdx
                vz[i, j] += bz[i, j] * dtz * dpdz
            end
        end
    end
    return nothing
end

function fused_update_velocity_acoustic!(W::AcousticWavefield, M_med::AcousticMedium,
    a_static::SVector{N,Float32}, dt, nbc::Int32) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)

    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8))

    @cuda threads = threads blocks = blocks _fused_vel_acoustic_cuda!(
        W.vx, W.vz, W.p, W.vx_old, W.vz_old,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 融合 kernel：边界带备份 + 全场更新（pressure 相）
# ──────────────────────────────────────────────────────────────────────────────
function _fused_prs_acoustic_cuda!(
    p, vx, vz, p_old,
    kappa,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds p_old[i, j] = p[i, j]
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dvxdx = 0.0f0
            dvzdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dvxdx += c * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += c * (vz[i, j+l-1] - vz[i, j-l])
            end
            @inbounds p[i, j] += kappa[i, j] * (dvxdx * dtx + dvzdz * dtz)
        end
    end
    return nothing
end

function fused_update_pressure_acoustic!(W::AcousticWavefield, M_med::AcousticMedium,
    a_static::SVector{N,Float32}, dt, nbc::Int32) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)

    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8))

    @cuda threads = threads blocks = blocks _fused_prs_acoustic_cuda!(
        W.p, W.vx, W.vz, W.p_old,
        M_med.kappa,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 三场检波：p / vx / vz 一次 launch（原来 3 次）
# ──────────────────────────────────────────────────────────────────────────────
function _record3_kernel!(dp, dvx, dvz, p, vx, vz, rec_i, rec_j, k::Int32, n_rec::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds begin
            ix = rec_i[idx]
            iz = rec_j[idx]
            dp[idx, k] = p[ix, iz]
            dvx[idx, k] = vx[ix, iz]
            dvz[idx, k] = vz[ix, iz]
        end
    end
    return nothing
end

function record_receivers3!(seis_p, seis_vx, seis_vz, W::AcousticWavefield,
    rec::ReceiverConfig, k::Int)
    n_rec = Int32(length(rec.rx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(Int(n_rec), threads)
    @cuda threads = threads blocks = blocks _record3_kernel!(
        seis_p, seis_vx, seis_vz, W.p, W.vx, W.vz, rec.rx, rec.rz, k32, n_rec)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 融合时间步循环（仅 HABC 路径；sponge 走原循环）
# 每步 6 次 launch，HABC/备份不再有空转线程
# ──────────────────────────────────────────────────────────────────────────────
function _acoustic2d_loop_fused!(W, M, S, R, B,
    a_static, dt, nt, pad,
    nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_p, seis_vx, seis_vz, snaps, snap_interval)
    snap_idx = 1
    # 确定性两遍 HABC 的帧暂存（velocity 相用 2 段，pressure 相复用第 1 段）
    scr = CUDA.zeros(Float32, 2 * _habc_frame_total(nx, nz, nbc))

    for it in 1:nt
        # A. Velocity 相：备份+更新（1 launch）→ det-HABC vx+vz（2 launch）
        fused_update_velocity_acoustic!(W, M, a_static, dt, nbc)
        apply_habc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # B. Pressure 相：备份+更新（1 launch）→ det-HABC p（2 launch）
        fused_update_pressure_acoustic!(W, M, a_static, dt, nbc)
        apply_habc_det_1!(W.p, W.p_old, B.w_tau, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # C. 震源
        inject_source!(W.p, S, it, dt)

        # D. 三场检波（1 launch）
        record_receivers3!(seis_p, seis_vx, seis_vz, W, R, it)

        # E. 快照
        if snap_interval > 0 && it % snap_interval == 0
            snaps[snap_idx] = Array(@view W.p[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end
