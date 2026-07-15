# src/equations/elastic2d/fused_kernels.jl
#
# 弹性波融合内核。
#
# 旧 HABC 路径每步 13 次 launch：
#   backup_v + update_v + habc(vx) + habc(vz)
#   + backup_s + update_s + habc(txx) + habc(tzz) + habc(txz)
#   + inject(txx) + inject(tzz) + record(vx) + record(vz)
#
# 新路径 6 次：
#   fused_v(备份+更新) + habc_frame_2(vx,vz)
#   + fused_s(备份+更新) + habc_frame_3(txx,tzz,txz)
#   + inject_pair(txx,tzz) + record2(vx,vz)
#
using CUDA
using StaticArrays

# ──────────────────────────────────────────────────────────────────────────────
# 融合：velocity 相（strip 备份 vx,vz + 内区 FD 更新）
# ──────────────────────────────────────────────────────────────────────────────
function _fused_vel_elastic_cuda!(
    vx, vz, txx, tzz, txz, vx_old, vz_old,
    bx, bz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds begin
                vx_old[i, j] = vx[i, j]
                vz_old[i, j] = vz[i, j]
            end
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dtxxdx = 0.0f0
            dtxzdz = 0.0f0
            dtxzdx = 0.0f0
            dtzzdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dtxxdx += c * (txx[i+l-1, j] - txx[i-l, j])
                dtxzdz += c * (txz[i, j+l-1] - txz[i, j-l])
                dtxzdx += c * (txz[i+l, j] - txz[i-l+1, j])
                dtzzdz += c * (tzz[i, j+l] - tzz[i, j-l+1])
            end
            @inbounds begin
                vx[i, j] += bx[i, j] * (dtx * dtxxdx + dtz * dtxzdz)
                vz[i, j] += bz[i, j] * (dtx * dtxzdx + dtz * dtzzdz)
            end
        end
    end
    return nothing
end

function fused_update_velocity_elastic!(W, M_med, a_static::SVector{N,Float32},
    dt, nbc::Int32) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)

    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8))

    @cuda threads = threads blocks = blocks _fused_vel_elastic_cuda!(
        W.vx, W.vz, W.txx, W.tzz, W.txz, W.vx_old, W.vz_old,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 融合：stress 相（strip 备份 txx,tzz,txz + 内区 FD 更新）
# ──────────────────────────────────────────────────────────────────────────────
function _fused_str_elastic_cuda!(
    txx, tzz, txz, vx, vz, txx_old, tzz_old, txz_old,
    lam, lam_2mu, mu_txz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds begin
                txx_old[i, j] = txx[i, j]
                tzz_old[i, j] = tzz[i, j]
                txz_old[i, j] = txz[i, j]
            end
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dvxdx = 0.0f0
            dvzdz = 0.0f0
            dvxdz = 0.0f0
            dvzdx = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dvxdx += c * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += c * (vz[i, j+l-1] - vz[i, j-l])
                dvxdz += c * (vx[i, j+l] - vx[i, j-l+1])
                dvzdx += c * (vz[i+l-1, j] - vz[i-l, j])
            end
            @inbounds begin
                l_val = lam[i, j]
                l2m_val = lam_2mu[i, j]
                txx[i, j] += l2m_val * dvxdx * dtx + l_val * dvzdz * dtz
                tzz[i, j] += l_val * dvxdx * dtx + l2m_val * dvzdz * dtz
                txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
            end
        end
    end
    return nothing
end

function fused_update_stress_elastic!(W, M_med, a_static::SVector{N,Float32},
    dt, nbc::Int32) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)

    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8))

    @cuda threads = threads blocks = blocks _fused_str_elastic_cuda!(
        W.txx, W.tzz, W.txz, W.vx, W.vz, W.txx_old, W.tzz_old, W.txz_old,
        M_med.lam, M_med.lam_2mu, M_med.mu_txz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 双场注入：txx 与 tzz 同一 launch（原来 2 次）
# ──────────────────────────────────────────────────────────────────────────────
function _inject_pair_kernel!(f1, f2, wavelet, sx, sz, k::Int32, n_src::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_src
        @inbounds begin
            ix = sx[idx]
            iz = sz[idx]
            wav = wavelet[idx, k]
            f1[ix, iz] += wav
            f2[ix, iz] += wav
        end
    end
    return nothing
end

function inject_source_pair!(f1, f2, S::SourceConfig, k::Int)
    n_src = Int32(length(S.sx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(Int(n_src), threads)
    @cuda threads = threads blocks = blocks _inject_pair_kernel!(
        f1, f2, S.wavelet, S.sx, S.sz, k32, n_src)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 双场检波：vx / vz 一次 launch（原来 2 次）
# ──────────────────────────────────────────────────────────────────────────────
function _record2_kernel!(dvx, dvz, vx, vz, rec_i, rec_j, k::Int32, n_rec::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds begin
            ix = rec_i[idx]
            iz = rec_j[idx]
            dvx[idx, k] = vx[ix, iz]
            dvz[idx, k] = vz[ix, iz]
        end
    end
    return nothing
end

function record_receivers2!(seis_vx, seis_vz, W, rec::ReceiverConfig, k::Int)
    n_rec = Int32(length(rec.rx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(Int(n_rec), threads)
    @cuda threads = threads blocks = blocks _record2_kernel!(
        seis_vx, seis_vz, W.vx, W.vz, rec.rx, rec.rz, k32, n_rec)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 融合时间步循环（仅 HABC 路径；sponge 走原循环）
# ──────────────────────────────────────────────────────────────────────────────
function _elastic2d_loop_fused!(W, M, S, R, B,
    a_static, dt, nt, pad,
    seis_vx, seis_vz, snaps, snap_interval)
    snap_idx = 1

    nx = Int32(M.nx)
    nz = Int32(M.nz)
    nbc = Int32(B.nbc)
    qx = Float32(B.qx)
    qz = Float32(B.qz)
    qt_x = Float32(B.qt_x)
    qt_z = Float32(B.qt_z)
    qxt = Float32(B.qxt)

    # 确定性两遍 HABC 的帧暂存（stress 相 3 场为最大需求）
    scr = CUDA.zeros(Float32, 3 * _habc_frame_total(nx, nz, nbc))

    for it in 1:nt
        # A. Velocity 相
        fused_update_velocity_elastic!(W, M, a_static, dt, nbc)
        apply_habc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # B. Stress 相
        fused_update_stress_elastic!(W, M, a_static, dt, nbc)
        apply_habc_det_3!(W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
            B.w_tau, scr, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)

        # C. 震源（txx + tzz 同一 launch）
        inject_source_pair!(W.txx, W.tzz, S, it)

        # D. 双场检波
        record_receivers2!(seis_vx, seis_vz, W, R, it)

        # E. 快照
        if snap_interval > 0 && it % snap_interval == 0
            snaps[snap_idx] = Array(@view W.vz[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end
