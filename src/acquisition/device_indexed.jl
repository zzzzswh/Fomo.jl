# src/acquisition/device_indexed.jl
#
# 设备侧时间索引的注入/记录变体 —— 为 CUDA Graphs 服务。
#
# 背景:graph 一旦录制,kernel 的标量参数(如时间步 k)就被固化;
#       要让同一张 graph 回放 nt 次而每次处理不同时间步,
#       k 必须放在【设备内存】里,由 graph 内的第一个 kernel 自增,
#       后续 kernel 从内存读出。
#
# 数学与原 inject_source! / record_receivers*! 完全相同(裸加 / 直取),
# 仅时间索引来源不同;k 越界时(理论上不会发生)静默跳过作保险。

using CUDA

# ── 步计数器 ───────────────────────────────────────────────────────────────
function _bump_kernel!(c)
    @inbounds c[1] += Int32(1)
    return nothing
end

"""步计数器自增(graph 中每次回放的第一个节点)。"""
function bump_counter!(c)
    @cuda threads = 1 blocks = 1 _bump_kernel!(c)
    return nothing
end

# ── 注入:单场 / 双场 ─────────────────────────────────────────────────────
function _inject_dev_kernel!(field, wavelet, sx, sz, kbuf, n_src::Int32, nt::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_src
        @inbounds begin
            k = kbuf[1]
            if (k >= Int32(1)) & (k <= nt)
                field[sx[idx], sz[idx]] += wavelet[idx, k]
            end
        end
    end
    return nothing
end

"""设备索引版 inject_source!(与原版数学相同:裸加,不乘 dt)。"""
function inject_source_dev!(field, S::SourceConfig, kbuf, nt::Int32)
    n_src = Int32(length(S.sx))
    threads = 256
    blocks = cld(Int(n_src), threads)
    @cuda threads = threads blocks = blocks _inject_dev_kernel!(
        field, S.wavelet, S.sx, S.sz, kbuf, n_src, nt)
    return nothing
end

function _inject_pair_dev_kernel!(f1, f2, wavelet, sx, sz, kbuf, n_src::Int32, nt::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_src
        @inbounds begin
            k = kbuf[1]
            if (k >= Int32(1)) & (k <= nt)
                ix = sx[idx]
                iz = sz[idx]
                wav = wavelet[idx, k]
                f1[ix, iz] += wav
                f2[ix, iz] += wav
            end
        end
    end
    return nothing
end

"""设备索引版 inject_source_pair!(txx + tzz 一次 launch)。"""
function inject_source_pair_dev!(f1, f2, S::SourceConfig, kbuf, nt::Int32)
    n_src = Int32(length(S.sx))
    threads = 256
    blocks = cld(Int(n_src), threads)
    @cuda threads = threads blocks = blocks _inject_pair_dev_kernel!(
        f1, f2, S.wavelet, S.sx, S.sz, kbuf, n_src, nt)
    return nothing
end

# ── 记录:三场 / 双场 ─────────────────────────────────────────────────────
function _record3_dev_kernel!(dp, dvx, dvz, p, vx, vz, rec_i, rec_j, kbuf,
    n_rec::Int32, nt::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds begin
            k = kbuf[1]
            if (k >= Int32(1)) & (k <= nt)
                i = rec_i[idx]
                j = rec_j[idx]
                dp[idx, k] = p[i, j]
                dvx[idx, k] = vx[i, j]
                dvz[idx, k] = vz[i, j]
            end
        end
    end
    return nothing
end

"""设备索引版三场记录(p / vx / vz)。"""
function record_receivers3_dev!(seis_p, seis_vx, seis_vz, p, vx, vz,
    R::ReceiverConfig, kbuf, nt::Int32)
    n_rec = Int32(length(R.rx))
    threads = 256
    blocks = cld(Int(n_rec), threads)
    @cuda threads = threads blocks = blocks _record3_dev_kernel!(
        seis_p, seis_vx, seis_vz, p, vx, vz, R.rx, R.rz, kbuf, n_rec, nt)
    return nothing
end

function _record2_dev_kernel!(dvx, dvz, vx, vz, rec_i, rec_j, kbuf,
    n_rec::Int32, nt::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n_rec
        @inbounds begin
            k = kbuf[1]
            if (k >= Int32(1)) & (k <= nt)
                i = rec_i[idx]
                j = rec_j[idx]
                dvx[idx, k] = vx[i, j]
                dvz[idx, k] = vz[i, j]
            end
        end
    end
    return nothing
end

"""设备索引版双场记录(vx / vz)。"""
function record_receivers2_dev!(seis_vx, seis_vz, vx, vz,
    R::ReceiverConfig, kbuf, nt::Int32)
    n_rec = Int32(length(R.rx))
    threads = 256
    blocks = cld(Int(n_rec), threads)
    @cuda threads = threads blocks = blocks _record2_dev_kernel!(
        seis_vx, seis_vz, vx, vz, R.rx, R.rz, kbuf, n_rec, nt)
    return nothing
end
