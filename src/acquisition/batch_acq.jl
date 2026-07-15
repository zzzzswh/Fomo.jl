# src/acquisition/batch_acq.jl
#
# 多炮批处理的采集组件。
#
# 布局约定:
#   - 波场:      (nx, nz, n_shots) 三维 CuArray,炮为最慢维
#   - 震源索引:  (n_src_per_shot, n_shots) Int32 矩阵(已加 pad)
#   - 子波:      (n_src_per_shot, nt),各炮共享同一子波
#   - 地震记录:  (n_rec, nt, n_shots),接收器排列各炮共享(固定排列)
#
# 各炮之间没有任何数据交互:批处理只是把同一套算术在第三维摆 n_shots 层,
# 因此【每一炮的结果必须与单炮运行逐位相等】—— 这是 verify_batch.jl 的判据。

using CUDA

struct BatchedSourceConfig{T<:AbstractMatrix{Float32},M<:AbstractMatrix{Int32}}
    sx::M               # (n_src_per_shot, n_shots),padded
    sz::M
    wavelet::T          # (n_src_per_shot, nt),各炮共享
end

"""
    init_batched_source(pad, sx, sz, wavelet_matrix)

多炮震源。`sx`/`sz` 为 (n_src_per_shot, n_shots) 原始网格坐标,自动加 pad 上 GPU。
"""
function init_batched_source(pad::Int, sx::AbstractMatrix{<:Integer},
    sz::AbstractMatrix{<:Integer}, wavelet_matrix::AbstractMatrix{Float32})
    size(sx) == size(sz) ||
        throw(ArgumentError("sx 与 sz 尺寸不一致: $(size(sx)) vs $(size(sz))"))
    size(wavelet_matrix, 1) == size(sx, 1) ||
        throw(ArgumentError("wavelet 行数 $(size(wavelet_matrix, 1)) ≠ 每炮震源数 $(size(sx, 1))"))
    return BatchedSourceConfig(
        CuArray(Int32.(sx) .+ Int32(pad)),
        CuArray(Int32.(sz) .+ Int32(pad)),
        CuArray(wavelet_matrix))
end

# ── 批量注入:单场 / 双场 ──────────────────────────────────────────────────
function _binject_kernel!(field, wavelet, sx, sz, k::Int32, n_src::Int32)
    q = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if q <= n_src
        @inbounds field[sx[q, s], sz[q, s], s] += wavelet[q, k]
    end
    return nothing
end

"""批量注入(与 inject_source! 同数学:裸加)。"""
function binject_source!(field, S::BatchedSourceConfig, k::Int)
    n_src, ns = size(S.sx)
    threads = 256
    @cuda threads = threads blocks = (cld(n_src, threads), ns) _binject_kernel!(
        field, S.wavelet, S.sx, S.sz, Int32(k), Int32(n_src))
    return nothing
end

function _binject_pair_kernel!(f1, f2, wavelet, sx, sz, k::Int32, n_src::Int32)
    q = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if q <= n_src
        @inbounds begin
            ix = sx[q, s]
            iz = sz[q, s]
            wav = wavelet[q, k]
            f1[ix, iz, s] += wav
            f2[ix, iz, s] += wav
        end
    end
    return nothing
end

"""批量双场注入(txx + tzz)。"""
function binject_source_pair!(f1, f2, S::BatchedSourceConfig, k::Int)
    n_src, ns = size(S.sx)
    threads = 256
    @cuda threads = threads blocks = (cld(n_src, threads), ns) _binject_pair_kernel!(
        f1, f2, S.wavelet, S.sx, S.sz, Int32(k), Int32(n_src))
    return nothing
end

# ── 批量记录:三场 / 双场 ──────────────────────────────────────────────────
function _brecord3_kernel!(dp, dvx, dvz, p, vx, vz, rec_i, rec_j, k::Int32, n_rec::Int32)
    r = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if r <= n_rec
        @inbounds begin
            i = rec_i[r]
            j = rec_j[r]
            dp[r, k, s] = p[i, j, s]
            dvx[r, k, s] = vx[i, j, s]
            dvz[r, k, s] = vz[i, j, s]
        end
    end
    return nothing
end

"""批量三场记录:seis 形状 (n_rec, nt, n_shots)。"""
function brecord_receivers3!(seis_p, seis_vx, seis_vz, p, vx, vz,
    R::ReceiverConfig, k::Int, ns::Int)
    n_rec = Int32(length(R.rx))
    threads = 256
    @cuda threads = threads blocks = (cld(Int(n_rec), threads), ns) _brecord3_kernel!(
        seis_p, seis_vx, seis_vz, p, vx, vz, R.rx, R.rz, Int32(k), n_rec)
    return nothing
end

function _brecord2_kernel!(dvx, dvz, vx, vz, rec_i, rec_j, k::Int32, n_rec::Int32)
    r = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if r <= n_rec
        @inbounds begin
            i = rec_i[r]
            j = rec_j[r]
            dvx[r, k, s] = vx[i, j, s]
            dvz[r, k, s] = vz[i, j, s]
        end
    end
    return nothing
end

"""批量双场记录(vx / vz)。"""
function brecord_receivers2!(seis_vx, seis_vz, vx, vz,
    R::ReceiverConfig, k::Int, ns::Int)
    n_rec = Int32(length(R.rx))
    threads = 256
    @cuda threads = threads blocks = (cld(Int(n_rec), threads), ns) _brecord2_kernel!(
        seis_vx, seis_vz, vx, vz, R.rx, R.rz, Int32(k), n_rec)
    return nothing
end
