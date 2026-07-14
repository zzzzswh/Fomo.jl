# src/acquisition/source.jl

using CUDA

struct SourceConfig{T<:AbstractMatrix{Float32},I<:AbstractVector{<:Integer}}
    sx::I               # X grid indices (GPU array, padded)
    sz::I               # Z grid indices (GPU array, padded)
    wavelet::T          # Source time functions [n_src × nt] (GPU array)
end

"""
    init_source(pad, sx, sz, wavelet_matrix)

创建震源配置。坐标自动加 pad 偏移并传输到 GPU。
- `sx`, `sz`: Int32 向量，原始网格坐标
- `wavelet_matrix`: [n_src × nt] 震源子波矩阵
"""
function init_source(pad::Int, sx::Vector{Int32}, sz::Vector{Int32},
    wavelet_matrix::AbstractMatrix{Float32})
    size(wavelet_matrix, 1) == length(sx) ||
        throw(ArgumentError("wavelet 行数 $(size(wavelet_matrix, 1)) ≠ 震源个数 $(length(sx))"))
    sx_pad = sx .+ Int32(pad)
    sz_pad = sz .+ Int32(pad)
    return SourceConfig(
        CuArray(sx_pad),
        CuArray(sz_pad),
        CuArray(wavelet_matrix)
    )
end