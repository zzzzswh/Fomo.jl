# src/acquisition/source_3d.jl

using CUDA

struct SourceConfig3D{T<:AbstractMatrix{Float32},I<:AbstractVector{<:Integer}}
    sx::I               # X grid indices (GPU array, padded)
    sy::I               # Y grid indices (GPU array, padded)
    sz::I               # Z grid indices (GPU array, padded)
    wavelet::T          # Source time functions [n_src × nt] (GPU array)
end

"""
    init_source_3d(pad, sx, sy, sz, wavelet_matrix)

创建3D震源配置。坐标自动加 pad 偏移并传输到 GPU。
"""
function init_source_3d(pad::Int, sx::Vector{Int32}, sy::Vector{Int32},
    sz::Vector{Int32}, wavelet_matrix::AbstractMatrix{Float32})
    sx_pad = sx .+ Int32(pad)
    sy_pad = sy .+ Int32(pad)
    sz_pad = sz .+ Int32(pad)
    return SourceConfig3D(
        CuArray(sx_pad),
        CuArray(sy_pad),
        CuArray(sz_pad),
        CuArray(wavelet_matrix)
    )
end
