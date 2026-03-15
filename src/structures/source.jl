abstract type AbstractSource end

"""
    Source{V<:AbstractVector}

Single source configuration.
"""
struct Source{V<:AbstractVector{Float32},I<:Integer} <: AbstractSource
    i::I                # X grid index
    j::I                # Z grid index
    wavelet::V          # Source time function
end

struct SourceConfig{T<:AbstractMatrix{Float32},I<:AbstractVector{<:Integer}}
    sx::I               # X grid indices (GPU array)
    sz::I               # Z grid indices (GPU array)
    wavelet::T          # Source time functions [n_src × nt] (GPU array)
end