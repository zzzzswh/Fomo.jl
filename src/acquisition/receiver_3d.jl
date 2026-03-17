# src/acquisition/receiver_3d.jl

using CUDA

struct ReceiverConfig3D{I<:AbstractVector{<:Integer}}
    rx::I               # X indices (GPU array, padded)
    ry::I               # Y indices (GPU array, padded)
    rz::I               # Z indices (GPU array, padded)
    type::Symbol        # :vz, :vx, :vy, :p
end

"""
    init_receiver_3d(pad, rx, ry, rz, type=:vz)

创建3D接收器配置。坐标自动加 pad 偏移并传输到 GPU。
"""
function init_receiver_3d(pad::Int, rx::Vector{Int32}, ry::Vector{Int32},
    rz::Vector{Int32}, type::Symbol=:vz)
    rx_pad = rx .+ Int32(pad)
    ry_pad = ry .+ Int32(pad)
    rz_pad = rz .+ Int32(pad)
    return ReceiverConfig3D(CuArray(rx_pad), CuArray(ry_pad), CuArray(rz_pad), type)
end
