# src/acqusition/receiver.jl
using CUDA

struct ReceiverConfig{I<:AbstractVector{<:Integer}}
    rx::I               # X indices (GPU array, padded)
    rz::I               # Z indices (GPU array, padded)
    type::Symbol        # :vz, :vx, :p
end

"""
    init_receiver(pad, rx, rz, type=:vz)

创建接收器配置。坐标自动加 pad 偏移并传输到 GPU。
"""
function init_receiver(pad::Int, rx::Vector{Int32}, rz::Vector{Int32},
    type::Symbol=:vz)
    rx_pad = rx .+ Int32(pad)
    rz_pad = rz .+ Int32(pad)
    return ReceiverConfig(CuArray(rx_pad), CuArray(rz_pad), type)
end