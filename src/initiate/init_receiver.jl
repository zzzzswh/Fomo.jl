# src/initiate/init_receiver.jl


function init_receiver(pad::Int, rx::Vector{Int32}, rz::Vector{Int32}, nt::Int, type::Symbol)
    n = length(rx)
    rx_pad = rx .+ Int32(pad)
    rz_pad = rz .+ Int32(pad)

    # data 缓冲区在 run_simulation! 里已经用 similar(W.vz) 分配在 GPU 上了，这里配置好坐标即可
    return ReceiverConfig(
        to_device(rx_pad),
        to_device(rz_pad),
        to_device(zeros(Float32, n, nt)),
        type
    )
end
