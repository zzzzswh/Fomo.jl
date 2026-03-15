

function init_source(pad::Int, sx::Vector{Int32}, sz::Vector{Int32}, wavelet::Matrix{Float32})
    # 加上 pad 偏移
    sx_pad = sx .+ Int32(pad)
    sz_pad = sz .+ Int32(pad)


    return SourceConfig(
        to_device(sx_pad),
        to_device(sz_pad),
        to_device(wavelet)
    )
end