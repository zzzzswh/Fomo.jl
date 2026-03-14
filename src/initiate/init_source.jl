function init_source_config(sx::Vector, sz::Vector, wavelet::AbstractArray)
    return SourceConfig(
        to_device(Int32.(sx)),
        to_device(Int32.(sz)),
        to_device(Float32.(wavelet))
    )
end