function ReceiverConfig(rx::Vector, rz::Vector, type::Symbol=:vz)
    n = length(rx)
    # Automatically determine nt if not provided? 
    # Actually, the struct definition includes `data::T`.
    # Usually we initialize `data` with zeros.
    # But we don't know `nt` here unless passed.
    error("Please provide `data` buffer or use `ReceiverConfig(rx, rz, nt, type)`")
end

function ReceiverConfig(rx::Vector, rz::Vector, nt::Int, type::Symbol=:vz)
    n = length(rx)
    return ReceiverConfig(
        to_device(Int32.(rx)),
        to_device(Int32.(rz)),
        to_device(zeros(Float32, n, nt)),
        type
    )
end

function ReceiverConfig(rx::Vector, rz::Vector, data::AbstractArray, type::Symbol=:vz)
    return ReceiverConfig(
        to_device(Int32.(rx)),
        to_device(Int32.(rz)),
        to_device(Float32.(data)),
        type
    )
end