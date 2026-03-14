"""
    Receivers{T,I}

Receiver configuration and data buffer.
"""
struct ReceiverConfig{T<:AbstractMatrix{Float32},I<:AbstractVector{<:Integer}}
    rx::I                   # X indices (GPU array)
    rz::I                   # Z indices (GPU array)
    data::T                 # [n_rec × nt] (Optional buffer)
    type::Symbol            # :vz, :vx, :p
end
