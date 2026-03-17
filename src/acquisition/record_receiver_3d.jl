# src/acquisition/record_receiver_3d.jl

using CUDA

function _record_3d_kernel!(data, field, rec_i, rec_j, rec_k, t::Int32, n_rec::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_rec
        @inbounds begin
            ix = rec_i[idx]
            iy = rec_j[idx]
            iz = rec_k[idx]
            data[idx, t] = field[ix, iy, iz]
        end
    end
    return nothing
end

function record_receivers_3d!(seis_data, field, rec::ReceiverConfig3D, k::Int)
    n_rec = Int32(length(rec.rx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(n_rec, Int32(threads))

    @cuda threads=threads blocks=blocks _record_3d_kernel!(
        seis_data, field, rec.rx, rec.ry, rec.rz, k32, n_rec)
    return nothing
end
