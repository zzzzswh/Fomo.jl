# src/acquisition/inject_source_3d.jl

using CUDA

function _inject_field_3d_kernel!(field, wavelet, sx, sy, sz, k::Int32, n_src::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_src
        @inbounds begin
            ix = sx[idx]
            iy = sy[idx]
            iz = sz[idx]
            wav = wavelet[idx, k]
            field[ix, iy, iz] += wav
        end
    end
    return nothing
end

function inject_source_3d!(field, S::SourceConfig3D, k::Int, dt::Float32)
    n_src = Int32(length(S.sx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(n_src, threads)

    @cuda threads=threads blocks=blocks _inject_field_3d_kernel!(
        field, S.wavelet, S.sx, S.sy, S.sz, k32, n_src)
    return nothing
end
