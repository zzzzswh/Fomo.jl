# ==============================================================================
# Receiver Recording (CUDA Optimized)
# ==============================================================================

function _record_kernel!(data, field, rec_i, rec_j, k::Int32, n_rec::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_rec
        @inbounds begin
            # field 是 (nx, nz)，第一维是 x，第二维是 z
            ix = rec_i[idx]
            iz = rec_j[idx]
            val = field[ix, iz]

            # 【极致优化】：连续的线程(idx)访问连续的内存地址，完美合并访存
            data[idx, k] = val
        end
    end
    return nothing
end

function _record_pressure_kernel!(data, txx, tzz, rec_i, rec_j, k::Int32, n_rec::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_rec
        @inbounds begin
            # field 是 (nx, nz)，第一维是 x，第二维是 z
            ix = rec_i[idx]
            iz = rec_j[idx]

            # 乘以 0.5f0 走 GPU FMA 指令
            val = (txx[ix, iz] + tzz[ix, iz]) * 0.5f0

            # 【极致优化】：连续的线程(idx)访问连续的内存地址，完美合并访存
            data[idx, k] = val
        end
    end
    return nothing
end

# 实际被 el2d_sim.jl 调用的函数保持不变
function record_receivers!(seis_data, field, rec::ReceiverConfig, k::Int)
    n_rec = Int32(length(rec.rx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(n_rec, Int32(threads))

    @cuda threads = threads blocks = blocks _record_kernel!(seis_data, field, rec.rx, rec.rz, k32, n_rec)
    return nothing
end