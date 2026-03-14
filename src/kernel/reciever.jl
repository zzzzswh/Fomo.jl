# ==============================================================================
# Receiver Recording (CUDA Optimized)
# ==============================================================================

function _record_kernel!(data, field, rec_i, rec_j, k::Int32, n_rec::Int32)
    # 修复：将 1i32 改为 Int32(1)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_rec
        @inbounds begin
            # 修复坐标顺序：rec_i 实际上是x坐标，rec_j 实际上是z坐标
            # 但在field数组中，z坐标应该作为第一个索引，x坐标作为第二个索引
            jj = rec_i[idx]  # x坐标作为第二个索引（列）
            ii = rec_j[idx]  # z坐标作为第一个索引（行）
            val = field[ii, jj]

            # 【性能警告】: data[k, idx] 在列主序(Julia)下是非合并内存访问，会掉速。
            # 如果可能，强烈建议将 data 在 GPU 上初始化为 [n_rec, nt] 大小，
            # 然后使用 data[idx, k] = val，这会带来巨大的显存带宽提升。
            data[k, idx] = val
        end
    end
    return nothing
end

function _record_pressure_kernel!(data, txx, tzz, rec_i, rec_j, k::Int32, n_rec::Int32)
    # 修复：将 1i32 改为 Int32(1)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_rec
        @inbounds begin
            # 修复坐标顺序：rec_i 实际上是x坐标，rec_j 实际上是z坐标
            # 但在field数组中，z坐标应该作为第一个索引，x坐标作为第二个索引
            jj = rec_i[idx]  # x坐标作为第二个索引（列）
            ii = rec_j[idx]  # z坐标作为第一个索引（行）
            # 乘以 0.5f0 在 GPU 上通常会被编译为 FMA（乘加）指令或快速标量乘法
            val = (txx[ii, jj] + tzz[ii, jj]) * 0.5f0
            data[k, idx] = val
        end
    end
    return nothing
end

# 这个是实际被 el2d_sim.jl 调用的函数 (4个参数)
function record_receivers!(seis_data, field, rec::ReceiverConfig, k::Int)
    # Note: `rec.data` is internal buffer, but simulator passes external buffer `seis_data`.
    # We should use `seis_data`.

    n_rec = Int32(length(rec.rx))
    k32 = Int32(k)
    threads = 256
    blocks = cld(n_rec, Int32(threads))

    @cuda threads = threads blocks = blocks _record_kernel!(seis_data, field, rec.rx, rec.rz, k32, n_rec)
    return nothing
end