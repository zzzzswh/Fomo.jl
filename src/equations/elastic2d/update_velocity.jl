# src/equations/elastic2d/update_velocity.jl
using CUDA
using StaticArrays

# ==============================================================================
# 手写 CUDA kernel —— 消除 @parallel 宏的堆分配
#
# 设计要点:
#   1. SVector{N} 作为参数按值传入 kernel，编译期展开循环，零分配
#   2. thread/block 配置只算一次，直接传 Int32 常量
#   3. 与原 @parallel_indices 版本完全等价的物理计算
# ==============================================================================

function _update_velocity_cuda!(
    vx, vz, txx, tzz, txz,
    bx, bz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32,
    inner_nx::Int32, inner_nz::Int32
) where {N}
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= inner_nx && iy <= inner_nz
        i = ix + M
        j = iy + M

        dtxxdx = 0.0f0
        dtxzdz = 0.0f0
        dtxzdx = 0.0f0
        dtzzdz = 0.0f0

        # SVector 循环在编译期完全展开
        @inbounds for l in 1:N
            c = a[l]
            dtxxdx += c * (txx[i+l-1, j] - txx[i-l, j])
            dtxzdz += c * (txz[i, j+l-1] - txz[i, j-l])
            dtxzdx += c * (txz[i+l, j] - txz[i-l+1, j])
            dtzzdz += c * (tzz[i, j+l] - tzz[i, j-l+1])
        end

        @inbounds begin
            vx[i, j] += bx[i, j] * (dtx * dtxxdx + dtz * dtxzdz)
            vz[i, j] += bz[i, j] * (dtx * dtxzdx + dtz * dtzzdz)
        end
    end
    return nothing
end

function update_velocity!(W, M_med, a_static::SVector{N,Float32}, dt, inner_nx::Int, inner_nz::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    M32 = Int32(M_med.M)
    inx = Int32(inner_nx)
    inz = Int32(inner_nz)

    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads = threads blocks = blocks _update_velocity_cuda!(
        W.vx, W.vz, W.txx, W.tzz, W.txz,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, inx, inz
    )
    return nothing
end