# src/utils/fd_utils.jl (或者直接放在 Fomo_gpu.jl 中)
using StaticArrays

# 使用 Tuple 或 SVector 来存储，避免 Vector 的堆分配
const FD_COEFFICIENTS = Dict{Int,SVector}(
    2 => @SVector(Float32[1.0]),
    4 => @SVector(Float32[1.125, -0.041666667]),
    6 => @SVector(Float32[1.171875, -0.065104167, 0.0046875]),
    8 => @SVector(Float32[1.1962890625, -0.079752604167, 0.0095703125, -0.000697544643]),
    10 => @SVector(Float32[1.2115478515625, -0.089721679687, 0.0138427734375, -0.00176565987723, 0.0001186795166])
)

"""
获取指定阶数的有限差分系数 (返回 SVector 以适配 CUDA 内核)
"""
function get_fd_coefficients(order::Int)
    haskey(FD_COEFFICIENTS, order) || error("Unsupported FD order: $order")
    return FD_COEFFICIENTS[order]
end