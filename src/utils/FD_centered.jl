# src/utils/FD_centered.jl
#
# 正则网格（非交错）中心差分系数
# 用于二阶势场方程 P̈ = ... 和 S̈ = ...
#
# 与 FD_utils.jl 中的交错网格系数不同：
#   - 交错网格系数用于 f[i+1/2] 处的导数（velocity-stress 格式）
#   - 这里的系数用于 f[i] 处的中心差分（second-order 格式）

using StaticArrays

# ──────────────────────────────────────────────────────────────────────────────
# 一阶导数（反对称）: f'(x) ≈ (1/h) Σ_{l=1}^{M} d1[l] * (f[i+l] - f[i-l])
# ──────────────────────────────────────────────────────────────────────────────
const CENTERED_D1 = Dict{Int,SVector}(
    2  => @SVector(Float32[0.5]),
    4  => @SVector(Float32[2/3, -1/12]),
    6  => @SVector(Float32[3/4, -3/20, 1/60]),
    8  => @SVector(Float32[4/5, -1/5, 4/105, -1/280]),
    10 => @SVector(Float32[5/6, -5/21, 5/84, -5/504, 1/1260]),
)

# ──────────────────────────────────────────────────────────────────────────────
# 二阶导数（对称）: f''(x) ≈ (1/h²) [d2_center * f[i]
#                              + Σ_{l=1}^{M} d2[l] * (f[i+l] + f[i-l])]
# ──────────────────────────────────────────────────────────────────────────────
const CENTERED_D2_CENTER = Dict{Int,Float32}(
    2  => Float32(-2),
    4  => Float32(-5/2),
    6  => Float32(-49/18),
    8  => Float32(-205/72),
    10 => Float32(-5269/1800),
)

const CENTERED_D2 = Dict{Int,SVector}(
    2  => @SVector(Float32[1.0]),
    4  => @SVector(Float32[4/3, -1/12]),
    6  => @SVector(Float32[3/2, -3/20, 1/90]),
    8  => @SVector(Float32[8/5, -1/5, 8/315, -1/560]),
    10 => @SVector(Float32[5/3, -5/21, 5/126, -5/1008, 1/3150]),
)

"""
    get_centered_d1(order) -> SVector

获取中心一阶导数系数（返回 SVector，适配 CUDA kernel）
"""
function get_centered_d1(order::Int)
    haskey(CENTERED_D1, order) || error("Unsupported FD order: $order (centered d1)")
    return CENTERED_D1[order]
end

"""
    get_centered_d2(order) -> (Float32, SVector)

获取中心二阶导数系数。返回 (c0, c_offsets):
  f''(x) ≈ (1/h²) [c0 * f[i] + Σ c_offsets[l] * (f[i+l] + f[i-l])]
"""
function get_centered_d2(order::Int)
    haskey(CENTERED_D2, order) || error("Unsupported FD order: $order (centered d2)")
    return CENTERED_D2_CENTER[order], CENTERED_D2[order]
end
