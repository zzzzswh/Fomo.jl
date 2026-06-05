# src/boundary/sponge.jl
#
# Cerjan-style 指数衰减吸收边界层
# 适用于二阶时间格式的波动方程（P̈ = ..., S̈ = ...）
#
# 参考: Cerjan et al. (1985), Geophysics
#
# 原理：在边界层内，每个时间步后将波场乘以衰减因子
#   damp(d) = exp(-[factor * (nbc - d)]²)
# 其中 d 是到最近边界的距离（网格点数），factor 通常取 0.015

using CUDA

struct SpongeConfig{T}
    damp::T         # 衰减系数数组 [nx_pad × nz_pad]，GPU 上
end

"""
    init_sponge(nx, nz, pad, nbc; factor=0.015f0)

初始化 Cerjan sponge 吸收边界。

- `nx, nz`: 原始模型尺寸
- `pad`: 总填充量（= nbc + M）
- `nbc`: 吸收边界层数
- `factor`: Cerjan 衰减系数（默认 0.015，标准值）
  - 越大吸收越强，但可能在边界处产生虚假反射
  - 推荐范围: 0.01 ~ 0.025
"""
function init_sponge(nx::Int, nz::Int, pad::Int, nbc::Int;
    factor::Float32=0.015f0)

    nx_pad = nx + 2 * pad
    nz_pad = nz + 2 * pad

    damp = ones(Float32, nx_pad, nz_pad)

    @inbounds for j in 1:nz_pad, i in 1:nx_pad
        # 到四条边界的最短距离（网格点数）
        d = min(i - 1, nx_pad - i, j - 1, nz_pad - j)

        if d < nbc
            r = factor * Float32(nbc - d)
            damp[i, j] = exp(-r * r)
        end
    end

    return SpongeConfig(to_device(damp))
end

# ──────────────────────────────────────────────────────────────────────────────
# CUDA kernel: 逐点乘以衰减系数
# ──────────────────────────────────────────────────────────────────────────────

function _apply_sponge_cuda!(field, damp, nx::Int32, nz::Int32)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= nx && iy <= nz
        @inbounds field[ix, iy] *= damp[ix, iy]
    end
    return nothing
end

"""
    apply_sponge!(field, sponge, nx, nz)

对单个波场施加 sponge 衰减。
"""
function apply_sponge!(field, sponge::SpongeConfig, nx::Integer, nz::Integer)
    nx32, nz32 = Int32(nx), Int32(nz)
    threads = (32, 8)
    blocks = (cld(nx, 32), cld(nz, 8))

    @cuda threads = threads blocks = blocks _apply_sponge_cuda!(
        field, sponge.damp, nx32, nz32)
    return nothing
end
