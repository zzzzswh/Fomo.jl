# src/equations/coupled2d/medium.jl
#
# 耦合 P-S 势场方程的介质参数
#
# ⚠ 关键数值处理：
#   方程中 P·∇²α、-2P·∇²β 等项包含介质参数的拉普拉斯量。
#   在尖锐界面处，∇²α ∝ δ'(x)，数值上为 O(Δα/dh²)，极度敏感于网格间距。
#   直接使用会导致低频不稳定。
#
#   解决方案：对 α 和 β 做高斯平滑后再计算空间导数。
#   物理依据：论文的 Born 近似将介质分解为光滑背景 + 粗糙扰动。
#   传播项（α·∇²P, β·∇²S）仍使用原始未平滑的 α, β，保留传播精度。

using CUDA

struct CoupledMedium{T}
    nx::Int
    nz::Int
    dh::Float32
    x_max::Float32
    z_max::Float32
    M::Int
    pad::Int
    # ── 原始介质参数（用于传播项 α∇²P, β∇²S）──
    alpha::T            # α = Vp²（未平滑）
    beta::T             # β = Vs²（未平滑）
    # ── 平滑后介质的空间导数（用于耦合/散射项）──
    dalpha_dx::T
    dalpha_dz::T
    dbeta_dx::T
    dbeta_dz::T
    lap_alpha::T
    lap_beta::T
end

"""
    init_coupled_medium(vp, vs, dh, nbc, fd_order; smooth_sigma=3.0)

初始化耦合方程介质（常密度 ρ=1）。

`smooth_sigma`: 高斯平滑的标准差（网格点数）。
  - 默认 3.0，适用于大多数模型
  - 设为 0 可关闭平滑（仅用于已经光滑的模型）
  - 界面越尖锐、dh 越小，需要越大的 sigma
"""
function init_coupled_medium(vp::Matrix, vs::Matrix,
    dh::Real, nbc::Int, fd_order::Int;
    smooth_sigma::Float64=3.0)

    M = fd_order ÷ 2
    pad = nbc + M

    nx_inner, nz_inner = size(vp)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dh)
    z_max = Float32((nz_inner - 1) * dh)

    # 填充模型
    vp_pad = _pad_array(vp, pad)
    vs_pad = _pad_array(vs, pad)

    # 计算 α = Vp², β = Vs²
    alpha = Float32.(vp_pad .^ 2)
    beta = Float32.(vs_pad .^ 2)

    # 平滑后的 α, β（用于计算空间导数）
    if smooth_sigma > 0
        alpha_s = _gaussian_smooth(alpha, smooth_sigma)
        beta_s = _gaussian_smooth(beta, smooth_sigma)
        @info "Medium smoothed: σ=$(smooth_sigma) grid points"
    else
        alpha_s = copy(alpha)
        beta_s = copy(beta)
    end

    # 获取中心差分系数
    d1_coeffs = get_centered_d1(fd_order)
    d2_c0, d2_coeffs = get_centered_d2(fd_order)

    h = Float32(dh)

    # 预计算 **平滑** 介质的空间导数
    dalpha_dx = _compute_gradient_x(alpha_s, d1_coeffs, h, M)
    dalpha_dz = _compute_gradient_z(alpha_s, d1_coeffs, h, M)
    dbeta_dx = _compute_gradient_x(beta_s, d1_coeffs, h, M)
    dbeta_dz = _compute_gradient_z(beta_s, d1_coeffs, h, M)
    lap_alpha = _compute_laplacian(alpha_s, d2_c0, d2_coeffs, h, M)
    lap_beta = _compute_laplacian(beta_s, d2_c0, d2_coeffs, h, M)

    # 诊断信息
    @info "  ∇²α range: [$(round(minimum(lap_alpha), digits=1)), $(round(maximum(lap_alpha), digits=1))]"
    @info "  ∇²β range: [$(round(minimum(lap_beta), digits=1)), $(round(maximum(lap_beta), digits=1))]"

    return CoupledMedium(
        nx, nz, h, x_max, z_max, M, pad,
        to_device(alpha), to_device(beta),
        to_device(dalpha_dx), to_device(dalpha_dz),
        to_device(dbeta_dx), to_device(dbeta_dz),
        to_device(lap_alpha), to_device(lap_beta)
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# 高斯平滑
# ──────────────────────────────────────────────────────────────────────────────

"""
2D 高斯平滑（CPU 端，仅初始化时调用一次）。
使用可分离卷积：先沿 x 方向，再沿 z 方向。
"""
function _gaussian_smooth(f::Matrix{Float32}, sigma::Float64)
    radius = ceil(Int, 3 * sigma)
    kernel = Float32[exp(-0.5 * (k / sigma)^2) for k in -radius:radius]
    kernel ./= sum(kernel)

    nx, nz = size(f)
    temp = zeros(Float32, nx, nz)
    result = zeros(Float32, nx, nz)

    # 沿 x 方向卷积
    @inbounds for j in 1:nz, i in 1:nx
        val = 0.0f0
        for (ki, k) in enumerate(-radius:radius)
            ii = clamp(i + k, 1, nx)
            val += kernel[ki] * f[ii, j]
        end
        temp[i, j] = val
    end

    # 沿 z 方向卷积
    @inbounds for j in 1:nz, i in 1:nx
        val = 0.0f0
        for (ki, k) in enumerate(-radius:radius)
            jj = clamp(j + k, 1, nz)
            val += kernel[ki] * temp[i, jj]
        end
        result[i, j] = val
    end

    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# CPU 端预计算辅助函数
# ──────────────────────────────────────────────────────────────────────────────

function _compute_gradient_x(f::Matrix{Float32}, d1::SVector{N,Float32},
    h::Float32, M::Int) where {N}
    nx, nz = size(f)
    result = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz, i in (M+1):(nx-M)
        val = 0.0f0
        for l in 1:N
            val += d1[l] * (f[i+l, j] - f[i-l, j])
        end
        result[i, j] = val / h
    end
    return result
end

function _compute_gradient_z(f::Matrix{Float32}, d1::SVector{N,Float32},
    h::Float32, M::Int) where {N}
    nx, nz = size(f)
    result = zeros(Float32, nx, nz)
    @inbounds for j in (M+1):(nz-M), i in 1:nx
        val = 0.0f0
        for l in 1:N
            val += d1[l] * (f[i, j+l] - f[i, j-l])
        end
        result[i, j] = val / h
    end
    return result
end

function _compute_laplacian(f::Matrix{Float32}, d2_c0::Float32,
    d2::SVector{N,Float32}, h::Float32, M::Int) where {N}
    nx, nz = size(f)
    result = zeros(Float32, nx, nz)
    ih2 = 1.0f0 / (h * h)
    @inbounds for j in (M+1):(nz-M), i in (M+1):(nx-M)
        d2x = d2_c0 * f[i, j]
        for l in 1:N
            d2x += d2[l] * (f[i+l, j] + f[i-l, j])
        end
        d2z = d2_c0 * f[i, j]
        for l in 1:N
            d2z += d2[l] * (f[i, j+l] + f[i, j-l])
        end
        result[i, j] = (d2x + d2z) * ih2
    end
    return result
end