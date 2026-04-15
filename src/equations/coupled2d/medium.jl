# src/equations/coupled2d/medium.jl
#
# 耦合 P-S 势场方程的介质参数
#
# 基于 Li et al. (2018) 方程 8 和 9：
#   P̈ = P·∇²α + 2∇α·∇P − 2P·∇²β − 2∇β·(∇×S) + α·∇²P + ∇·f
#   S̈ = ∇β·∇S − (∇β)×(∇×S) + 2(∇β)×(∇P) + β·∇²S + ∇×f
#
# 在 2D (x-z 平面) 中，S 只有 y 分量 Sy，方程简化为：
#   P̈ = P·lap_α + 2(αx·Px + αz·Pz) − 2P·lap_β
#       + 2(βx·Sz − βz·Sx) + α·∇²P + ∇·f
#   S̈y = 2(βx·Sx + βz·Sz) + 2(βz·Px − βx·Pz) + β·∇²Sy + (∇×f)_y
#
# 其中 αx = ∂α/∂x, Px = ∂P/∂x 等
# 介质参数的梯度和拉普拉斯量是预计算的（模型不随时间变化）

using CUDA

struct CoupledMedium{T}
    nx::Int             # padded 网格 x 方向点数
    nz::Int             # padded 网格 z 方向点数
    dh::Float32         # 网格间距
    x_max::Float32
    z_max::Float32
    M::Int              # 半阶数 = fd_order ÷ 2
    pad::Int            # 总填充 = nbc + M
    # ── 介质参数（GPU 上）──
    alpha::T            # α = Vp²
    beta::T             # β = Vs²
    # ── 预计算的空间导数（GPU 上）──
    dalpha_dx::T        # ∂α/∂x
    dalpha_dz::T        # ∂α/∂z
    dbeta_dx::T         # ∂β/∂x
    dbeta_dz::T         # ∂β/∂z
    lap_alpha::T        # ∇²α
    lap_beta::T         # ∇²β
end

"""
    init_coupled_medium(vp, vs, dh, nbc, fd_order)

初始化耦合方程介质。假设常密度 ρ=1。

预计算 α, β 的空间梯度和拉普拉斯量，避免时间步内重复计算。
"""
function init_coupled_medium(vp::Matrix, vs::Matrix,
    dh::Real, nbc::Int, fd_order::Int)

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
    beta  = Float32.(vs_pad .^ 2)

    # 获取中心差分系数
    d1_coeffs = get_centered_d1(fd_order)
    d2_c0, d2_coeffs = get_centered_d2(fd_order)

    h = Float32(dh)

    # 预计算介质参数的空间导数
    dalpha_dx = _compute_gradient_x(alpha, d1_coeffs, h, M)
    dalpha_dz = _compute_gradient_z(alpha, d1_coeffs, h, M)
    dbeta_dx  = _compute_gradient_x(beta, d1_coeffs, h, M)
    dbeta_dz  = _compute_gradient_z(beta, d1_coeffs, h, M)
    lap_alpha = _compute_laplacian(alpha, d2_c0, d2_coeffs, h, M)
    lap_beta  = _compute_laplacian(beta, d2_c0, d2_coeffs, h, M)

    return CoupledMedium(
        nx, nz, h, x_max, z_max, M, pad,
        to_device(alpha), to_device(beta),
        to_device(dalpha_dx), to_device(dalpha_dz),
        to_device(dbeta_dx), to_device(dbeta_dz),
        to_device(lap_alpha), to_device(lap_beta)
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# CPU 端预计算辅助函数
# ──────────────────────────────────────────────────────────────────────────────

"""
计算 x 方向一阶导数（中心差分）: ∂f/∂x ≈ (1/h) Σ c[l]*(f[i+l,j]-f[i-l,j])
"""
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

"""
计算 z 方向一阶导数（中心差分）: ∂f/∂z ≈ (1/h) Σ c[l]*(f[i,j+l]-f[i,j-l])
"""
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

"""
计算拉普拉斯量（中心差分）:
  ∇²f = ∂²f/∂x² + ∂²f/∂z²
      ≈ (1/h²)[c0·f[i,j] + Σ c[l]·(f[i+l,j]+f[i-l,j])]
      + (1/h²)[c0·f[i,j] + Σ c[l]·(f[i,j+l]+f[i,j-l])]
"""
function _compute_laplacian(f::Matrix{Float32}, d2_c0::Float32,
    d2::SVector{N,Float32}, h::Float32, M::Int) where {N}
    nx, nz = size(f)
    result = zeros(Float32, nx, nz)
    ih2 = 1.0f0 / (h * h)

    @inbounds for j in (M+1):(nz-M), i in (M+1):(nx-M)
        # x 方向二阶导数
        d2x = d2_c0 * f[i, j]
        for l in 1:N
            d2x += d2[l] * (f[i+l, j] + f[i-l, j])
        end
        # z 方向二阶导数
        d2z = d2_c0 * f[i, j]
        for l in 1:N
            d2z += d2[l] * (f[i, j+l] + f[i, j-l])
        end
        result[i, j] = (d2x + d2z) * ih2
    end
    return result
end
