# src/equations/coupled2d/update_fields.jl
#
# 耦合 P-S 势场：速度-位置分裂更新 kernel
#
# 将二阶 leapfrog 拆为两个一阶半步，与 velocity-stress 同构：
#
#   半步 1（"速度"更新）：
#     dPdt += dt · RHS_P(P, S, medium)     ← 计算量集中在这里
#     dSdt += dt · RHS_S(P, S, medium)
#
#   半步 2（"位置"更新）：
#     P += dt · dPdt                        ← 简单逐点加法
#     S += dt · dSdt
#
# 每个半步后各施加一次 HABC → 等效二阶 Higdon ABC

using CUDA
using StaticArrays

# ==============================================================================
# Kernel 1: "速度"更新 — 计算 RHS 并更新 dPdt, dSdt
#
# 这是计算密集的 kernel（包含所有空间差分）
# 对应 velocity-stress 格式中的 update_velocity!
# ==============================================================================

function _update_coupled_velocity_cuda!(
    dPdt, dSdt, P, S,                               # 波场
    alpha, beta,                                      # 介质 α=Vp², β=Vs²
    dalpha_dx, dalpha_dz,                            # ∇α (预计算)
    dbeta_dx, dbeta_dz,                              # ∇β (预计算)
    lap_alpha, lap_beta,                              # ∇²α, ∇²β (预计算)
    d1::SVector{N,Float32},                           # 一阶导数系数
    d2::SVector{N,Float32},                           # 二阶导数系数（off-center）
    d2_c0::Float32,                                   # 二阶导数系数（center）
    dt::Float32, ih::Float32, ih2::Float32,           # dt, 1/h, 1/h²
    M::Int32, inner_nx::Int32, inner_nz::Int32
) where {N}

    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= inner_nx && iy <= inner_nz
        i = ix + M
        j = iy + M

        # ── P 的空间导数 ──
        dPdx = 0.0f0
        dPdz = 0.0f0
        lap_P_x = d2_c0 * P[i, j]
        lap_P_z = d2_c0 * P[i, j]

        # ── S 的空间导数 ──
        dSdx = 0.0f0
        dSdz = 0.0f0
        lap_S_x = d2_c0 * S[i, j]
        lap_S_z = d2_c0 * S[i, j]

        @inbounds for l in 1:N
            c1 = d1[l]
            c2 = d2[l]

            dPdx   += c1 * (P[i+l, j] - P[i-l, j])
            dPdz   += c1 * (P[i, j+l] - P[i, j-l])
            lap_P_x += c2 * (P[i+l, j] + P[i-l, j])
            lap_P_z += c2 * (P[i, j+l] + P[i, j-l])

            dSdx   += c1 * (S[i+l, j] - S[i-l, j])
            dSdz   += c1 * (S[i, j+l] - S[i, j-l])
            lap_S_x += c2 * (S[i+l, j] + S[i-l, j])
            lap_S_z += c2 * (S[i, j+l] + S[i, j-l])
        end

        dPdx *= ih
        dPdz *= ih
        dSdx *= ih
        dSdz *= ih
        lap_P = (lap_P_x + lap_P_z) * ih2
        lap_S = (lap_S_x + lap_S_z) * ih2

        @inbounds begin
            P_val = P[i, j]

            ax = dalpha_dx[i, j]
            az = dalpha_dz[i, j]
            bx = dbeta_dx[i, j]
            bz = dbeta_dz[i, j]
            la = lap_alpha[i, j]
            lb = lap_beta[i, j]
            a  = alpha[i, j]
            b  = beta[i, j]

            # ── P 方程 RHS ──
            rhs_P = P_val * la +
                    2.0f0 * (ax * dPdx + az * dPdz) -
                    2.0f0 * P_val * lb +
                    2.0f0 * (bx * dSdz - bz * dSdx) +
                    a * lap_P

            # ── S 方程 RHS ──
            rhs_S = 2.0f0 * (bx * dSdx + bz * dSdz) +
                    2.0f0 * (bz * dPdx - bx * dPdz) +
                    b * lap_S

            # ── "速度"更新：dPdt += dt * RHS ──
            dPdt[i, j] += dt * rhs_P
            dSdt[i, j] += dt * rhs_S
        end
    end
    return nothing
end

# ==============================================================================
# Kernel 2: "位置"更新 — 简单逐点推进
#
# P += dt · dPdt,  S += dt · dSdt
# 对应 velocity-stress 格式中的 update_stress!
# ==============================================================================

function _update_coupled_position_cuda!(
    P, S, dPdt, dSdt,
    dt::Float32,
    M::Int32, inner_nx::Int32, inner_nz::Int32
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if ix <= inner_nx && iy <= inner_nz
        i = ix + M
        j = iy + M

        @inbounds begin
            P[i, j] += dt * dPdt[i, j]
            S[i, j] += dt * dSdt[i, j]
        end
    end
    return nothing
end

# ==============================================================================
# Host 端接口
# ==============================================================================

"""
    update_coupled_velocity!(W, M_med, d1, d2, d2_c0, dt, ...)

"速度"半步：计算 RHS 并更新 dPdt, dSdt。
对应 elastic2d 中的 update_velocity!
"""
function update_coupled_velocity!(
    W::CoupledWavefield, M_med::CoupledMedium,
    d1::SVector{N,Float32}, d2::SVector{N,Float32}, d2_c0::Float32,
    dt::Float32, inner_nx::Int, inner_nz::Int
) where {N}
    ih  = 1.0f0 / M_med.dh
    ih2 = ih * ih
    M32 = Int32(M_med.M)

    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads=threads blocks=blocks _update_coupled_velocity_cuda!(
        W.dPdt, W.dSdt, W.P, W.S,
        M_med.alpha, M_med.beta,
        M_med.dalpha_dx, M_med.dalpha_dz,
        M_med.dbeta_dx, M_med.dbeta_dz,
        M_med.lap_alpha, M_med.lap_beta,
        d1, d2, d2_c0,
        dt, ih, ih2,
        M32, Int32(inner_nx), Int32(inner_nz)
    )
    return nothing
end

"""
    update_coupled_position!(W, dt, M, inner_nx, inner_nz)

"位置"半步：P += dt·dPdt, S += dt·dSdt。
对应 elastic2d 中的 update_stress!
"""
function update_coupled_position!(
    W::CoupledWavefield,
    dt::Float32, M::Int, inner_nx::Int, inner_nz::Int
)
    threads = (32, 8)
    blocks = (cld(inner_nx, 32), cld(inner_nz, 8))

    @cuda threads=threads blocks=blocks _update_coupled_position_cuda!(
        W.P, W.S, W.dPdt, W.dSdt,
        dt, Int32(M), Int32(inner_nx), Int32(inner_nz)
    )
    return nothing
end
