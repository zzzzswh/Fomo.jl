# src/equations/coupled2d/wavefield.jl
#
# 耦合 P-S 势场波场（速度-位置分裂格式）
#
# 将二阶 leapfrog 拆成两个一阶半步：
#   Ṗ^{n+1/2} = Ṗ^{n-1/2} + dt·F_P     ("速度"半步)
#   P^{n+1}   = P^n       + dt·Ṗ^{n+1/2} ("位置"半步)
#
# 物理场（4 个）：
#   P, S        — 势场（"位置"，整数时间步 n, n+1, ...）
#   dPdt, dSdt  — 势场时间导数（"速度"，半整数时间步 n±1/2）
#
# HABC 备份（4 个）：
#   P_old, S_old, dPdt_old, dSdt_old  — 每个半步前备份边界值
#
# 总共 8 个场（vs 弹性波 10 个，减少 20%）

using CUDA

mutable struct CoupledWavefield{T}
    # ── 物理场 ──
    P::T            # P 波势场
    dPdt::T         # ∂P/∂t（"速度"）
    S::T            # S 波势场 Sy
    dSdt::T         # ∂Sy/∂t（"速度"）
    # ── HABC 边界备份 ──
    P_old::T
    dPdt_old::T
    S_old::T
    dSdt_old::T
end

function CoupledWavefield(nx::Int, nz::Int, pad::Int)
    nx_pad = nx + 2 * pad
    nz_pad = nz + 2 * pad
    z() = CUDA.zeros(Float32, nx_pad, nz_pad)
    return CoupledWavefield(z(), z(), z(), z(), z(), z(), z(), z())
end

function reset!(W::CoupledWavefield)
    for f in (W.P, W.dPdt, W.S, W.dSdt,
              W.P_old, W.dPdt_old, W.S_old, W.dSdt_old)
        fill!(f, 0.0f0)
    end
end
