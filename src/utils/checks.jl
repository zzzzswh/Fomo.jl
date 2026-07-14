# src/utils/checks.jl
#
# 入口参数校验：
#   - 几何越界（inject/record kernel 均为 @inbounds 直写，越界 = 静默显存踩踏）
#   - CFL 稳定性（超限直接 error，避免跑出一屏 NaN 再回头排查）
#   - 频散提示（最短波长采样点数不足时 @warn）

function _check_geometry(nx::Int, nz::Int, sx, sz, rx, rz)
    (all(1 .<= sx .<= nx) && all(1 .<= sz .<= nz)) ||
        throw(ArgumentError("震源索引越界: 需 1 ≤ sx ≤ $nx, 1 ≤ sz ≤ $nz"))
    (all(1 .<= rx .<= nx) && all(1 .<= rz .<= nz)) ||
        throw(ArgumentError("检波点索引越界: 需 1 ≤ rx ≤ $nx, 1 ≤ rz ≤ $nz"))
    return nothing
end

"""
    _warn_dispersion(vmin_disp, f0, dh; ppw_min=4.0)

频散提示。fmax 按 Ricker 有效最高频 ≈ 2.5·f0 估计；
使用自定义子波时请按实际带宽语义给 f0。
"""
function _warn_dispersion(vmin_disp::Real, f0::Real, dh::Real; ppw_min::Real=4.0)
    fmax = 2.5f0 * f0
    ppw = vmin_disp / fmax / dh
    ppw < ppw_min && @warn "频散风险: 最短波长仅 $(round(ppw, digits=1)) 个网格点 " *
        "(vmin=$vmin_disp, fmax≈$(round(fmax, digits=1)) Hz)，建议减小 dh 或降低 f0"
    return nothing
end

"""
    _check_numerics(vmax, vmin_disp, dh, dt, f0, fd_order; ppw_min=4.0)

交错网格 leapfrog（acoustic2d/elastic2d/3D）的 CFL 稳定性 + 频散检查。
- `vmax`: 稳定性用（max(vp)）
- `vmin_disp`: 频散用（弹性传 min(vs>0)，声波传 min(vp>0)）

2D 稳定性条件：dt ≤ dh / (vmax · √2 · Σ|a_l|)，a_l 为交错网格 FD 系数。
"""
function _check_numerics(vmax::Real, vmin_disp::Real, dh::Real, dt::Real,
    f0::Real, fd_order::Int; ppw_min::Real=4.0)
    a = get_fd_coefficients(fd_order)
    dt_max = dh / (vmax * sqrt(2.0f0) * sum(abs, a))
    dt <= dt_max || throw(ArgumentError(
        "CFL 不稳定: dt=$dt > dt_max=$(round(dt_max, sigdigits=4)) " *
        "(vmax=$vmax, dh=$dh, fd_order=$fd_order)"))
    _warn_dispersion(vmin_disp, f0, dh; ppw_min=ppw_min)
    return nothing
end

"""
    _check_numerics_centered(vmax, vmin_disp, dh, dt, f0, fd_order; ppw_min=4.0)

正则网格中心差分二阶方程（coupled2d）的 CFL 稳定性 + 频散检查。

单维 -∂²ₓ 离散算子最大特征值 ≤ (|c0| + 2Σ|c_l|)/h²，2D 两个维度相加，
leapfrog 稳定要求 dt²·λ·v² ≤ 4：
    dt ≤ 2·dh / (vmax · √(2·(|c0| + 2Σ|c_l|)))
（2 阶时退化为经典 dt ≤ dh/(v·√2)。耦合方程还含变系数一阶项，
此界为主项估计，接近上限时请留安全余量。）
"""
function _check_numerics_centered(vmax::Real, vmin_disp::Real, dh::Real, dt::Real,
    f0::Real, fd_order::Int; ppw_min::Real=4.0)
    c0, d2 = get_centered_d2(fd_order)
    lam_max = abs(c0) + 2.0f0 * sum(abs, d2)
    dt_max = 2.0f0 * dh / (vmax * sqrt(2.0f0 * lam_max))
    dt <= dt_max || throw(ArgumentError(
        "CFL 不稳定 (centered): dt=$dt > dt_max=$(round(dt_max, sigdigits=4)) " *
        "(vmax=$vmax, dh=$dh, fd_order=$fd_order)"))
    _warn_dispersion(vmin_disp, f0, dh; ppw_min=ppw_min)
    return nothing
end
