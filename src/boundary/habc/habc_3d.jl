# src/boundary/habc/habc_3d.jl
#
# 3D Higdon 吸收边界条件
# 从 2D 版本扩展：增加 y 方向的边界处理

using CUDA

struct HABCConfig3D{T}
    nbc::Int
    qx::Float32
    qy::Float32
    qz::Float32
    qt_x::Float32
    qt_y::Float32
    qt_z::Float32
    qxt::Float32
    w_vx::T
    w_vy::T
    w_vz::T
    w_tau::T
end

function init_habc_3d(nx::Int, ny::Int, nz::Int, pad::Int,
    dt::Real, dh::Real, v_ref::Real)
    nx_pad = nx + 2pad
    ny_pad = ny + 2pad
    nz_pad = nz + 2pad

    r = Float32(v_ref * dt / dh)
    b_p = 0.45f0
    beta = 1.0f0

    q  = Float32((b_p * (beta + r) - r) / ((beta + r) * (1 - b_p)))
    qt = Float32((b_p * (beta + r) - beta) / ((beta + r) * (1 - b_p)))
    qxt_val = Float32(b_p / (b_p - 1.0f0))

    # 3D距离函数：到6个面的最小距离
    dist(i, j, k) = min(i - 1, nx_pad - i, j - 1, ny_pad - j, k - 1, nz_pad - k)

    w_vx  = [Float32(clamp((dist(i,j,k) - 0.0)  / (pad - 1), 0.0, 1.0)) for i in 1:nx_pad, j in 1:ny_pad, k in 1:nz_pad]
    w_vy  = [Float32(clamp((dist(i,j,k) - 0.25) / (pad - 1), 0.0, 1.0)) for i in 1:nx_pad, j in 1:ny_pad, k in 1:nz_pad]
    w_vz  = [Float32(clamp((dist(i,j,k) - 0.5)  / (pad - 1), 0.0, 1.0)) for i in 1:nx_pad, j in 1:ny_pad, k in 1:nz_pad]
    w_tau = [Float32(clamp((dist(i,j,k) - 0.75) / (pad - 1), 0.0, 1.0)) for i in 1:nx_pad, j in 1:ny_pad, k in 1:nz_pad]

    return HABCConfig3D(
        pad - 1,
        q, q, q,       # qx, qy, qz
        qt, qt, qt,     # qt_x, qt_y, qt_z
        qxt_val,
        to_device(w_vx),
        to_device(w_vy),
        to_device(w_vz),
        to_device(w_tau)
    )
end
