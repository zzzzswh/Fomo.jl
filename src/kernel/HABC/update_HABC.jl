# src/kernel/HABC/update_HABC.jl
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@parallel_indices (ix, iy) function copy_strip_kernel!(old, new, start_i, start_j)
    # Translate indices to the specified rectangle starting point
    i = ix + start_i - 1
    j = iy + start_j - 1
    old[i, j] = new[i, j]
    return nothing
end

"""
    backup_boundary_field!(old, new, nbc, nx, nz)

Precisely backup boundary data of a single field variable. Only launch threads on the four rectangular strips where boundaries are located.
"""
function backup_boundary_field!(old, new, nbc, nx, nz)
    # Left and right strips (full height nz)
    @parallel (1:nbc+2, 1:nz) copy_strip_kernel!(old, new, 1, 1)                  # Left
    @parallel (1:nbc+2, 1:nz) copy_strip_kernel!(old, new, nx - nbc - 1, 1)           # Right

    # Top and bottom strips (trimmed width nx-2nbc-4)
    w_x = nx - 2 * nbc - 4
    @parallel (1:w_x, 1:nbc+2) copy_strip_kernel!(old, new, nbc + 3, 1)             # Top
    @parallel (1:w_x, 1:nbc+2) copy_strip_kernel!(old, new, nbc + 3, nz - nbc - 1)      # Bottom
    return nothing
end

function backup_boundary!(W, H::HABCConfig, M::Medium)
    nx, nz = M.nx, M.nz
    nbc = H.nbc

    backup_boundary_field!(W.vx_old, W.vx, nbc, nx, nz)
    backup_boundary_field!(W.vz_old, W.vz, nbc, nx, nz)
    backup_boundary_field!(W.txx_old, W.txx, nbc, nx, nz)
    backup_boundary_field!(W.tzz_old, W.tzz, nbc, nx, nz)
    backup_boundary_field!(W.txz_old, W.txz, nbc, nx, nz)
    return nothing
end

# ==============================================================================
# 2. HABC 核函数 (Edges - 1D) 没有任何 if 判断！
# ==============================================================================

@parallel_indices (ix, iy) function habc_left_edge_kernel!(f, f_old, w, qx, qt_x, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
    wt = w[j, i] # Note: maintaining the original w[j, i] indexing order
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_x
    return nothing
end

@parallel_indices (ix, iy) function habc_right_edge_kernel!(f, f_old, w, qx, qt_x, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_x
    return nothing
end

@parallel_indices (ix, iy) function habc_bottom_edge_kernel!(f, f_old, w, qz, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_z
    return nothing
end

@parallel_indices (ix, iy) function habc_top_edge_kernel!(f, f_old, w, qz, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_z
    return nothing
end

# ==============================================================================
# 3. HABC 核函数 (Corners - 2D) 用两边平均
# ==============================================================================

@parallel_indices (ix, iy) function habc_lb_corner_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
    return nothing
end

@parallel_indices (ix, iy) function habc_rb_corner_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
    return nothing
end

@parallel_indices (ix, iy) function habc_lt_corner_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
    return nothing
end

@parallel_indices (ix, iy) function habc_rt_corner_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, start_i, start_j)
    i, j = ix + start_i - 1, iy + start_j - 1
    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
    wt = w[j, i]
    f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
    return nothing
end

"""
    apply_habc_field!(f, f_old, H, weights, nx, nz)

Apply boundary updates for a single field variable. Precisely calculate dimensions and starting coordinates for the 8 boundary rectangles, then rapidly invoke GPU kernels.
"""
function apply_habc_field!(f, f_old, H::HABCConfig, weights, nx::Int, nz::Int)
    nbc = H.nbc
    qx, qz, qt_x, qt_z, qxt = H.qx, H.qz, H.qt_x, H.qt_z, H.qxt

    # --- 1. 计算各个维度的启动线程数量 (区域尺寸) ---
    len_edge_y = nz - 2 * nbc - 2  # 左右边缘的高度
    len_edge_x = nx - 2 * nbc - 2  # 上下边缘的宽度
    len_corner = nbc               # 角落的边长

    # --- 2. 处理 Edges (四条边) ---
    @parallel (1:nbc, 1:len_edge_y) habc_left_edge_kernel!(f, f_old, weights, qx, qt_x, qxt, 2, nbc + 2)
    @parallel (1:nbc, 1:len_edge_y) habc_right_edge_kernel!(f, f_old, weights, qx, qt_x, qxt, nx - nbc, nbc + 2)
    @parallel (1:len_edge_x, 1:nbc) habc_bottom_edge_kernel!(f, f_old, weights, qz, qt_z, qxt, nbc + 2, nz - nbc)
    @parallel (1:len_edge_x, 1:nbc) habc_top_edge_kernel!(f, f_old, weights, qz, qt_z, qxt, nbc + 2, 2)

    # --- 3. 处理 Corners (四个角) ---
    @parallel (1:nbc, 1:nbc) habc_lb_corner_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, 2, nz - nbc)
    @parallel (1:nbc, 1:nbc) habc_rb_corner_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, nx - nbc, nz - nbc)
    @parallel (1:nbc, 1:nbc) habc_lt_corner_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, 2, 2)
    @parallel (1:nbc, 1:nbc) habc_rt_corner_kernel!(f, f_old, weights, qx, qz, qt_x, qt_z, qxt, nx - nbc, 2)

    return nothing
end

# 顶层应用接口
function apply_habc_velocity!(W, H::HABCConfig, M::Medium)
    apply_habc_field!(W.vx, W.vx_old, H, H.w_vx, M.nx, M.nz)
    apply_habc_field!(W.vz, W.vz_old, H, H.w_vz, M.nx, M.nz)
    return nothing
end

function apply_habc_stress!(W, H::HABCConfig, M::Medium)
    apply_habc_field!(W.txx, W.txx_old, H, H.w_tau, M.nx, M.nz)
    apply_habc_field!(W.tzz, W.tzz_old, H, H.w_tau, M.nx, M.nz)
    apply_habc_field!(W.txz, W.txz_old, H, H.w_tau, M.nx, M.nz)
    return nothing
end