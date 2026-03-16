# src/kernel/HABC/update_HABC.jl
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

# ==============================================================================
# 1. 大一统边界备份 Kernel (同时处理 5 个场)
# ==============================================================================
@parallel_indices (i, j) function backup_all_boundaries_kernel!(
    vx_old, vx, vz_old, vz, txx_old, txx, tzz_old, tzz, txz_old, txz,
    nbc, nx, nz
)
    if i <= nx && j <= nz
        # 只备份边缘厚度为 nbc+2 的区域，跳过庞大的中心网格
        if i <= nbc + 2 || i >= nx - nbc - 1 || j <= nbc + 2 || j >= nz - nbc - 1
            vx_old[i, j] = vx[i, j]
            vz_old[i, j] = vz[i, j]
            txx_old[i, j] = txx[i, j]
            tzz_old[i, j] = tzz[i, j]
            txz_old[i, j] = txz[i, j]
        end
    end
    return nothing
end

function backup_boundary!(W, H, M)
    # 【核心修复：将结构体属性提取为强类型本地变量，彻底杀灭 CPU 动态分配】
    nx::Int = M.nx
    nz::Int = M.nz
    nbc::Int = H.nbc
    vx_o = W.vx_old
    vx = W.vx
    vz_o = W.vz_old
    vz = W.vz
    txx_o = W.txx_old
    txx = W.txx
    tzz_o = W.tzz_old
    tzz = W.tzz
    txz_o = W.txz_old
    txz = W.txz

    @parallel (1:nx, 1:nz) backup_all_boundaries_kernel!(
        vx_o, vx, vz_o, vz, txx_o, txx, tzz_o, tzz, txz_o, txz,
        nbc, nx, nz
    )
    return nothing
end

# ==============================================================================
# 2. HABC 核函数 (左右、上下、四角整合版)
# ==============================================================================
@parallel_indices (iy) function habc_lr_edges_kernel!(f, f_old, w, qx, qt_x, qxt, nx, nz, nbc)
    j = iy
    if j >= 1 && j <= nz
        # Left (正向循环)
        for i in 2:nbc+1
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_x
        end
        # Right (反向循环)
        for i in nx-1:-1:nx-nbc
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_x
        end
    end
    return nothing
end

@parallel_indices (ix) function habc_tb_edges_kernel!(f, f_old, w, qz, qt_z, qxt, nx, nz, nbc)
    i = ix
    if i >= 1 && i <= nx
        # Top (正向循环)
        for j in 2:nbc+1
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_z
        end
        # Bottom (反向循环)
        for j in nz-1:-1:nz-nbc
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * sum_z
        end
    end
    return nothing
end

@parallel_indices (idx) function habc_corners_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    if idx == 1
        # Left-Top
        for i in 2:nbc+1, j in 2:nbc+1
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        end
        # Right-Top
        for i in nx-1:-1:nx-nbc, j in 2:nbc+1
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        end
        # Left-Bottom
        for i in 2:nbc+1, j in nz-1:-1:nz-nbc
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        end
        # Right-Bottom
        for i in nx-1:-1:nx-nbc, j in nz-1:-1:nz-nbc
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            wt = w[i, j]
            f[i, j] = wt * f[i, j] + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        end
    end
    return nothing
end

# ==============================================================================
# 3. 顶层应用接口
# ==============================================================================
function apply_habc_single_field!(f, f_old, w, H, M)
    # 【核心修复：同上，消灭结构体解包带来的 CPU 开销】
    nx::Int = M.nx
    nz::Int = M.nz
    nbc::Int = H.nbc
    qx::Float32 = H.qx
    qz::Float32 = H.qz
    qt_x::Float32 = H.qt_x
    qt_z::Float32 = H.qt_z
    qxt::Float32 = H.qxt

    @parallel (1:nz) habc_lr_edges_kernel!(f, f_old, w, qx, qt_x, qxt, nx, nz, nbc)
    @parallel (1:nx) habc_tb_edges_kernel!(f, f_old, w, qz, qt_z, qxt, nx, nz, nbc)
    @parallel (1:1) habc_corners_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    return nothing
end

function apply_habc_velocity!(W, H, M)
    apply_habc_single_field!(W.vx, W.vx_old, H.w_vx, H, M)
    apply_habc_single_field!(W.vz, W.vz_old, H.w_vz, H, M)
    return nothing
end

function apply_habc_stress!(W, H, M)
    apply_habc_single_field!(W.txx, W.txx_old, H.w_tau, H, M)
    apply_habc_single_field!(W.tzz, W.tzz_old, H.w_tau, H, M)
    apply_habc_single_field!(W.txz, W.txz_old, H.w_tau, H, M)
    return nothing
end