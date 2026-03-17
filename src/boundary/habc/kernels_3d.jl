# src/boundary/habc/kernels_3d.jl
#
# 3D HABC kernels
# 从 2D 版本扩展：
#   - 所有 kernel 从 2D (i,j) 变为 3D (i,j,k)
#   - 边界判断增加 y 方向 (is_front / is_back)
#   - 角落/棱边取多方向平均

using CUDA

# ==============================================================================
# 1. 边界备份 —— 3D 版本
# ==============================================================================

function _backup_single_field_3d_cuda!(f_old, f, nbc, nx, ny, nz)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i <= nx && j <= ny && k <= nz
        if i <= nbc + 2 || i >= nx - nbc - 1 ||
           j <= nbc + 2 || j >= ny - nbc - 1 ||
           k <= nbc + 2 || k >= nz - nbc - 1
            @inbounds f_old[i, j, k] = f[i, j, k]
        end
    end
    return nothing
end

"""
    backup_single_field_3d!(dst, src, nbc, nx, ny, nz)

通用3D边界单场备份。
"""
function backup_single_field_3d!(dst, src, nbc::Int32, nx::Int32, ny::Int32, nz::Int32)
    threads = (8, 8, 4)
    blocks = (cld(nx, 8), cld(ny, 8), cld(nz, 4))
    @cuda threads=threads blocks=blocks _backup_single_field_3d_cuda!(
        dst, src, nbc, nx, ny, nz)
    return nothing
end

# --- Velocity phase: 备份 vx, vy, vz ---
function _backup_velocity_3d_cuda!(vx_o, vx, vy_o, vy, vz_o, vz, nbc, nx, ny, nz)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i <= nx && j <= ny && k <= nz
        if i <= nbc + 2 || i >= nx - nbc - 1 ||
           j <= nbc + 2 || j >= ny - nbc - 1 ||
           k <= nbc + 2 || k >= nz - nbc - 1
            @inbounds begin
                vx_o[i, j, k] = vx[i, j, k]
                vy_o[i, j, k] = vy[i, j, k]
                vz_o[i, j, k] = vz[i, j, k]
            end
        end
    end
    return nothing
end

function backup_velocity_3d!(W, H, M)
    nx = Int32(M.nx); ny = Int32(M.ny); nz = Int32(M.nz)
    nbc = Int32(H.nbc)
    threads = (8, 8, 4)
    blocks = (cld(nx, 8), cld(ny, 8), cld(nz, 4))
    @cuda threads=threads blocks=blocks _backup_velocity_3d_cuda!(
        W.vx_old, W.vx, W.vy_old, W.vy, W.vz_old, W.vz, nbc, nx, ny, nz)
    return nothing
end

# --- Stress phase: 备份 txx, tyy, tzz, txy, txz, tyz ---
function _backup_stress_3d_cuda!(
    txx_o, txx, tyy_o, tyy, tzz_o, tzz,
    txy_o, txy, txz_o, txz, tyz_o, tyz,
    nbc, nx, ny, nz)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i <= nx && j <= ny && k <= nz
        if i <= nbc + 2 || i >= nx - nbc - 1 ||
           j <= nbc + 2 || j >= ny - nbc - 1 ||
           k <= nbc + 2 || k >= nz - nbc - 1
            @inbounds begin
                txx_o[i, j, k] = txx[i, j, k]
                tyy_o[i, j, k] = tyy[i, j, k]
                tzz_o[i, j, k] = tzz[i, j, k]
                txy_o[i, j, k] = txy[i, j, k]
                txz_o[i, j, k] = txz[i, j, k]
                tyz_o[i, j, k] = tyz[i, j, k]
            end
        end
    end
    return nothing
end

function backup_stress_3d!(W, H, M)
    nx = Int32(M.nx); ny = Int32(M.ny); nz = Int32(M.nz)
    nbc = Int32(H.nbc)
    threads = (8, 8, 4)
    blocks = (cld(nx, 8), cld(ny, 8), cld(nz, 4))
    @cuda threads=threads blocks=blocks _backup_stress_3d_cuda!(
        W.txx_old, W.txx, W.tyy_old, W.tyy, W.tzz_old, W.tzz,
        W.txy_old, W.txy, W.txz_old, W.txz, W.tyz_old, W.tyz,
        nbc, nx, ny, nz)
    return nothing
end

# ==============================================================================
# 2. 全并行 3D HABC kernel
#
# 从 2D 版本扩展：
#   - 增加 y 方向 (is_front / is_back)
#   - 边界区域从 4 边 + 4 角 变为 6 面 + 12 棱 + 8 角
#   - 多方向边界交汇处取各方向 HABC 修正的平均值
# ==============================================================================

function _habc_3d_kernel!(f, f_old, w, qx, qy, qz, qt_x, qt_y, qt_z, qxt,
    nx, ny, nz, nbc)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    i = ix + Int32(1)
    j = iy + Int32(1)
    k = iz + Int32(1)

    if i > nx - 1 || j > ny - 1 || k > nz - 1
        return nothing
    end

    is_left   = (i <= nbc + 1)
    is_right  = (i >= nx - nbc)
    is_front  = (j <= nbc + 1)
    is_back   = (j >= ny - nbc)
    is_top    = (k <= nbc + 1)
    is_bottom = (k >= nz - nbc)

    in_x = is_left  | is_right
    in_y = is_front | is_back
    in_z = is_top   | is_bottom

    # 内部点跳过
    if !in_x && !in_y && !in_z
        return nothing
    end

    @inbounds begin
        wt = w[i, j, k]
        f_cur = f[i, j, k]

        sum_total = 0.0f0
        n_dirs = Int32(0)

        # x 方向 HABC
        if in_x
            if is_left
                sum_total += -qx * f[i+1, j, k] - qt_x * f_old[i, j, k] - qxt * f_old[i+1, j, k]
            else
                sum_total += -qx * f[i-1, j, k] - qt_x * f_old[i, j, k] - qxt * f_old[i-1, j, k]
            end
            n_dirs += Int32(1)
        end

        # y 方向 HABC
        if in_y
            if is_front
                sum_total += -qy * f[i, j+1, k] - qt_y * f_old[i, j, k] - qxt * f_old[i, j+1, k]
            else
                sum_total += -qy * f[i, j-1, k] - qt_y * f_old[i, j, k] - qxt * f_old[i, j-1, k]
            end
            n_dirs += Int32(1)
        end

        # z 方向 HABC
        if in_z
            if is_top
                sum_total += -qz * f[i, j, k+1] - qt_z * f_old[i, j, k] - qxt * f_old[i, j, k+1]
            else
                sum_total += -qz * f[i, j, k-1] - qt_z * f_old[i, j, k] - qxt * f_old[i, j, k-1]
            end
            n_dirs += Int32(1)
        end

        # 多方向取平均
        f[i, j, k] = wt * f_cur + (1.0f0 - wt) * sum_total / Float32(n_dirs)
    end

    return nothing
end

# ==============================================================================
# 3. 顶层接口
# ==============================================================================

function apply_habc_single_field_3d!(f, f_old, w, qx, qy, qz, qt_x, qt_y, qt_z, qxt,
    nx, ny, nz, nbc)
    threads = (8, 8, 4)
    blocks = (cld(Int(nx) - 2, 8), cld(Int(ny) - 2, 8), cld(Int(nz) - 2, 4))
    @cuda threads=threads blocks=blocks _habc_3d_kernel!(
        f, f_old, w, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    return nothing
end

function apply_habc_velocity_3d!(W, H, M)
    nx = Int32(M.nx); ny = Int32(M.ny); nz = Int32(M.nz)
    nbc = Int32(H.nbc)
    qx = Float32(H.qx); qy = Float32(H.qy); qz = Float32(H.qz)
    qt_x = Float32(H.qt_x); qt_y = Float32(H.qt_y); qt_z = Float32(H.qt_z)
    qxt = Float32(H.qxt)

    apply_habc_single_field_3d!(W.vx, W.vx_old, H.w_vx, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.vy, W.vy_old, H.w_vy, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.vz, W.vz_old, H.w_vz, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    return nothing
end

function apply_habc_stress_3d!(W, H, M)
    nx = Int32(M.nx); ny = Int32(M.ny); nz = Int32(M.nz)
    nbc = Int32(H.nbc)
    qx = Float32(H.qx); qy = Float32(H.qy); qz = Float32(H.qz)
    qt_x = Float32(H.qt_x); qt_y = Float32(H.qt_y); qt_z = Float32(H.qt_z)
    qxt = Float32(H.qxt)

    apply_habc_single_field_3d!(W.txx, W.txx_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.tyy, W.tyy_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.tzz, W.tzz_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.txy, W.txy_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.txz, W.txz_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    apply_habc_single_field_3d!(W.tyz, W.tyz_old, H.w_tau, qx, qy, qz, qt_x, qt_y, qt_z, qxt, nx, ny, nz, nbc)
    return nothing
end
