# src/boundary/habc/kernels_fused.jl
#
# HABC 融合内核（性能补丁核心之二）
#
# 旧实现的两个问题：
#   1. 每个场一次 kernel launch（声波 3 次 / 弹性 5 次），launch 开销在小中网格上占主导
#   2. launch 网格覆盖整个 padded 网格，但只有边界框架内的线程做事
#      （nx=nz≈500, nbc≈53 时约 60% 线程空转；网格越大浪费越多）
#
# 新实现：
#   - 线程只映射到边界框架（4 个矩形的 1D 展开），零空转
#   - 一次 launch 同时处理 2 个场（vx+vz）或 3 个场（txx+tzz+txz）
#   - Higdon 公式与 _habc_2d_kernel! 逐字一致（含角落平均），
#     并保留原实现同样的邻点 race（见 kernels.jl 中的说明），
#     因此数值行为与旧路径在容差内一致
#
using CUDA

# ──────────────────────────────────────────────────────────────────────────────
# 帧区域 1D 索引 → (i, j)
#
# 覆盖区域（与 _habc_2d_kernel! 的活跃区完全一致）：
#   top    : i ∈ [2, nx-1],        j ∈ [2, nbc+1]
#   bottom : i ∈ [2, nx-1],        j ∈ [nz-nbc, nz-1]
#   left   : i ∈ [2, nbc+1],       j ∈ [nbc+2, nz-nbc-1]
#   right  : i ∈ [nx-nbc, nx-1],   j ∈ [nbc+2, nz-nbc-1]
# ──────────────────────────────────────────────────────────────────────────────
@inline function _habc_frame_ij(idx::Int32, nx::Int32, nz::Int32, nbc::Int32)
    Wd = nx - Int32(2)                     # 水平带宽度
    Nh = Wd * nbc                          # 单条水平带的点数
    if idx <= Nh                           # top
        q, r = divrem(idx - Int32(1), Wd)
        return (Int32(2) + r, Int32(2) + q)
    elseif idx <= Int32(2) * Nh            # bottom
        t = idx - Nh - Int32(1)
        q, r = divrem(t, Wd)
        return (Int32(2) + r, nz - nbc + q)
    else                                   # left + right 侧柱
        t = idx - Int32(2) * Nh - Int32(1)
        q, r = divrem(t, Int32(2) * nbc)   # q: 行号, r: 列号（左 nbc 列 + 右 nbc 列）
        j = nbc + Int32(2) + q
        i = r < nbc ? Int32(2) + r : nx - nbc + (r - nbc)
        return (i, j)
    end
end

"""帧区域线程总数（host 端计算）。"""
function _habc_frame_total(nx::Integer, nz::Integer, nbc::Integer)
    Wd = Int(nx) - 2
    Hc = Int(nz) - 2 * Int(nbc) - 2        # 中段高度（padded 网格恒 > 0）
    return 2 * Wd * Int(nbc) + 2 * Int(nbc) * Hc
end

# ──────────────────────────────────────────────────────────────────────────────
# 单点 Higdon 修正 —— 与 _habc_2d_kernel! 的数学完全一致
# ──────────────────────────────────────────────────────────────────────────────
@inline function _habc_apply_point!(f, f_old, w,
    i::Int32, j::Int32,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32)

    is_left = (i <= nbc + Int32(1))
    is_right = (i >= nx - nbc)
    is_top = (j <= nbc + Int32(1))
    is_bottom = (j >= nz - nbc)

    in_x = is_left | is_right
    in_z = is_top | is_bottom

    @inbounds begin
        wt = w[i, j]
        f_cur = f[i, j]

        if in_x & in_z
            sum_x = is_left ?
                    (-qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]) :
                    (-qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j])
            sum_z = is_top ?
                    (-qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]) :
                    (-qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1])
            f[i, j] = wt * f_cur + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        elseif in_x
            sum_x = is_left ?
                    (-qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]) :
                    (-qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j])
            f[i, j] = wt * f_cur + (1.0f0 - wt) * sum_x
        elseif in_z
            sum_z = is_top ?
                    (-qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]) :
                    (-qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1])
            f[i, j] = wt * f_cur + (1.0f0 - wt) * sum_z
        end
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# 帧映射 kernel：1 / 2 / 3 个场共用一次 launch
# ──────────────────────────────────────────────────────────────────────────────
function _habc_frame_1!(f, f_old, w,
    qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        _habc_apply_point!(f, f_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    end
    return nothing
end

function _habc_frame_2!(f1, f1_old, w1, f2, f2_old, w2,
    qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        _habc_apply_point!(f1, f1_old, w1, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        _habc_apply_point!(f2, f2_old, w2, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    end
    return nothing
end

function _habc_frame_3!(f1, f1_old, f2, f2_old, f3, f3_old, w,
    qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        _habc_apply_point!(f1, f1_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        _habc_apply_point!(f2, f2_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        _habc_apply_point!(f3, f3_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# host 端包装
# ──────────────────────────────────────────────────────────────────────────────
function apply_habc_frame_1!(f, f_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_frame_1!(
        f, f_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    return nothing
end

function apply_habc_frame_2!(f1, f1_old, w1, f2, f2_old, w2,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_frame_2!(
        f1, f1_old, w1, f2, f2_old, w2, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    return nothing
end

function apply_habc_frame_3!(f1, f1_old, f2, f2_old, f3, f3_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_frame_3!(
        f1, f1_old, f2, f2_old, f3, f3_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    return nothing
end
