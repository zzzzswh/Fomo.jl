# src/boundary/habc/kernels_det_batch.jl
#
# 确定性两遍 HABC 的多炮批处理版:
#   - 场为 (nx, nz, n_shots) 三维;权重 w 与 Higdon 系数各炮共享(二维/标量)
#   - 帧线程沿 x 维,炮沿 blockIdx().y
#   - scratch 布局:每炮占 n_fields×total,炮 s 的场 m 偏移
#     off = (s-1)·n_fields·total + (m-1)·total
# 数学与 kernels_det.jl 逐字一致,仅多一个炮下标 —— 各炮之间零交互,
# 因此每炮结果与单炮 det 版逐位相等。

using CUDA

@inline function _habc_det_value_b(f, f_old, w,
    i::Int32, j::Int32, s::Int32,
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
        f_cur = f[i, j, s]
        if in_x & in_z
            sum_x = is_left ?
                    (-qx * f[i+1, j, s] - qt_x * f_old[i, j, s] - qxt * f_old[i+1, j, s]) :
                    (-qx * f[i-1, j, s] - qt_x * f_old[i, j, s] - qxt * f_old[i-1, j, s])
            sum_z = is_top ?
                    (-qz * f[i, j+1, s] - qt_z * f_old[i, j, s] - qxt * f_old[i, j+1, s]) :
                    (-qz * f[i, j-1, s] - qt_z * f_old[i, j, s] - qxt * f_old[i, j-1, s])
            return wt * f_cur + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        elseif in_x
            sum_x = is_left ?
                    (-qx * f[i+1, j, s] - qt_x * f_old[i, j, s] - qxt * f_old[i+1, j, s]) :
                    (-qx * f[i-1, j, s] - qt_x * f_old[i, j, s] - qxt * f_old[i-1, j, s])
            return wt * f_cur + (1.0f0 - wt) * sum_x
        else
            sum_z = is_top ?
                    (-qz * f[i, j+1, s] - qt_z * f_old[i, j, s] - qxt * f_old[i, j+1, s]) :
                    (-qz * f[i, j-1, s] - qt_z * f_old[i, j, s] - qxt * f_old[i, j-1, s])
            return wt * f_cur + (1.0f0 - wt) * sum_z
        end
    end
end

# ── 单场 ──────────────────────────────────────────────────────────────────
function _bhabc_det_pass1_1!(scr, f, f_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * total
        @inbounds scr[off+idx] = _habc_det_value_b(f, f_old, w, i, j, s,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    end
    return nothing
end

function _bhabc_det_pass2_1!(scr, f, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * total
        @inbounds f[i, j, s] = scr[off+idx]
    end
    return nothing
end

"""批处理确定性 HABC(单场)。scratch 长度须 ≥ total×n_shots。"""
function apply_bhabc_det_1!(f, f_old, w, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32, ns::Int)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = (cld(Int(total), threads), ns)
    @cuda threads = threads blocks = blocks _bhabc_det_pass1_1!(
        scr, f, f_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _bhabc_det_pass2_1!(
        scr, f, nx, nz, nbc, total)
    return nothing
end

# ── 双场(各自权重)─────────────────────────────────────────────────────────
function _bhabc_det_pass1_2!(scr, f1, f1_old, w1, f2, f2_old, w2,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * Int32(2) * total
        @inbounds begin
            scr[off+idx] = _habc_det_value_b(f1, f1_old, w1, i, j, s,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[off+total+idx] = _habc_det_value_b(f2, f2_old, w2, i, j, s,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        end
    end
    return nothing
end

function _bhabc_det_pass2_2!(scr, f1, f2, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * Int32(2) * total
        @inbounds begin
            f1[i, j, s] = scr[off+idx]
            f2[i, j, s] = scr[off+total+idx]
        end
    end
    return nothing
end

"""批处理确定性 HABC(双场)。scratch 长度须 ≥ 2×total×n_shots。"""
function apply_bhabc_det_2!(f1, f1_old, w1, f2, f2_old, w2, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32, ns::Int)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = (cld(Int(total), threads), ns)
    @cuda threads = threads blocks = blocks _bhabc_det_pass1_2!(
        scr, f1, f1_old, w1, f2, f2_old, w2, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _bhabc_det_pass2_2!(
        scr, f1, f2, nx, nz, nbc, total)
    return nothing
end

# ── 三场(共享权重)─────────────────────────────────────────────────────────
function _bhabc_det_pass1_3!(scr, f1, f1_old, f2, f2_old, f3, f3_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * Int32(3) * total
        @inbounds begin
            scr[off+idx] = _habc_det_value_b(f1, f1_old, w, i, j, s,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[off+total+idx] = _habc_det_value_b(f2, f2_old, w, i, j, s,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[off+total+total+idx] = _habc_det_value_b(f3, f3_old, w, i, j, s,
                qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        end
    end
    return nothing
end

function _bhabc_det_pass2_3!(scr, f1, f2, f3, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s = blockIdx().y
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        off = (s - Int32(1)) * Int32(3) * total
        @inbounds begin
            f1[i, j, s] = scr[off+idx]
            f2[i, j, s] = scr[off+total+idx]
            f3[i, j, s] = scr[off+total+total+idx]
        end
    end
    return nothing
end

"""批处理确定性 HABC(三场)。scratch 长度须 ≥ 3×total×n_shots。"""
function apply_bhabc_det_3!(f1, f1_old, f2, f2_old, f3, f3_old, w, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32, ns::Int)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = (cld(Int(total), threads), ns)
    @cuda threads = threads blocks = blocks _bhabc_det_pass1_3!(
        scr, f1, f1_old, f2, f2_old, f3, f3_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _bhabc_det_pass2_3!(
        scr, f1, f2, f3, nx, nz, nbc, total)
    return nothing
end
