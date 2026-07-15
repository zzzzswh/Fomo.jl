# src/boundary/habc/kernels_det.jl
#
# 确定性两遍 HABC —— 消除 kernels.jl 中已文档化的邻点 race。
#
#   pass 1:每个帧点只读【修正前】的 f 与 f_old,把修正值写入 scratch;
#   pass 2:把 scratch 统一写回 f。
#
# 这正是原实现注释所述的本意("f[i±1,j] 读的是 PDE 更新后的值"——
# 即邻点应读到 HABC 修正前的 PDE 值),racy 版只是概率性地做到这一点。
# 两遍版在任何调度、任何 launch 几何、任何显卡上逐位可复现。
#
# 代价:每次调用 2 次 launch(racy 版 1 次),但只覆盖帧区线程;
#       配合 CUDA Graphs(整步 1 次 graph launch)后额外代价 ≈ 0。
#
# scratch 布局:一维 Float32 向量,长度 ≥ n_fields × total,
#               第 m 个场占 [(m-1)*total+1, m*total]。
# 依赖:_habc_frame_ij / _habc_frame_total(kernels_fused.jl)。

using CUDA

# ── 单点修正值(纯函数:所有读取都来自传入数组的当前内容)─────────────────
@inline function _habc_det_value(f, f_old, w,
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
            return wt * f_cur + (1.0f0 - wt) * 0.5f0 * (sum_x + sum_z)
        elseif in_x
            sum_x = is_left ?
                    (-qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]) :
                    (-qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j])
            return wt * f_cur + (1.0f0 - wt) * sum_x
        else   # 帧内点必属 in_x | in_z,此分支即纯 in_z
            sum_z = is_top ?
                    (-qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]) :
                    (-qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1])
            return wt * f_cur + (1.0f0 - wt) * sum_z
        end
    end
end

# ── pass 1 / pass 2 内核:1、2、3 个场 ─────────────────────────────────────
function _habc_det_pass1_1!(scr, f, f_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds scr[idx] = _habc_det_value(f, f_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
    end
    return nothing
end

function _habc_det_pass2_1!(scr, f, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds f[i, j] = scr[idx]
    end
    return nothing
end

function _habc_det_pass1_2!(scr, f1, f1_old, w1, f2, f2_old, w2,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds begin
            scr[idx] = _habc_det_value(f1, f1_old, w1, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[idx+total] = _habc_det_value(f2, f2_old, w2, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        end
    end
    return nothing
end

function _habc_det_pass2_2!(scr, f1, f2, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds begin
            f1[i, j] = scr[idx]
            f2[i, j] = scr[idx+total]
        end
    end
    return nothing
end

function _habc_det_pass1_3!(scr, f1, f1_old, f2, f2_old, f3, f3_old, w,
    qx::Float32, qz::Float32, qt_x::Float32, qt_z::Float32, qxt::Float32,
    nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds begin
            scr[idx] = _habc_det_value(f1, f1_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[idx+total] = _habc_det_value(f2, f2_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
            scr[idx+total+total] = _habc_det_value(f3, f3_old, w, i, j, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        end
    end
    return nothing
end

function _habc_det_pass2_3!(scr, f1, f2, f3, nx::Int32, nz::Int32, nbc::Int32, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= total
        i, j = _habc_frame_ij(idx, nx, nz, nbc)
        @inbounds begin
            f1[i, j] = scr[idx]
            f2[i, j] = scr[idx+total]
            f3[i, j] = scr[idx+total+total]
        end
    end
    return nothing
end

# ── host 包装 ──────────────────────────────────────────────────────────────
"""确定性 HABC(单场)。scratch 长度须 ≥ _habc_frame_total(nx,nz,nbc)。"""
function apply_habc_det_1!(f, f_old, w, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_det_pass1_1!(
        scr, f, f_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _habc_det_pass2_1!(
        scr, f, nx, nz, nbc, total)
    return nothing
end

"""确定性 HABC(双场,各自权重)。scratch 长度须 ≥ 2×total。"""
function apply_habc_det_2!(f1, f1_old, w1, f2, f2_old, w2, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_det_pass1_2!(
        scr, f1, f1_old, w1, f2, f2_old, w2, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _habc_det_pass2_2!(
        scr, f1, f2, nx, nz, nbc, total)
    return nothing
end

"""确定性 HABC(三场,共享权重)。scratch 长度须 ≥ 3×total。"""
function apply_habc_det_3!(f1, f1_old, f2, f2_old, f3, f3_old, w, scr,
    qx, qz, qt_x, qt_z, qxt, nx::Int32, nz::Int32, nbc::Int32)
    total = Int32(_habc_frame_total(nx, nz, nbc))
    threads = 256
    blocks = cld(Int(total), threads)
    @cuda threads = threads blocks = blocks _habc_det_pass1_3!(
        scr, f1, f1_old, f2, f2_old, f3, f3_old, w, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, total)
    @cuda threads = threads blocks = blocks _habc_det_pass2_3!(
        scr, f1, f2, f3, nx, nz, nbc, total)
    return nothing
end
