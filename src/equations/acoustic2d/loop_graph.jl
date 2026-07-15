# src/equations/acoustic2d/loop_graph.jl
#
# CUDA Graph 主循环(声波,HABC,确定性两遍版):
#   整个时间步(计数器自增 → 融合速度更新 → det-HABC(vx,vz)
#   → 融合压力更新 → det-HABC(p) → 注入 → 三场记录,共 9 次 launch)
#   录制为一张 graph;此后每个时间步只需 1 次 graph launch。
#
# 与 _acoustic2d_loop_fused!(det 版)逐位等价:kernel 序列、参数、
# 数学完全一致,仅时间索引改由设备侧计数器提供(取值相同)。
#
# 限制:graph 内不能做主机端拷贝,故不支持快照;
#       snap_interval > 0 时上层会改走 _acoustic2d_loop_fused!。
# 兜底:capture/instantiate 失败(旧驱动等)时警告并回退融合循环,
#       结果不受影响。

using CUDA

function _acoustic2d_loop_graph!(W, M, S, R, B,
    a_static, dt, nt, pad,
    nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_p, seis_vx, seis_vz)

    total = _habc_frame_total(nx, nz, nbc)
    scr = CUDA.zeros(Float32, 2 * total)
    kbuf = CuArray(Int32[0])
    nt32 = Int32(nt)

    step!() = begin
        bump_counter!(kbuf)
        fused_update_velocity_acoustic!(W, M, a_static, dt, nbc)
        apply_habc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        fused_update_pressure_acoustic!(W, M, a_static, dt, nbc)
        apply_habc_det_1!(W.p, W.p_old, B.w_tau, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        inject_source_dev!(W.p, S, kbuf, nt32)
        record_receivers3_dev!(seis_p, seis_vx, seis_vz, W.p, W.vx, W.vz, R, kbuf, nt32)
        return nothing
    end

    # 1) 先真实执行一步,触发全部 kernel 编译(capture 期间禁止 JIT)
    step!()
    CUDA.synchronize()
    # 2) 抹掉预热步的痕迹
    reset!(W)
    fill!(seis_p, 0.0f0)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)
    fill!(kbuf, Int32(0))
    # 3) 录制 + 实例化;失败则回退融合循环(功能等价,仅小网格性能略降)
    exec = try
        graph = CUDA.capture() do
            step!()
        end
        CUDA.instantiate(graph)
    catch err
        @warn "CUDA Graph 捕获失败,回退到融合循环" exception = err
        nothing
    end
    if exec === nothing
        reset!(W)
        fill!(seis_p, 0.0f0)
        fill!(seis_vx, 0.0f0)
        fill!(seis_vz, 0.0f0)
        _acoustic2d_loop_fused!(W, M, S, R, B, a_static, dt, nt, pad,
            nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
            seis_p, seis_vx, seis_vz, Matrix{Float32}[], 0)
        return nothing
    end
    # 4) 回放 nt 次:每个时间步 1 次 graph launch
    for _ in 1:nt
        CUDA.launch(exec)
    end
    CUDA.synchronize()
    return nothing
end
