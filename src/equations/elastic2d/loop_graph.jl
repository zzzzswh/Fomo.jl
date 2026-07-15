# src/equations/elastic2d/loop_graph.jl
#
# CUDA Graph 主循环(弹性波,HABC,确定性两遍版):
#   计数器自增 → 融合速度更新 → det-HABC(vx,vz) → 融合应力更新
#   → det-HABC(txx,tzz,txz) → 双场注入 → 双场记录,共 9 次 launch,
#   录制为一张 graph,每个时间步 1 次 graph launch。
# 与 _elastic2d_loop_fused!(det 版)逐位等价;不支持快照;
# capture 失败自动回退融合循环。

using CUDA

function _elastic2d_loop_graph!(W, M, S, R, B,
    a_static, dt, nt, pad,
    seis_vx, seis_vz)

    nx = Int32(M.nx)
    nz = Int32(M.nz)
    nbc = Int32(B.nbc)
    qx = Float32(B.qx)
    qz = Float32(B.qz)
    qt_x = Float32(B.qt_x)
    qt_z = Float32(B.qt_z)
    qxt = Float32(B.qxt)

    total = _habc_frame_total(nx, nz, nbc)
    scr = CUDA.zeros(Float32, 3 * total)
    kbuf = CuArray(Int32[0])
    nt32 = Int32(nt)

    step!() = begin
        bump_counter!(kbuf)
        fused_update_velocity_elastic!(W, M, a_static, dt, nbc)
        apply_habc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        fused_update_stress_elastic!(W, M, a_static, dt, nbc)
        apply_habc_det_3!(W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
            B.w_tau, scr, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        inject_source_pair_dev!(W.txx, W.tzz, S, kbuf, nt32)
        record_receivers2_dev!(seis_vx, seis_vz, W.vx, W.vz, R, kbuf, nt32)
        return nothing
    end

    # 1) 预热编译
    step!()
    CUDA.synchronize()
    # 2) 清场
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)
    fill!(kbuf, Int32(0))
    # 3) 录制 + 实例化,失败回退
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
        fill!(seis_vx, 0.0f0)
        fill!(seis_vz, 0.0f0)
        _elastic2d_loop_fused!(W, M, S, R, B, a_static, dt, nt, pad,
            seis_vx, seis_vz, Matrix{Float32}[], 0)
        return nothing
    end
    # 4) 回放
    for _ in 1:nt
        CUDA.launch(exec)
    end
    CUDA.synchronize()
    return nothing
end
