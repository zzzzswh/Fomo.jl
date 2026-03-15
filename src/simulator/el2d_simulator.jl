using ProgressMeter

function _run_core!(W, M, S, R, H, dt, nt, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)
    pad = M.pad

    # 用于记录快照存到了第几个
    snap_idx = 1

    @showprogress 1 "Simulation: " for it in 1:nt
        # A update_velocity
        backup_boundary!(W, H, M)
        # ⚠️ 注意：这里去掉了 p，改传 dt。
        # 你需要同步修改你的 update_velocity! 和 update_stress! 函数签名！
        update_velocity!(W, M, a_static, dt, inner_nx, inner_nz)
        apply_habc_velocity!(W, H, M)

        # B update_stress
        backup_boundary!(W, H, M)
        update_stress!(W, M, a_static, dt, inner_nx, inner_nz)
        apply_habc_stress!(W, H, M)

        # C inject_source
        inject_source!(W.txx, S, it, dt)
        inject_source!(W.tzz, S, it, dt)

        # D record seismograms
        record_receivers!(seis_vx, W.vx, R, it)
        record_receivers!(seis_vz, W.vz, R, it)

        # E save snapshots (移到了循环内部！)
        if snap_interval > 0 && it % snap_interval == 0
            # 在 GPU 上进行切片，然后再传回 CPU (Array)，减少总传输量
            snaps[snap_idx] = Array(W.vz[pad+1:end-pad, pad+1:end-pad])
            snap_idx += 1
        end
    end
end

"""
Minimized entry function:
Only need to pass in snap_interval to return vx receiver data, vz receiver data, and wavefield snapshots list.
"""
function run_simulation!(W::Wavefield, M::Medium, S::SourceConfig, R::ReceiverConfig, H::HABCConfig, dt, nt, fd_order::Int; snap_interval::Int=0)
    a_static = get_fd_coefficients(fd_order)
    pad = M.pad

    inner_nx, inner_nz = M.nx - fd_order, M.nz - fd_order

    num_receivers = length(R.rx)

    # 在 GPU 上分配接收器数据内存，并原地填充 0.0f0
    seis_vx = fill!(similar(W.vz, num_receivers, nt), 0.0f0)
    seis_vz = fill!(similar(W.vz, num_receivers, nt), 0.0f0)

    # 正确处理快照数组的预分配
    if snap_interval > 0
        num_snaps = nt ÷ snap_interval
        # 预先分配好指定长度的 Vector，避免 push! 带来的内存重新分配开销
        snaps = Vector{Matrix{Float32}}(undef, num_snaps)
    else
        snaps = Vector{Matrix{Float32}}() # 不保存快照时为空数组
    end

    @info "Starting Simulation..."

    # 删除了原来游离在这里的 if snap_interval 代码块（因为这里没有 it 变量）
    # 调用核心循环，注意把 M_med 改成了 M，去掉了 p，传入了 dt 和 nt
    _run_core!(W, M, S, R, H, dt, nt, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)

    @info "Simulation Complete!"

    return Array(seis_vx), Array(seis_vz), snaps
end