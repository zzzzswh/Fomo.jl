# src/simulator/elastic2d.jl
using ProgressMeter
using TimerOutputs  # 新增：强大的计时分析库

# 声明一个全局定时器
const to = TimerOutput()

function _run_core!(W, M, S, R, H, dt, nt, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)
    pad = M.pad
    snap_idx = 1
    p = Progress(nt, 1, "Simulation: ")

    for it in 1:nt
        # A update_velocity
        @timeit to "1. Backup Bounds" begin
            CUDA.@sync backup_boundary!(W, H, M)
        end

        @timeit to "2. Velocity Update" begin
            CUDA.@sync update_velocity!(W, M, a_static, dt, inner_nx, inner_nz)
        end

        @timeit to "3. HABC Velocity" begin
            CUDA.@sync apply_habc_velocity!(W, H, M)
        end

        # B update_stress
        @timeit to "1. Backup Bounds" begin # 名字相同会自动合并统计
            CUDA.@sync backup_boundary!(W, H, M)
        end

        @timeit to "4. Stress Update" begin
            CUDA.@sync update_stress!(W, M, a_static, dt, inner_nx, inner_nz)
        end

        @timeit to "5. HABC Stress" begin
            CUDA.@sync apply_habc_stress!(W, H, M)
        end

        # C inject_source
        @timeit to "6. Inject Source" begin
            CUDA.@sync inject_source!(W.txx, S, it, dt)
            CUDA.@sync inject_source!(W.tzz, S, it, dt)
        end

        # D record seismograms
        @timeit to "7. Record Receivers" begin
            CUDA.@sync record_receivers!(seis_vx, W.vx, R, it)
            CUDA.@sync record_receivers!(seis_vz, W.vz, R, it)
        end

        # E save snapshots
        if snap_interval > 0 && it % snap_interval == 0
            @timeit to "8. Snapshots (D2H)" begin
                # 💡 关键优化：加上 @view 避免在 GPU 端产生临时的内存分配
                snaps[snap_idx] = Array(@view W.vz[pad+1:end-pad, pad+1:end-pad])
                snap_idx += 1
            end
        end

        # 进度条本身也有开销，单独抓出来看看
        @timeit to "9. Progress Bar" begin
            if it % 100 == 0
                ProgressMeter.update!(p, it)
            end
        end
    end
end

function run_simulation!(W::Wavefield, M::Medium, S::SourceConfig, R::ReceiverConfig, H::HABCConfig, dt, nt, fd_order::Int; snap_interval::Int=0)
    a_static = get_fd_coefficients(fd_order)
    pad = M.pad
    inner_nx, inner_nz = M.nx - fd_order, M.nz - fd_order
    num_receivers = length(R.rx)

    seis_vx = fill!(similar(W.vz, num_receivers, nt), 0.0f0)
    seis_vz = fill!(similar(W.vz, num_receivers, nt), 0.0f0)

    if snap_interval > 0
        num_snaps = nt ÷ snap_interval
        snaps = Vector{Matrix{Float32}}(undef, num_snaps)
    else
        snaps = Vector{Matrix{Float32}}()
    end

    # 💡 每次运行前清空旧的计时数据
    reset_timer!(to)

    @info "Starting Simulation..."

    _run_core!(W, M, S, R, H, dt, nt, a_static, inner_nx, inner_nz, seis_vx, seis_vz, snaps, snap_interval)

    @info "Simulation Complete!"

    # 💡 打印超详细的性能剖析表，按耗时排序
    println("\n$(repeat("=", 60))")
    println("🔍 CORE LOOP PERFORMANCE BREAKDOWN")
    println("$(repeat("=", 60))")
    show(to; sortby=:time, compact=false)
    println("\n$(repeat("=", 60))\n")

    return Array(seis_vx), Array(seis_vz), snaps
end