using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CUDA
using BenchmarkTools
using Printf

include("../src/Fomo_gpu.jl")
using .Fomo_gpu

# ==========================================
# 1. 参数配置 (与你的主程序保持完全一致)
# ==========================================
const NX = 400
const NZ = 300
const DH = 10.0f0
const NT = 2000
const DT = 0.001f0
const F0 = 15.0f0
const FD_ORDER = 8
const NBC = 50

function setup_simulation()
    # 震源与检波器几何配置
    sx = [round(Int, NX / 2)]
    sz = [10]
    rx = 1:2:NX
    rz = ones(Int, length(rx)) .* 10

    # 模型构建 (包含真空层)
    vp = fill(2500.0f0, NX, NZ)
    vs = fill(1200.0f0, NX, NZ)
    rho = fill(2000.0f0, NX, NZ)

    vp[:, 1] .= 0.0f0
    vs[:, 1] .= 0.0f0
    rho[:, 1] .= 0.0f0

    # 初始化 Fomo_gpu 组件
    medium = init_medium(vp, vs, rho, DH, NBC, FD_ORDER)
    HABC = init_habc(NX, NZ, medium.pad, DT, DH, 2000.0f0)
    wavefield = Wavefield(NX, NZ, medium.pad)

    wavelet_data = ricker_wavelet(F0, DT, NT)
    wavelet_matrix = reshape(wavelet_data, 1, NT)

    source = init_source(medium.pad, Int32.(sx), Int32.(sz), wavelet_matrix)
    receiver = init_receiver(medium.pad, Int32.(rx), Int32.(rz), NT, :vz)

    return medium, HABC, wavefield, source, receiver
end

# ==========================================
# 2. 核心性能压测模块
# ==========================================
function run_profiling()
    println("$(repeat("=", 50))")
    println("🚀 FOMO_GPU PERFORMANCE PROFILER")
    println("$(repeat("=", 50))")

    medium, HABC, wavefield, source, receiver = setup_simulation()

    # 提取内部网格尺寸和差分系数，用于微观测试
    inner_nx = NX - 2 * medium.pad
    inner_nz = NZ - 2 * medium.pad
    fd_coeffs = get_fd_coefficients(FD_ORDER)

    println("\n[1/3] JIT Warm-up (预热编译器)...")
    # 预热：跑很少的步数，触发Julia的JIT编译，不计入成绩
    run_simulation!(wavefield, medium, source, receiver, HABC, DT, 10, FD_ORDER; snap_interval=0)

    println("\n[2/3] Macro Benchmark (宏观吞吐量测试)...")
    # 宏观测试：测试整个 run_simulation! 的总耗时，这里跑 500 步作为基准
    test_nt = 500
    macro_time = @belapsed begin
        CUDA.@sync run_simulation!($wavefield, $medium, $source, $receiver, $HABC, $DT, $test_nt, $FD_ORDER; snap_interval=99999)
    end samples = 5

    total_cells = NX * NZ * test_nt
    throughput = (total_cells / macro_time) / 1e6
    @printf("  -> Total time for %d steps: %.4f seconds\n", test_nt, macro_time)
    @printf("  -> Overall Throughput:      %.2f MCells/s\n", throughput)

    println("\n[3/3] Micro Benchmark (微观内核剖析)...")
    # 微观测试：带 CUDA.@sync 的单步内核耗时，寻找绝对瓶颈

    stress_time = @belapsed begin
        CUDA.@sync update_stress!($wavefield, $medium, $fd_coeffs, $DT, $inner_nx, $inner_nz)
    end samples = 100

    velocity_time = @belapsed begin
        CUDA.@sync update_velocity!($wavefield, $medium, $fd_coeffs, $DT, $inner_nx, $inner_nz)
    end samples = 100

    source_time = @belapsed begin
        CUDA.@sync inject_source!($wavefield.vx, $source, 1, $DT)
    end samples = 100

    receiver_time = @belapsed begin
        CUDA.@sync record_receivers!($receiver.data, $wavefield.vz, $receiver, 1)
    end samples = 100

    total_micro = stress_time + velocity_time + source_time + receiver_time

    @printf("  -> Stress Update:   %8.2f μs (%.1f%%)\n", stress_time * 1e6, (stress_time / total_micro) * 100)
    @printf("  -> Velocity Update: %8.2f μs (%.1f%%)\n", velocity_time * 1e6, (velocity_time / total_micro) * 100)
    @printf("  -> Source Inject:   %8.2f μs (%.1f%%)\n", source_time * 1e6, (source_time / total_micro) * 100)
    @printf("  -> Receiver Record: %8.2f μs (%.1f%%)\n", receiver_time * 1e6, (receiver_time / total_micro) * 100)

    # 内存使用情况 (使用最新的 CUDA API)
    println("\n💾 GPU Memory Usage:")
    try
        mem_info = CUDA.memory_info()
        @printf("  -> Allocated: %.2f MB\n", mem_info.alloc / 1024^2)
        @printf("  -> Free:      %.2f MB\n", mem_info.free / 1024^2)
    catch e
        println("  -> Failed to retrieve memory info: $(e)")
    end
    println("$(repeat("=", 50))\n")
end

# 运行压测
run_profiling()