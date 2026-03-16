using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CUDA
using BenchmarkTools

include("../src/Fomo_gpu.jl")
using .Fomo_gpu

# Configure small-scale test for rapid profiling
const TEST_NX = 512
const TEST_NZ = 512
const TEST_NT = 100
const TEST_DH = 10.0f0
const TEST_DT = 0.001f0
const TEST_FD_ORDER = 8
const TEST_NBC = 50

function initialize_test_environment()
    # Create homogeneous medium for consistent testing
    vp = 2000.0f0 .* ones(Float32, TEST_NX, TEST_NZ)
    vs = 1000.0f0 .* ones(Float32, TEST_NX, TEST_NZ)
    rho = 2000.0f0 .* ones(Float32, TEST_NX, TEST_NZ)

    medium = init_medium(vp, vs, rho, TEST_DH, TEST_NBC, TEST_FD_ORDER)
    HABC = init_habc(TEST_NX, TEST_NZ, medium.pad, TEST_DT, TEST_DH, 2000.0f0)
    
    # Initialize wavefield with GPU buffers
    wavefield = Wavefield(TEST_NX, TEST_NZ, medium.pad)

    # Create minimal source/receiver for testing
    source = init_source(medium.pad, Int32[TEST_NX ÷ 2], Int32[10], reshape(ricker_wavelet(15.0f0, TEST_DT, TEST_NT), 1, TEST_NT))
    receiver = init_receiver(medium.pad, Int32.(1:2:TEST_NX), fill(Int32(2), TEST_NX÷2), TEST_NT, :vz)

    return (medium, HABC, wavefield, source, receiver)
end

function benchmark_core_components()
    (medium, HABC, wavefield, source, receiver) = initialize_test_environment()

    println("\n🚀 Starting Detailed Performance Benchmarking")
    println("----------------------------------------")
    println("Test Configuration:")
    println("  Grid size: $(TEST_NX) × $(TEST_NZ)")
    println("  Time steps: $TEST_NT")
    println("  FD order: $TEST_FD_ORDER")

    # Get FD coefficients as SVector
    fd_coeffs = get_fd_coefficients(TEST_FD_ORDER)
    M = length(fd_coeffs)
    inner_nx = TEST_NX - 2 * medium.pad
    inner_nz = TEST_NZ - 2 * medium.pad

    # Warm-up to trigger JIT compilation
    for _ in 1:5
        update_stress!(wavefield, medium, fd_coeffs, TEST_DT, inner_nx, inner_nz)
        update_velocity!(wavefield, medium, fd_coeffs, TEST_DT, inner_nx, inner_nz)
        inject_source!(wavefield.vx, source, 1, TEST_DT) # 添加了TEST_DT参数
        record_receivers!(receiver.data, wavefield.vz, receiver, 1) # 修正为record_receivers!
    end
    
    # Measure individual components
    stress_time = @belapsed update_stress!($wavefield, $medium, $fd_coeffs, $TEST_DT, $inner_nx, $inner_nz) samples=10
    velocity_time = @belapsed update_velocity!($wavefield, $medium, $fd_coeffs, $TEST_DT, $inner_nx, $inner_nz) samples=10
    source_time = @belapsed inject_source!($wavefield.vx, $source, 1, $TEST_DT) samples=10 # 添加了TEST_DT参数
    receiver_time = @belapsed record_receivers!($receiver.data, $wavefield.vz, $receiver, 1) samples=10 # 修正为record_receivers!

    total_time = stress_time + velocity_time + source_time + receiver_time
    
    println("\n⏱️  Component Timing Results (per time step):")
    println("  Stress update:    $(round(stress_time * 1e6, digits=2)) μs ($(round(stress_time/total_time*100, digits=1))%)")
    println("  Velocity update:  $(round(velocity_time * 1e6, digits=2)) μs ($(round(velocity_time/total_time*100, digits=1))%)")
    println("  Source injection: $(round(source_time * 1e6, digits=2)) μs ($(round(source_time/total_time*100, digits=1))%)")
    println("  Receiver record:  $(round(receiver_time * 1e6, digits=2)) μs ($(round(receiver_time/total_time*100, digits=1))%)")

    # Calculate throughput
    total_cells = TEST_NX * TEST_NZ * TEST_NT
    throughput = total_cells / (total_time * TEST_NT) / 1e6
    println("\n📊 Throughput: $(round(throughput, digits=2)) MCells/s")
    
    # Memory usage check
    println("\n💾 Memory Usage:")
    try
        pool = CUDA.Mem.pool()
        allocated = CUDA.Mem.usage(pool)
        reserved = CUDA.Mem.poolsize(pool)
        println("  Allocated: $(round(allocated / 1e6, digits=2)) MB")
        println("  Reserved:  $(round(reserved / 1e6, digits=2)) MB")
    catch e
        println("  Failed to retrieve memory info: $(sprint(showerror, e))")
    end
end

function main()
    # Ensure non-GUI backend for server environments
    ENV["GKSwstype"] = "nonsvg"

    println("\n\n$(repeat("=", 50))")
    println(" PERFORMANCE BOTTLENECK ANALYSIS TOOL")
    println("$(repeat("=", 50))")

    try
        benchmark_core_components()
    catch e
        println("\n❌ Error during benchmark: $(sprint(showerror, e))")
        rethrow()
    end

    println("\n$(repeat("=", 50))")
    println(" Analysis complete. Check component timing to identify bottlenecks.")
    println("$(repeat("=", 50))\n")
end

main()