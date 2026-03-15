using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots
using Printf
using StaticArrays
using CUDA

# 移除这一行，因为在模块内部已经初始化了
# @init_parallel_stencil(CUDA, Float32, 2)

# 直接包含本地开发的包
include("../src/Fomo_gpu.jl")
using .Fomo_gpu

nx = 400
nz = 200
dh = 10.0f0
nt = 2000
dt = 0.001f0
f0 = 15.0f0
fd_order = 8
nbc = 50

sx = [round(Int, nx / 2)]        # 中间位置作为震源x坐标
sz = [10]                         # 浅层作为震源z坐标

rx = 1:2:nx                       # 接收器x坐标序列
rz = ones(Int, length(rx)) .* 2   # 所有接收器位于z=2处

vp = 2000.0f0 .* ones(Float32, nx, nz)
vs = 1000.0f0 .* ones(Float32, nx, nz)
rho = 2000.0f0 .* ones(Float32, nx, nz)

vp[:, 100:end] .= 3000.0f0
vs[:, 100:end] .= 1500.0f0
rho[:, 100:end] .= 2500.0f0

println("Plotting Vp model...")
p_vp = heatmap(1:nx, 1:nz, vp',
    title="Vp Velocity Model (Two-layer)",
    xlabel="Distance X (grid points)",
    ylabel="Depth Z (grid points)",
    c=:viridis,        # 也可以换成 :jet 等其他色带
    yflip=true,        # 翻转 Y 轴，让深度 Z 向下递增
    aspect_ratio=:equal, # 保持 X 和 Z 的比例 1:1，避免图像变形
    colorbar_title="Velocity (m/s)"
)
savefig(p_vp, "vp_model_check.png")


# CFL check
println("\n[Diagnostics] Checking CFL stability condition...")
v_max = maximum(vp)

# 8阶交错网格有限差分系数绝对值之和近似为 1.196
# (c1=1225/1024, c2=-245/3072, c3=49/5120, c4=-5/7168)
coef_sum_8th = 1.19625f0

# 2D CFL 极限时间步长公式
dt_max = dh / (v_max * sqrt(2.0f0) * coef_sum_8th)

@printf("  -> Max Velocity (Vp): %8.2f m/s\n", v_max)
@printf("  -> Grid spacing (dh): %8.2f m\n", dh)
@printf("  -> Current dt:        %8.6f s\n", dt)
@printf("  -> Max allowed dt:    %8.6f s\n", dt_max)

if dt > dt_max
    println()
    @error "💥 CFL CONDITION VIOLATED! 💥"
    @error "Your dt ($dt) is larger than the theoretical stability limit ($dt_max)."
    @error "The simulation will become unstable and generate NaNs."
    @error "Please either decrease `dt` to be <= $dt_max or increase `dh`."
    error("Simulation aborted due to CFL violation.") # 直接中断程序运行
else
    println("  -> CFL check passed! ✅\n")
end

println("Initializing ...")

medium = init_medium(vp, vs, rho, dh, nbc, fd_order)
HABC = init_habc(nx, nz, medium.pad, dt, dh, 2000.0f0)
wavefield = Wavefield(nx, nz, medium.pad)

wavelet_data = ricker_wavelet(f0, dt, nt)
wavelet_matrix = reshape(wavelet_data, 1, nt)  # 时间维度为nt

source = init_source(
    medium.pad,
    Int32.(sx),   # x方向索引（水平）
    Int32.(sz),   # z方向索引（垂直）
    wavelet_matrix
)

receiver = init_receiver(
    medium.pad,
    Int32.(rx),   # x方向位置
    Int32.(rz),   # z方向位置
    nt,           # 记录时间步数
    :vz          # 记录垂直速度分量
)


# RUN!!!
println("Starting Simulation...")
vx, vz, snaps = run_simulation!(wavefield, medium, source, receiver, HABC, dt, nt, fd_order; snap_interval=50)

# Plots
vz = trace_norm(vz, dims=2)
plot_shot(vz, "vz.png")

vx_norm = trace_norm(vx, dims=2)
plot_shot(vx_norm, "vx.png")

plot_wavefield_video(snaps, 50, "wavefield.mp4", fps=10)

using CUDA

println("\n" * "="^40)
println("🚀 PERFORMANCE BENCHMARKING 🚀")
println("="^40)

# 1. 预热 (Warm-up)：只跑 10 步，触发所有的 JIT 编译和 GPU 内核加载
println("Warming up JIT compiler and GPU...")
run_simulation!(wavefield, medium, source, receiver, HABC, dt, 10, fd_order; snap_interval=0)

# 2. 真实测速 (Benchmarking)
# 注意：务必将 snap_interval 设为 0！
# 因为 GPU 传数据回 CPU (Array) 非常慢，测纯算力时必须屏蔽这种 I/O 瓶颈。
println("\nRunning full simulation for benchmarking ($nt steps)...")

# 使用 CUDA.@time 测量：它会自动同步 GPU 并报告真实的显存分配
CUDA.@time run_simulation!(wavefield, medium, source, receiver, HABC, dt, nt, fd_order; snap_interval=0)

# 3. 计算行业标准指标：吞吐量 (Mega-cells per second, MCells/s)
# CUDA.@elapsed 会返回严格同步后的纯 GPU 执行时间（秒）
t_sec = CUDA.@elapsed run_simulation!(wavefield, medium, source, receiver, HABC, dt, nt, fd_order; snap_interval=0)

# 计算总共处理了多少个网格点次
total_grid_updates = nx * nz * nt
mcells_per_sec = (total_grid_updates / t_sec) / 1e6

println("\n📊 --- Benchmark Results ---")
println("Grid Size   : $nx x $nz")
println("Time Steps  : $nt")
println("FD Order    : $fd_order")
println("Elapsed Time: ", round(t_sec, digits=4), " seconds")
println("Throughput  : ", round(mcells_per_sec, digits=2), " MCells/s (百万网格/秒)")
println("="^40)
