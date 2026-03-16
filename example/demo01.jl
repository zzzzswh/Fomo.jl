# example/demo.jl
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
nz = 300
dh = 10.0f0
nt = 2000
dt = 0.001f0
f0 = 15.0f0
fd_order = 8
nbc = 100

sx = [round(Int, nx / 2)]        # 中间位置作为震源x坐标
sz = [10]                         # 浅层作为震源z坐标

rx = 1:2:nx                       # 接收器x坐标序列
rz = ones(Int, length(rx)) .* 10   # 所有接收器位于z=2处

# 1. 创建你的原始模型 (比如 200x200)
vp = fill(2500.0f0, nx, nz)
vs = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# 2. 挖去地形，将真空区域的参数设为 0
# 例如：把深度 z < 30 的区域设为真空
vp[:, 1] .= 0.0f0
vs[:, 1] .= 0.0f0
rho[:, 1] .= 0.0f0

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
@info "init medium success"
HABC = init_habc(nx, nz, medium.pad, dt, dh, 2000.0f0)
@info "init HABC success"
wavefield = Wavefield(nx, nz, medium.pad)
@info "init wavefield success"

wavelet_data = ricker_wavelet(f0, dt, nt)
wavelet_matrix = reshape(wavelet_data, 1, nt)  # 时间维度为nt

source = init_source(
    medium.pad,
    Int32.(sx),   # x方向索引（水平）
    Int32.(sz),   # z方向索引（垂直）
    wavelet_matrix
)

@info "init source success"

receiver = init_receiver(
    medium.pad,
    Int32.(rx),   # x方向位置
    Int32.(rz),   # z方向位置
    nt,           # 记录时间步数
    :vz          # 记录垂直速度分量
)
@info "init receiver success"

# RUN!!!
println("Starting Simulation...")
vx, vz, snaps = run_simulation!(wavefield, medium, source, receiver, HABC, dt, nt, fd_order; snap_interval=50)

# Plots
vz = trace_norm(vz, dims=2)
plot_shot(vz, "vz.png")

vx_norm = trace_norm(vx, dims=2)
plot_shot(vx_norm, "vx.png")

plot_wavefield_video(snaps, 50, "wavefield.mp4", fps=10, adaptive_clims=true)

