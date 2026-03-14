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

nx = 300
nz = 200
dh = 10.0f0
dz = 10.0f0
nt = 1000
dt = 0.001f0
fd_order = 8
nbc = 50

p = SimParams(dt, nt, dh, fd_order)

vp = 2000.0f0 .* ones(Float32, nx, nz) # nx行，nz列？
vs = 1000.0f0 .* ones(Float32, nx, nz)
rho = 2000.0f0 .* ones(Float32, nx, nz)

vp[:, 100:end] .= 3000.0f0
vs[:, 100:end] .= 1500.0f0
rho[:, 100:end] .= 2500.0f0

println("Plotting Vp model...")
# 注意这里的 vp' 是转置操作，用来让 nx 对应 X 轴，nz 对应 Y 轴
p_vp = heatmap(1:nx, 1:nz, vp',
    title="Vp Velocity Model (Two-layer)",
    xlabel="Distance X (grid points)",
    ylabel="Depth Z (grid points)",
    c=:viridis,        # 也可以换成 :jet 等其他色带
    yflip=true,        # 翻转 Y 轴，让深度 Z 向下递增
    aspect_ratio=:equal, # 保持 X 和 Z 的比例 1:1，避免图像变形
    colorbar_title="Velocity (m/s)"
)

# 保存图像查看
savefig(p_vp, "vp_model_check.png")
println("Vp model saved as vp_model_check.png")

println("Initializing ...")

M_med = init_medium(vp, vs, rho, dh, nbc, fd_order)
H = init_habc(M_med.nx, M_med.nz, nbc, M_med.pad, p.dt, dh, 2000.0f0)

f0 = 15.0f0
wavelet_data = ricker_wavelet(f0, dt, nt)

# 定义震源位置 (x方向为水平距离, z方向为深度)
src_x_idx = [round(Int, nx / 2)]        # 中间位置作为震源x坐标
src_z_idx = [10]                         # 浅层作为震源z坐标
pad = M_med.pad
sx = src_x_idx .+ pad                    # 添加边界padding偏移
sz = src_z_idx .+ pad

wavelet_matrix = reshape(wavelet_data, 1, nt)  # 时间维度为nt

println("Initializing Source...")
# 创建震源配置：指定震源在扩展网格中的位置(sx, sz)及波形数据
S = create_source_config(
    Int32.(sx),   # x方向索引（水平）
    Int32.(sz),   # z方向索引（垂直）
    wavelet_matrix
)

# 接收器布设：沿x方向每隔一个点放置一个接收器，固定z深度=2
rec_x_idx = 1:2:nx                       # 接收器x坐标序列
rec_z_idx = ones(Int, length(rec_x_idx)) .* 2   # 所有接收器位于z=2处

rx = rec_x_idx .+ pad                    # 添加padding后的实际x位置
rz = rec_z_idx .+ pad                    # 添加padding后的实际z位置

println("Initializing Receivers...")
# 创建接收器配置：记录vx或vz分量
R = ReceiverConfig(
    Int32.(rx),   # x方向位置
    Int32.(rz),   # z方向位置
    nt,           # 记录时间步数
    :vz          # 记录垂直速度分量
)

println("Initializing Wavefield...")
W = Wavefield(M_med.nx, M_med.nz)

# ==============================================================================
# 8. 运行极简版正演模拟 (带进度条)
# ==============================================================================
println("Starting Simulation...")

# snap_interval=50 表示每50个时间步保存一次快照，用于视频制作
vx, vz, snaps = run_simulation!(W, M_med, S, R, H, p; snap_interval=50)

# ==============================================================================
# 9. 保存地震道集图
# ==============================================================================
println("Saving seismogram...")
vz = trace_norm(vz, dims=1)
plot_shot(vz, "vz.png")

# 另外生成VX方向的地震图
println("Saving VX seismogram...")
vx_norm = trace_norm(vx, dims=1)
plot_shot(vx_norm, "vx.png")

# 同时保存合成地震图（如果有两个分量的数据）
println("Creating combined visualization...")
combined_seis = [vz vx]  # 合并两个分量的数据
plot_shot(combined_seis, "combined_seis.png")

# ==============================================================================
# 10. 生成波场视频
# ==============================================================================
println("Generating wavefield video...")

plot_wavefield_video(snaps, 50, "wavefield.mp4", fps=10)