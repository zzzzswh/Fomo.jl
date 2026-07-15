# src/Fomo.jl
#
# 新增: coupled2d — 耦合 P-S 势场求解器
# 基于 Li et al. (2018) "Elastic RTM using acoustic propagators"

module Fomo

using CUDA
using StaticArrays

# 1. 工具（2D）
include("utils/to_device.jl")
include("utils/ricker_wavelet.jl")
include("utils/FD_utils.jl")
include("utils/FD_centered.jl")         # [新增] 正则网格中心差分系数
include("utils/checks.jl")              # [新增] 入口参数校验（几何/CFL/频散）
include("utils/trace_norm.jl")
include("utils/pad_array.jl")           # 2D pad + buoyancy

# 1b. 工具（3D）
include("utils/pad_array_3d.jl")        # 3D pad + buoyancy

# 2. 采集系统（2D）
include("acquisition/source.jl")
include("acquisition/receiver.jl")
include("acquisition/inject_source.jl")
include("acquisition/record_receiver.jl")

# 2b. 采集系统（3D）
include("acquisition/source_3d.jl")
include("acquisition/receiver_3d.jl")
include("acquisition/inject_source_3d.jl")
include("acquisition/record_receiver_3d.jl")
include("acquisition/device_indexed.jl")   # [新增] 设备侧时间索引注入/记录（CUDA Graphs 用）
include("acquisition/batch_acq.jl")         # [新增] 多炮批处理采集

# 3. 边界条件（2D）
include("boundary/habc/habc.jl")
include("boundary/habc/kernels.jl")
include("boundary/habc/kernels_fused.jl")  # [新增] 帧映射多场融合 HABC
include("boundary/habc/kernels_det.jl")    # [新增] 确定性两遍 HABC
include("boundary/habc/kernels_det_batch.jl")  # [新增] 确定性 HABC（多炮批处理版）
include("boundary/sponge.jl")           # [新增] Sponge 吸收边界

# 3b. 边界条件（3D）
include("boundary/habc/habc_3d.jl")
include("boundary/habc/kernels_3d.jl")

# 4. 弹性波方程（2D）
include("equations/elastic2d/medium.jl")
include("equations/elastic2d/wavefield.jl")
include("equations/elastic2d/update_velocity.jl")
include("equations/elastic2d/update_stress.jl")
include("equations/elastic2d/fused_kernels.jl")  # [新增] 融合内核 + 融合循环
include("equations/elastic2d/loop_graph.jl")     # [新增] CUDA Graph 循环
include("equations/elastic2d/elastic2d.jl")
include("equations/elastic2d/batch.jl")          # [新增] 多炮批处理

# 5. 声波方程（2D）
include("equations/acoustic2d/medium.jl")
include("equations/acoustic2d/wavefield.jl")
include("equations/acoustic2d/update_velocity.jl")
include("equations/acoustic2d/update_pressure.jl")
include("equations/acoustic2d/fused_kernels.jl")  # [新增] 融合内核 + 融合循环
include("equations/acoustic2d/loop_graph.jl")     # [新增] CUDA Graph 循环
include("equations/acoustic2d/acoustic2d.jl")
include("equations/acoustic2d/batch.jl")          # [新增] 多炮批处理
include("equations/scalar2d/scalar2d.jl")         # [新增] 二阶标量求解器

# 6. 弹性波方程（3D）
include("equations/elastic3d/medium.jl")
include("equations/elastic3d/wavefield.jl")
include("equations/elastic3d/update_velocity.jl")
include("equations/elastic3d/update_stress.jl")
include("equations/elastic3d/elastic3d.jl")

# 7. 声波方程（3D）
include("equations/acoustic3d/medium.jl")
include("equations/acoustic3d/wavefield.jl")
include("equations/acoustic3d/update_velocity.jl")
include("equations/acoustic3d/update_pressure.jl")
include("equations/acoustic3d/acoustic3d.jl")

# 8. [新增] 耦合 P-S 势场方程（2D）
include("equations/coupled2d/medium.jl")
include("equations/coupled2d/wavefield.jl")
include("equations/coupled2d/update_fields.jl")
include("equations/coupled2d/coupled2d.jl")

# 9. 可视化（package extension：需 `using Plots` 才加载，见 ext/FomoPlotsExt.jl）
#    这样 `using Fomo` 不再拖入 Plots/GR，显著缩短加载时间
"""
    plot_shot(shot, save_path)

绘制炮记录热力图并保存。需先 `using Plots` 以加载 FomoPlotsExt 扩展。
"""
function plot_shot end

"""
    plot_wavefield_video(snaps, snapshot_interval, save_path; fps=10, adaptive_clims=false)

将波场快照导出为视频。需先 `using Plots` 以加载 FomoPlotsExt 扩展。
"""
function plot_wavefield_video end

# ── Exports ──
export elastic2d                        # 2D弹性波
export acoustic2d                       # 2D声波
export acoustic2d_batch                 # [新增] 2D声波多炮批处理
export elastic2d_batch                  # [新增] 2D弹性波多炮批处理
export scalar2d                         # [新增] 2D二阶标量（deepwave scalar 对位）
export coupled2d                        # [新增] 2D耦合P-S势场
export elastic3d                        # 3D弹性波
export acoustic3d                       # 3D声波
export ricker_wavelet
export trace_norm
export plot_shot, plot_wavefield_video

end  # module
