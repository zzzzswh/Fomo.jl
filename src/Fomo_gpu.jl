# src/Fomo_gpu.jl

module Fomo_gpu

using CUDA
using StaticArrays

# 1. 工具
include("utils/to_device.jl")
include("utils/ricker_wavelet.jl")
include("utils/FD_utils.jl")
include("utils/trace_norm.jl")
include("utils/pad_array.jl")         # ← NEW: 共享的 pad + buoyancy

# 2. 采集系统
include("acquisition/source.jl")
include("acquisition/receiver.jl")
include("acquisition/inject_source.jl")
include("acquisition/record_receiver.jl")

# 3. 边界条件
include("boundary/habc/habc.jl")
include("boundary/habc/kernels.jl")

# 4. 弹性波方程
include("equations/elastic2d/medium.jl")
include("equations/elastic2d/wavefield.jl")
include("equations/elastic2d/update_velocity.jl")
include("equations/elastic2d/update_stress.jl")
include("equations/elastic2d/elastic2d.jl")

# 5. 声波方程                          ← NEW
include("equations/acoustic2d/medium.jl")
include("equations/acoustic2d/wavefield.jl")
include("equations/acoustic2d/update_velocity.jl")
include("equations/acoustic2d/update_pressure.jl")
include("equations/acoustic2d/acoustic2d.jl")

# 6. 可视化
include("visualization/plot_shot.jl")
include("visualization/plot_video.jl")

# ── Exports ──
export elastic2d                        # 弹性波
export acoustic2d                       # 声波 ← NEW
export ricker_wavelet
export trace_norm
export plot_shot, plot_wavefield_video

end  # module

