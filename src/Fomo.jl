# src/Fomo.jl

module Fomo

using CUDA
using StaticArrays

# 1. 工具（2D）
include("utils/to_device.jl")
include("utils/ricker_wavelet.jl")
include("utils/FD_utils.jl")
include("utils/trace_norm.jl")
include("utils/pad_array.jl")         # 2D pad + buoyancy

# 1b. 工具（3D）
include("utils/pad_array_3d.jl")      # 3D pad + buoyancy

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

# 3. 边界条件（2D）
include("boundary/habc/habc.jl")
include("boundary/habc/kernels.jl")

# 3b. 边界条件（3D）
include("boundary/habc/habc_3d.jl")
include("boundary/habc/kernels_3d.jl")

# 4. 弹性波方程（2D）
include("equations/elastic2d/medium.jl")
include("equations/elastic2d/wavefield.jl")
include("equations/elastic2d/update_velocity.jl")
include("equations/elastic2d/update_stress.jl")
include("equations/elastic2d/elastic2d.jl")

# 5. 声波方程（2D）
include("equations/acoustic2d/medium.jl")
include("equations/acoustic2d/wavefield.jl")
include("equations/acoustic2d/update_velocity.jl")
include("equations/acoustic2d/update_pressure.jl")
include("equations/acoustic2d/acoustic2d.jl")

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

# 8. 可视化
include("visualization/plot_shot.jl")
include("visualization/plot_video.jl")

# ── Exports ──
export elastic2d                        # 2D弹性波
export acoustic2d                       # 2D声波
export elastic3d                        # 3D弹性波
export acoustic3d                       # 3D声波
export ricker_wavelet
export trace_norm
export plot_shot, plot_wavefield_video

end  # module

