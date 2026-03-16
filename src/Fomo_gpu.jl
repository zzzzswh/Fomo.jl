# src/Fomo_gpu.jl
module Fomo_gpu

using CUDA
using StaticArrays

# 1. 工具（无依赖，最先加载）
include("utils/to_device.jl")
include("utils/ricker_wavelet.jl")
include("utils/FD_utils.jl")
include("utils/trace_norm.jl")

# 2. 采集系统（通用）
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
#include("equations/elastic2d/vacuum.jl")
include("equations/elastic2d/update_velocity.jl")
include("equations/elastic2d/update_stress.jl")
include("equations/elastic2d/elastic2d.jl")    # 入口函数，最后加载

# 5. 可视化
include("visualization/plot_shot.jl")
include("visualization/plot_video.jl")

# ── Exports ──
# 用户只需要这几个
export elastic2d                     # 弹性波一站式 API
export ricker_wavelet                # 可能需要单独用
export trace_norm                    # 后处理
export plot_shot, plot_wavefield_video  # 可视化

end  # module
