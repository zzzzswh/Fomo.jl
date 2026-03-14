module Fomo_gpu

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CUDA
using StaticArrays
using ProgressMeter

@init_parallel_stencil(CUDA, Float32, 2)

include("structures/sim_para.jl")
include("structures/medium.jl")
include("structures/wavefield.jl")
include("structures/source.jl")
include("structures/receiver.jl")
include("utils/to_device.jl")
include("utils/ricker_wavelet.jl")
include("initiate/init_medium.jl")
include("initiate/init_HABC.jl")
include("kernel/elastic2d/update_stress.jl")
include("kernel/elastic2d/update_velocity.jl")
include("kernel/HABC/update_HABC.jl")
include("kernel/inject_source.jl")
include("kernel/record_reciever.jl")
include("simulator/el2d_simulator.jl")
include("initiate/init_source.jl")
include("initiate/init_receiver.jl")
include("visualization/plot_video.jl")
include("visualization/plot_shot.jl")
include("utils/trace_norm.jl")

# 导出重要的函数
export update_stress!, update_velocity!
export run_simulation!
export inject_source!, record_receivers!
export init_source, init_receiver
export _run_core!
export init_medium, init_habc
export ricker_wavelet
export SimParams, Medium, Wavefield, ReceiverConfig, create_source_config
export plot_shot, plot_wavefield_video
export trace_norm

end  # module