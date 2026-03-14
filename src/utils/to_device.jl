using CUDA

"""
    to_device(x)

Transfer data to the GPU.
"""
function to_device(x)
    return CuArray(x)
end