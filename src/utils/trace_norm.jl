# src/utils/trace_norm.jl

"""
    trace_norm(data::AbstractMatrix{T}; dims=2, epsilon=0)

Apply trace normalization to a matrix.

Parameters
----------
data: AbstractMatrix{T}
    Input matrix.
dims: Int
    Dimension along which to normalize. 
    like input (n_gathers, n_time_steps) the dim = 1
    like input (n_time_steps, n_gathers) the dim = 2
epsilon: Float
    Small value to prevent division by zero.

Returns
-------
Normalized matrix.
"""
function trace_norm(data::AbstractMatrix{T}; dims=2, epsilon=1e-10) where T

    max_vals = maximum(abs, data, dims=dims)

    # 2. 防止除以 0 放大噪声：把极小的值替换为 1.0
    # T() 保证类型稳定，如果 data 是 Float32，1.0 也会变成 1.0f0
    max_vals[max_vals.<=T(epsilon)] .= T(1.0)

    return data ./ max_vals
end