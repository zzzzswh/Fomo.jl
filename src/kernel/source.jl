# ==============================================================================
# Source Injection (CUDA Optimized)
# ==============================================================================

# 核心计算内核：只负责根据索引把值加进去
function _inject_field_at_kernel!(field, wavelet, sx, sz, k::Int32, n_src::Int32)
    # 修复 1i32 为 1
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_src
        @inbounds begin
            # 修复坐标顺序：第一个索引应为z方向，第二个索引应为x方向
            j0 = sx[idx]  # x坐标作为第二个索引（列）
            i0 = sz[idx]  # z坐标作为第一个索引（行）

            # 从 wavelet 读出当前时刻 k 的值
            # 假设 wavelet 的维度是 [n_src, nt] 或 [nt, n_src]
            # 注意：在 GPU 上，第一维是最快变化的，尽量保证连续访问
            wav = wavelet[idx, k]

            field[i0, j0] += wav
        end
    end
    return nothing
end

# 直接接收 4 个参数，专为 GPU 设计
function inject_source!(field, S::SourceConfig, k::Int, dt::Float32)
    # 1. 在 CPU 端准备好常量，转为 Int32
    n_src = Int32(length(S.sx))
    k32 = Int32(k)

    # 2. 线程配置
    threads = 256
    blocks = cld(n_src, threads)

    # 3. 直接启动 CUDA 内核
    @cuda threads = threads blocks = blocks _inject_field_at_kernel!(
        field,
        S.wavelet,
        S.sx,
        S.sz,
        k32,
        n_src
    )

    return nothing
end