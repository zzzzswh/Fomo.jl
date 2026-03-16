# src/kernel/inject_source.jl

# 核心计算内核：只负责根据索引把值加进去
function _inject_field_at_kernel!(field, wavelet, sx, sz, k::Int32, n_src::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= n_src
        @inbounds begin
            # 【修复】：现在 field 是 (nx, nz)，所以第一维是 x (sx)，第二维是 z (sz)
            ix = sx[idx]
            iz = sz[idx]

            # 从 wavelet 读出当前时刻 k 的值
            # 【点赞】：你这里的 wavelet[idx, k] 写得非常完美！
            # 因为 idx 是当前线程号，在 Julia 的列主序下，第一维是最快变化的，
            # 这保证了 GPU 的合并访存 (Coalesced Memory Access)，读取效率拉满。
            wav = wavelet[idx, k]

            # 注入震源（比如加到速度场或应力场）
            field[ix, iz] += wav
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