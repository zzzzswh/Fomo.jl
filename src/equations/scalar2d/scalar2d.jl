# src/equations/scalar2d/scalar2d.jl
#
# 二阶标量(常密度)声波方程求解器 —— deepwave scalar 的同公式对位。
#
#   u^{n+1} = 2uⁿ − u^{n−1} + (v·dt/dh)²·[dh²·∇²uⁿ]     (中心差分,leapfrog)
#
# 为什么值得单列一个求解器:一阶 v-p 交错格式每步约 12 次数组遍历
# (p/vx/vz 三场 + κ/浮力×2),本格式只有 ~4 次(uⁿ 模板、u^{n−1}、c²、写 u^{n+1}),
# 在 memory-bound 网格上理论吞吐是一阶声波的 ~3 倍 —— 与 deepwave scalar
# 同一重量级,消除公式差异后正面对比。代价:仅常密度(ρ 不参与)。
#
# 实现要点:
#   - 两数组 ping-pong:u^{n+1} 写入持有 u^{n−1} 的数组,随后角色轮换;
#   - kernel 全网格写:内区做 leapfrog,外圈(pad 环)携带 uⁿ 原值 —— 与一阶
#     求解器"更新不触及处保持原值"的语义一致,保证 HABC 输入定义良好;
#   - HABC 直接复用确定性两遍版 apply_habc_det_1!(f=u^{n+1}, f_old=uⁿ):
#     二阶格式天然持有上一时刻场,连备份 kernel 都不需要;
#   - 每步 5 次 launch:更新 1 + det-HABC 2 + 注入 1 + 记录 1。
#
# 震源缩放(source_scale):
#   :v2dt2 (默认) —— 子波 × v(源点)²·dt²,deepwave scalar 同款,
#                    振幅随 dt/dh 细化收敛到物理点源;
#   :dt / :none    —— × dt / 裸加(对齐 Fomo 一阶求解器的旧约定)。

using CUDA
using StaticArrays

# ── 更新内核 ───────────────────────────────────────────────────────────────
function _scalar_update_cuda!(
    u_next, u, u_prev, c2,
    d2::SVector{L,Float32}, c0::Float32,
    M::Int32, nx::Int32, nz::Int32
) where {L}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        @inbounds begin
            uc = u[i, j]
            un = uc   # 外圈:携带原值(与一阶求解器语义一致)
            if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
                lap = 2.0f0 * c0 * uc
                for l in 1:L
                    cl = d2[l]
                    lap += cl * (u[i+l, j] + u[i-l, j] + u[i, j+l] + u[i, j-l])
                end
                un = 2.0f0 * uc - u_prev[i, j] + c2[i, j] * lap
            end
            u_next[i, j] = un
        end
    end
    return nothing
end

function scalar_update!(u_next, u, u_prev, c2,
    c0::Float32, d2::SVector{L,Float32}, M32::Int32, nx::Int32, nz::Int32) where {L}
    threads = (32, 8)
    blocks = (cld(Int(nx), 32), cld(Int(nz), 8))
    @cuda threads = threads blocks = blocks _scalar_update_cuda!(
        u_next, u, u_prev, c2, d2, c0, M32, nx, nz)
    return nothing
end

# ── 时间步循环(det HABC,5 launch/步)────────────────────────────────────
"""返回终态 u^{nt}(所在数组的引用)。"""
function _scalar2d_loop!(ua, ub, c2, S, R, B,
    c0, d2, dt, nt, M32, nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_u, scr)
    u_old = ua   # u^{n-1}
    u_cur = ub   # u^{n}
    for it in 1:nt
        # 原地 leapfrog:u_next 与 u_prev 同为 u_old 数组 —— 安全,因为
        # 每线程只读写【本点】的 u_prev/u_next,模板读取全部来自 u_cur。
        scalar_update!(u_old, u_cur, u_old, c2, c0, d2, M32, nx, nz)
        u_next = u_old   # 该数组现持有 u^{n+1}
        apply_habc_det_1!(u_next, u_cur, B.w_tau, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc)
        inject_source!(u_next, S, it, dt)
        record_receivers!(seis_u, u_next, R, it)
        u_old = u_cur
        u_cur = u_next
    end
    return u_cur
end

# ── 公共 API ───────────────────────────────────────────────────────────────
"""
    scalar2d(vp, dh, dt, nt, f0; sx, sz, rx, rz, kwargs...)

二阶标量(常密度)声波正演,HABC,确定性。与 deepwave scalar 同公式,
内存流量约为一阶 `acoustic2d` 的 1/3。

# 关键字
- `sx`, `sz`, `rx`, `rz`: 震源/接收器整数网格坐标(向量)
- `nbc=50`, `fd_order=8`(中心差分,2/4/6/8/10)
- `source_scale=:v2dt2`: 震源缩放,见文件头;`:dt` / `:none` 对齐旧约定
- `wavelet`: 自定义子波(长度 nt),默认 Ricker(f0)
- `v_ref`, `verbose` 同 `acoustic2d`

# 返回(NamedTuple)
- `seis_u`: (n_rec, nt) 标量场记录(压力类)
- `stats`: (kernel_time_s,)
"""
function scalar2d(
    vp::AbstractMatrix{Float32},
    dh::Float32, dt::Float32, nt::Int, f0::Float32;
    sx, sz, rx, rz,
    nbc::Int=50,
    fd_order::Int=8,
    v_ref::Float32=Float32(minimum(x for x in vp if x > 0.0f0)),
    wavelet::Union{Nothing,AbstractVector{<:Real}}=nothing,
    source_scale::Symbol=:v2dt2,
    verbose::Bool=true,
)
    source_scale in (:v2dt2, :dt, :none) ||
        throw(ArgumentError("source_scale 须为 :v2dt2 / :dt / :none,got $source_scale"))
    nx, nz = size(vp)

    _check_geometry(nx, nz, sx, sz, rx, rz)
    _check_numerics_centered(maximum(vp), minimum(x for x in vp if x > 0.0f0),
        dh, dt, f0, fd_order)

    c0, d2 = get_centered_d2(fd_order)
    M = fd_order ÷ 2
    pad = nbc + M

    # 介质:c² = (v·dt/dh)²,边界外推填充
    c2_host = _pad_array(Matrix(vp), pad)
    c2_host .= (c2_host .* Float32(dt / dh)) .^ 2
    c2 = CuArray(c2_host)
    nxp = nx + 2pad
    nzp = nz + 2pad

    bc = init_habc(nx, nz, pad, dt, dh, v_ref)

    # 震源:默认 Ricker;按 source_scale 缩放子波行
    wavelet_data = isnothing(wavelet) ? ricker_wavelet(f0, dt, nt) : Float32.(wavelet)
    length(wavelet_data) == nt ||
        throw(ArgumentError("wavelet 长度 $(length(wavelet_data)) ≠ nt=$nt"))
    wavelet_matrix = repeat(reshape(wavelet_data, 1, nt), length(sx), 1)
    if source_scale === :v2dt2
        for q in 1:length(sx)
            wavelet_matrix[q, :] .*= Float32(vp[sx[q], sz[q]]^2 * dt^2)
        end
    elseif source_scale === :dt
        wavelet_matrix .*= dt
    end
    source = init_source(pad, Int32.(collect(sx)), Int32.(collect(sz)), wavelet_matrix)

    receiver = init_receiver(pad, Int32.(collect(rx)), Int32.(collect(rz)), :p)
    n_rec = length(receiver.rx)
    seis_u = CUDA.zeros(Float32, n_rec, nt)

    ua = CUDA.zeros(Float32, nxp, nzp)
    ub = CUDA.zeros(Float32, nxp, nzp)

    nx_i = Int32(nxp)
    nz_i = Int32(nzp)
    nbc_i = Int32(bc.nbc)
    M32 = Int32(M)
    qx = Float32(bc.qx)
    qz = Float32(bc.qz)
    qt_x = Float32(bc.qt_x)
    qt_z = Float32(bc.qt_z)
    qxt = Float32(bc.qxt)
    scr = CUDA.zeros(Float32, _habc_frame_total(nx_i, nz_i, nbc_i))

    # ── Warmup ──
    verbose && @info "Warming up kernels (scalar)..."
    _scalar2d_loop!(ua, ub, c2, source, receiver, bc,
        c0, d2, dt, 1, M32, nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
        seis_u, scr)
    CUDA.synchronize()
    fill!(ua, 0.0f0)
    fill!(ub, 0.0f0)
    fill!(seis_u, 0.0f0)

    # ── Run ──
    verbose && @info "Starting scalar2d... (nx=$nx, nz=$nz, nt=$nt)"
    elapsed = CUDA.@elapsed begin
        _scalar2d_loop!(ua, ub, c2, source, receiver, bc,
            c0, d2, dt, nt, M32, nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_u, scr)
    end
    verbose && @info "Scalar complete! GPU time: $(round(elapsed, digits=3))s"

    return (seis_u=Array(seis_u),
        stats=(kernel_time_s=Float64(elapsed),))
end
