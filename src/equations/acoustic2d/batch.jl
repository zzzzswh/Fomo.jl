# src/equations/acoustic2d/batch.jl
#
# 声波多炮批处理正演:场为 (nx, nz, n_shots),介质/差分系数/HABC 权重
# 各炮共享,一次 kernel launch 同时推进全部炮(blockIdx().z = 炮号)。
#
# 各炮之间零数据交互 → 每炮结果与单炮 det 生产路径【逐位相等】
# (verify_batch.jl 的判据)。适用场景:FWI/RTM 多炮吞吐;小中网格
# 单炮吃不满 GPU 时,批处理近似把吞吐乘以炮数。

using CUDA
using StaticArrays

# ── 批处理波场 ─────────────────────────────────────────────────────────────
mutable struct BatchedAcousticWavefield{T}
    vx::T
    vz::T
    p::T
    vx_old::T
    vz_old::T
    p_old::T
end

function BatchedAcousticWavefield(nx_in::Int, nz_in::Int, pad::Int, ns::Int)
    nx = nx_in + 2pad
    nz = nz_in + 2pad
    z() = CUDA.zeros(Float32, nx, nz, ns)
    return BatchedAcousticWavefield(z(), z(), z(), z(), z(), z())
end

function reset!(W::BatchedAcousticWavefield)
    for f in fieldnames(BatchedAcousticWavefield)
        fill!(getfield(W, f), 0.0f0)
    end
    return W
end

# ── 批量融合内核(数学与 fused_kernels.jl 逐字一致,多一个炮下标)──────────
function _bfused_vel_acoustic_cuda!(
    vx, vz, p, vx_old, vz_old,
    bx, bz,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    s = blockIdx().z

    if i <= nx && j <= nz
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds begin
                vx_old[i, j, s] = vx[i, j, s]
                vz_old[i, j, s] = vz[i, j, s]
            end
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dpdx = 0.0f0
            dpdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dpdx += c * (p[i+l-1, j, s] - p[i-l, j, s])
                dpdz += c * (p[i, j+l, s] - p[i, j-l+1, s])
            end
            @inbounds begin
                vx[i, j, s] += bx[i, j] * dtx * dpdx
                vz[i, j, s] += bz[i, j] * dtz * dpdz
            end
        end
    end
    return nothing
end

function bfused_update_velocity_acoustic!(W::BatchedAcousticWavefield,
    M_med::AcousticMedium, a_static::SVector{N,Float32}, dt, nbc::Int32, ns::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)
    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8), ns)
    @cuda threads = threads blocks = blocks _bfused_vel_acoustic_cuda!(
        W.vx, W.vz, W.p, W.vx_old, W.vz_old,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

function _bfused_prs_acoustic_cuda!(
    p, vx, vz, p_old,
    kappa,
    a::SVector{N,Float32},
    dtx::Float32, dtz::Float32, M::Int32, nbc::Int32,
    nx::Int32, nz::Int32
) where {N}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    s = blockIdx().z

    if i <= nx && j <= nz
        if (i <= nbc + Int32(2)) | (i >= nx - nbc - Int32(1)) |
           (j <= nbc + Int32(2)) | (j >= nz - nbc - Int32(1))
            @inbounds p_old[i, j, s] = p[i, j, s]
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dvxdx = 0.0f0
            dvzdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dvxdx += c * (vx[i+l, j, s] - vx[i-l+1, j, s])
                dvzdz += c * (vz[i, j+l-1, s] - vz[i, j-l, s])
            end
            @inbounds p[i, j, s] += kappa[i, j] * (dvxdx * dtx + dvzdz * dtz)
        end
    end
    return nothing
end

function bfused_update_pressure_acoustic!(W::BatchedAcousticWavefield,
    M_med::AcousticMedium, a_static::SVector{N,Float32}, dt, nbc::Int32, ns::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)
    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8), ns)
    @cuda threads = threads blocks = blocks _bfused_prs_acoustic_cuda!(
        W.p, W.vx, W.vz, W.p_old,
        M_med.kappa,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ── 批处理时间步循环(det HABC,与单炮 det 融合循环同结构)────────────────
function _acoustic2d_loop_batch!(W, M, S, R, B,
    a_static, dt, nt,
    nbc, nx, nz, qx, qz, qt_x, qt_z, qxt,
    seis_p, seis_vx, seis_vz, ns)

    scr = CUDA.zeros(Float32, 2 * _habc_frame_total(nx, nz, nbc) * ns)

    for it in 1:nt
        bfused_update_velocity_acoustic!(W, M, a_static, dt, nbc, ns)
        apply_bhabc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, ns)
        bfused_update_pressure_acoustic!(W, M, a_static, dt, nbc, ns)
        apply_bhabc_det_1!(W.p, W.p_old, B.w_tau, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, ns)
        binject_source!(W.p, S, it)
        brecord_receivers3!(seis_p, seis_vx, seis_vz, W.p, W.vx, W.vz, R, it, ns)
    end
    return nothing
end

# ── 公共 API ───────────────────────────────────────────────────────────────
"""
    acoustic2d_batch(vp, rho, dh, dt, nt, f0; sx, sz, rx, rz, kwargs...)

声波多炮批处理正演(HABC,确定性)。一次 GPU 调用同时推进 n_shots 炮,
介质与接收器排列各炮共享;每炮结果与单炮 `acoustic2d` 逐位相同。

# 炮几何
- `sx`, `sz`: (n_shots × n_src_per_shot) 整数矩阵;传向量视为每炮 1 个震源
  (长度 = n_shots)。**注意与单炮 API 语义不同**:单炮 API 的向量是
  同一炮内的多个震源。
- `rx`, `rz`: 接收器排列,各炮共享(固定排列)。

# 关键字
- `nbc`, `fd_order`, `v_ref`, `wavelet`, `scale_source_by_dt`, `verbose`
  与 `acoustic2d` 同义;边界固定为 HABC;暂不支持快照。

# 返回(NamedTuple)
- `seis_p`, `seis_vx`, `seis_vz`: (n_rec, nt, n_shots)
- `stats`: (kernel_time_s, n_shots)
"""
function acoustic2d_batch(
    vp::AbstractMatrix{Float32}, rho::AbstractMatrix{Float32},
    dh::Float32, dt::Float32, nt::Int, f0::Float32;
    sx, sz, rx, rz,
    nbc::Int=50,
    fd_order::Int=8,
    v_ref::Float32=Float32(minimum(x for x in vp if x > 0.0f0)),
    wavelet::Union{Nothing,AbstractVector{<:Real}}=nothing,
    scale_source_by_dt::Bool=false,
    verbose::Bool=true,
)
    nx, nz = size(vp)

    # ── 炮几何整理:(n_shots, n_src_per_shot) ──
    sxm = sx isa AbstractVector ? reshape(collect(Int, sx), :, 1) : collect(Int, sx)
    szm = sz isa AbstractVector ? reshape(collect(Int, sz), :, 1) : collect(Int, sz)
    size(sxm) == size(szm) ||
        throw(ArgumentError("sx 与 sz 尺寸不一致: $(size(sxm)) vs $(size(szm))"))
    ns, n_src = size(sxm)
    ns >= 1 || throw(ArgumentError("至少需要 1 炮"))

    # ── 校验(逐炮几何 + 数值稳定性)──
    for s in 1:ns
        _check_geometry(nx, nz, vec(sxm[s, :]), vec(szm[s, :]), rx, rz)
    end
    _check_numerics(maximum(vp), minimum(x for x in vp if x > 0.0f0),
        dh, dt, f0, fd_order)

    # ── 初始化(与单炮完全一致的介质/边界/系数)──
    a_static = get_fd_coefficients(fd_order)
    medium = init_acoustic_medium(vp, rho, dh, nbc, fd_order)
    bc = init_habc(nx, nz, medium.pad, dt, dh, v_ref)
    W = BatchedAcousticWavefield(nx, nz, medium.pad, ns)

    wavelet_data = isnothing(wavelet) ? ricker_wavelet(f0, dt, nt) : Float32.(wavelet)
    length(wavelet_data) == nt ||
        throw(ArgumentError("wavelet 长度 $(length(wavelet_data)) ≠ nt=$nt"))
    wavelet_matrix = repeat(reshape(wavelet_data, 1, nt), n_src, 1)
    scale_source_by_dt && (wavelet_matrix .*= dt)
    # 设备布局 (n_src_per_shot, n_shots)
    source = init_batched_source(medium.pad, permutedims(sxm), permutedims(szm), wavelet_matrix)

    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :p)
    n_rec = length(receiver.rx)
    seis_p = CUDA.zeros(Float32, n_rec, nt, ns)
    seis_vx = CUDA.zeros(Float32, n_rec, nt, ns)
    seis_vz = CUDA.zeros(Float32, n_rec, nt, ns)

    nx_i = Int32(medium.nx)
    nz_i = Int32(medium.nz)
    nbc_i = Int32(bc.nbc)
    qx = Float32(bc.qx)
    qz = Float32(bc.qz)
    qt_x = Float32(bc.qt_x)
    qt_z = Float32(bc.qt_z)
    qxt = Float32(bc.qxt)

    # ── Warmup ──
    verbose && @info "Warming up kernels (batch)..."
    _acoustic2d_loop_batch!(W, medium, source, receiver, bc,
        a_static, dt, 1, nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
        seis_p, seis_vx, seis_vz, ns)
    CUDA.synchronize()
    reset!(W)
    fill!(seis_p, 0.0f0)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    verbose && @info "Starting acoustic2d_batch... (nx=$nx, nz=$nz, nt=$nt, n_shots=$ns)"
    elapsed = CUDA.@elapsed begin
        _acoustic2d_loop_batch!(W, medium, source, receiver, bc,
            a_static, dt, nt, nbc_i, nx_i, nz_i, qx, qz, qt_x, qt_z, qxt,
            seis_p, seis_vx, seis_vz, ns)
    end
    verbose && @info "Batch complete! GPU time: $(round(elapsed, digits=3))s ($(ns) shots)"

    return (seis_p=Array(seis_p),
        seis_vx=Array(seis_vx),
        seis_vz=Array(seis_vz),
        stats=(kernel_time_s=Float64(elapsed), n_shots=ns))
end
