# src/equations/elastic2d/batch.jl
#
# 弹性波多炮批处理正演。结构与声波批处理完全平行:
# 场 (nx, nz, n_shots),介质/权重共享,各炮零交互 → 每炮与单炮逐位相等。

using CUDA
using StaticArrays

# ── 批处理波场 ─────────────────────────────────────────────────────────────
mutable struct BatchedWavefield{T}
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

function BatchedWavefield(nx_in::Int, nz_in::Int, pad::Int, ns::Int)
    nx = nx_in + 2pad
    nz = nz_in + 2pad
    z() = CUDA.zeros(Float32, nx, nz, ns)
    return BatchedWavefield(z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end

function reset!(W::BatchedWavefield)
    for f in fieldnames(BatchedWavefield)
        fill!(getfield(W, f), 0.0f0)
    end
    return W
end

# ── 批量融合内核(数学与 fused_kernels.jl 逐字一致)────────────────────────
function _bfused_vel_elastic_cuda!(
    vx, vz, txx, tzz, txz, vx_old, vz_old,
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
            dtxxdx = 0.0f0
            dtxzdz = 0.0f0
            dtxzdx = 0.0f0
            dtzzdz = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dtxxdx += c * (txx[i+l-1, j, s] - txx[i-l, j, s])
                dtxzdz += c * (txz[i, j+l-1, s] - txz[i, j-l, s])
                dtxzdx += c * (txz[i+l, j, s] - txz[i-l+1, j, s])
                dtzzdz += c * (tzz[i, j+l, s] - tzz[i, j-l+1, s])
            end
            @inbounds begin
                vx[i, j, s] += bx[i, j] * (dtx * dtxxdx + dtz * dtxzdz)
                vz[i, j, s] += bz[i, j] * (dtx * dtxzdx + dtz * dtzzdz)
            end
        end
    end
    return nothing
end

function bfused_update_velocity_elastic!(W::BatchedWavefield, M_med,
    a_static::SVector{N,Float32}, dt, nbc::Int32, ns::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)
    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8), ns)
    @cuda threads = threads blocks = blocks _bfused_vel_elastic_cuda!(
        W.vx, W.vz, W.txx, W.tzz, W.txz, W.vx_old, W.vz_old,
        M_med.buoy_vx, M_med.buoy_vz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

function _bfused_str_elastic_cuda!(
    txx, tzz, txz, vx, vz, txx_old, tzz_old, txz_old,
    lam, lam_2mu, mu_txz,
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
                txx_old[i, j, s] = txx[i, j, s]
                tzz_old[i, j, s] = tzz[i, j, s]
                txz_old[i, j, s] = txz[i, j, s]
            end
        end

        if (i > M) & (i <= nx - M) & (j > M) & (j <= nz - M)
            dvxdx = 0.0f0
            dvzdz = 0.0f0
            dvxdz = 0.0f0
            dvzdx = 0.0f0
            @inbounds for l in 1:N
                c = a[l]
                dvxdx += c * (vx[i+l, j, s] - vx[i-l+1, j, s])
                dvzdz += c * (vz[i, j+l-1, s] - vz[i, j-l, s])
                dvxdz += c * (vx[i, j+l, s] - vx[i, j-l+1, s])
                dvzdx += c * (vz[i+l-1, j, s] - vz[i-l, j, s])
            end
            @inbounds begin
                l_val = lam[i, j]
                l2m_val = lam_2mu[i, j]
                txx[i, j, s] += l2m_val * dvxdx * dtx + l_val * dvzdz * dtz
                tzz[i, j, s] += l_val * dvxdx * dtx + l2m_val * dvzdz * dtz
                txz[i, j, s] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
            end
        end
    end
    return nothing
end

function bfused_update_stress_elastic!(W::BatchedWavefield, M_med,
    a_static::SVector{N,Float32}, dt, nbc::Int32, ns::Int) where {N}
    dtx = Float32(dt / M_med.dh)
    dtz = Float32(dt / M_med.dh)
    nx = Int32(M_med.nx)
    nz = Int32(M_med.nz)
    M32 = Int32(M_med.M)
    threads = (32, 8)
    blocks = (cld(M_med.nx, 32), cld(M_med.nz, 8), ns)
    @cuda threads = threads blocks = blocks _bfused_str_elastic_cuda!(
        W.txx, W.tzz, W.txz, W.vx, W.vz, W.txx_old, W.tzz_old, W.txz_old,
        M_med.lam, M_med.lam_2mu, M_med.mu_txz,
        a_static, dtx, dtz, M32, nbc, nx, nz)
    return nothing
end

# ── 批处理时间步循环 ───────────────────────────────────────────────────────
function _elastic2d_loop_batch!(W, M, S, R, B,
    a_static, dt, nt,
    seis_vx, seis_vz, ns)

    nx = Int32(M.nx)
    nz = Int32(M.nz)
    nbc = Int32(B.nbc)
    qx = Float32(B.qx)
    qz = Float32(B.qz)
    qt_x = Float32(B.qt_x)
    qt_z = Float32(B.qt_z)
    qxt = Float32(B.qxt)

    scr = CUDA.zeros(Float32, 3 * _habc_frame_total(nx, nz, nbc) * ns)

    for it in 1:nt
        bfused_update_velocity_elastic!(W, M, a_static, dt, nbc, ns)
        apply_bhabc_det_2!(W.vx, W.vx_old, B.w_vx, W.vz, W.vz_old, B.w_vz, scr,
            qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, ns)
        bfused_update_stress_elastic!(W, M, a_static, dt, nbc, ns)
        apply_bhabc_det_3!(W.txx, W.txx_old, W.tzz, W.tzz_old, W.txz, W.txz_old,
            B.w_tau, scr, qx, qz, qt_x, qt_z, qxt, nx, nz, nbc, ns)
        binject_source_pair!(W.txx, W.tzz, S, it)
        brecord_receivers2!(seis_vx, seis_vz, W.vx, W.vz, R, it, ns)
    end
    return nothing
end

# ── 公共 API ───────────────────────────────────────────────────────────────
"""
    elastic2d_batch(vp, vs, rho, dh, dt, nt, f0; sx, sz, rx, rz, kwargs...)

弹性波多炮批处理正演(HABC,确定性)。语义与 `acoustic2d_batch` 平行:
`sx`/`sz` 为 (n_shots × n_src_per_shot) 矩阵(向量视为每炮 1 个震源),
接收器排列各炮共享;每炮结果与单炮 `elastic2d` 逐位相同。

# 返回(NamedTuple)
- `seis_vx`, `seis_vz`: (n_rec, nt, n_shots)
- `stats`: (kernel_time_s, n_shots)
"""
function elastic2d_batch(
    vp::AbstractMatrix{Float32}, vs::AbstractMatrix{Float32},
    rho::AbstractMatrix{Float32},
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

    sxm = sx isa AbstractVector ? reshape(collect(Int, sx), :, 1) : collect(Int, sx)
    szm = sz isa AbstractVector ? reshape(collect(Int, sz), :, 1) : collect(Int, sz)
    size(sxm) == size(szm) ||
        throw(ArgumentError("sx 与 sz 尺寸不一致: $(size(sxm)) vs $(size(szm))"))
    ns, n_src = size(sxm)
    ns >= 1 || throw(ArgumentError("至少需要 1 炮"))

    for s in 1:ns
        _check_geometry(nx, nz, vec(sxm[s, :]), vec(szm[s, :]), rx, rz)
    end
    _check_numerics(maximum(vp), minimum(x for x in vp if x > 0.0f0),
        dh, dt, f0, fd_order)

    a_static = get_fd_coefficients(fd_order)
    medium = init_medium(vp, vs, rho, dh, nbc, fd_order)
    bc = init_habc(nx, nz, medium.pad, dt, dh, v_ref)
    W = BatchedWavefield(nx, nz, medium.pad, ns)

    wavelet_data = isnothing(wavelet) ? ricker_wavelet(f0, dt, nt) : Float32.(wavelet)
    length(wavelet_data) == nt ||
        throw(ArgumentError("wavelet 长度 $(length(wavelet_data)) ≠ nt=$nt"))
    wavelet_matrix = repeat(reshape(wavelet_data, 1, nt), n_src, 1)
    scale_source_by_dt && (wavelet_matrix .*= dt)
    source = init_batched_source(medium.pad, permutedims(sxm), permutedims(szm), wavelet_matrix)

    receiver = init_receiver(medium.pad, Int32.(collect(rx)), Int32.(collect(rz)), :vz)
    n_rec = length(receiver.rx)
    seis_vx = CUDA.zeros(Float32, n_rec, nt, ns)
    seis_vz = CUDA.zeros(Float32, n_rec, nt, ns)

    # ── Warmup ──
    verbose && @info "Warming up kernels (batch)..."
    _elastic2d_loop_batch!(W, medium, source, receiver, bc,
        a_static, dt, 1, seis_vx, seis_vz, ns)
    CUDA.synchronize()
    reset!(W)
    fill!(seis_vx, 0.0f0)
    fill!(seis_vz, 0.0f0)

    # ── Run ──
    verbose && @info "Starting elastic2d_batch... (nx=$nx, nz=$nz, nt=$nt, n_shots=$ns)"
    elapsed = CUDA.@elapsed begin
        _elastic2d_loop_batch!(W, medium, source, receiver, bc,
            a_static, dt, nt, seis_vx, seis_vz, ns)
    end
    verbose && @info "Batch complete! GPU time: $(round(elapsed, digits=3))s ($(ns) shots)"

    return (seis_vx=Array(seis_vx),
        seis_vz=Array(seis_vz),
        stats=(kernel_time_s=Float64(elapsed), n_shots=ns))
end
