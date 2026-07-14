# test/verify_patch.jl — v0.2.0 补丁验证脚本 v4（需 GPU）
#
# 用法: julia --project=. test/verify_patch.jl
#
# ── 阈值标定依据（CPU float64 逐算子参考实现）────────────────────────────
# sponge 镜像实验:
#   修复后+原样sponge: p=1.887e-3（与GPU四位一致）; 修复前bug: p=3.9e-3;
#   因果屏蔽几何: 正确代码=精确0.0, 带bug代码≥1.2e-3 → 阈值1e-5两侧余量>100×
# HABC 实验（nbc=20, v_ref=min(vp), 大域L=270无边界参考, nt=1.5s）:
#   新代码(zone=pad-1): 残差1.245e-2; 旧代码(zone=nbc): 1.336e-2;
#   sponge同层数: 2.87e-2 → HABC吸收质量阈值取 2.5e-2（2×余量）
#   屏蔽几何+HABC 镜像=精确0.0（race噪声产生于边界条带，同样被因果排除）
# ──────────────────────────────────────────────────────────────────────────

using Fomo, CUDA, Test, LinearAlgebra

function make_model(nx::Int, nz::Int)
    xc = (nx + 1) / 2
    vp = fill(2000.0f0, nx, nz)
    for i in 1:nx
        zint = 100 + round(Int, 20 * cos(2π * (i - xc) / nx))
        vp[i, zint:end] .= 3000.0f0
    end
    xg = ((1:nx) .- xc) ./ 30
    zg = ((1:nz)' .- 60) ./ 15
    vp .+= 400.0f0 .* Float32.(exp.(-(xg .^ 2 .+ zg .^ 2)))
    vs = vp ./ 1.73f0
    rho = Float32.(310.0 .* Float64.(vp) .^ 0.25)   # Gardner
    return vp, vs, rho
end

# 边缘延拓：大域参考模型（波在记录窗内碰不到大域边界 → 无反射真值）
function edge_extend(m::Matrix{Float32}, L::Int)
    ix = clamp.((1:size(m, 1)+2L) .- L, 1, size(m, 1))
    iz = clamp.((1:size(m, 2)+2L) .- L, 1, size(m, 2))
    return m[ix, iz]
end

dh, dt, f0 = 10.0f0, 1.0f-3, 10.0f0

# p/vz 位于整数 x 节点：直接翻转配对；vx 位于 (i-1/2,j)：镜像=索引603-r，取2..end行翻转
rel_sym(d) = norm(d .- reverse(d, dims=1)) / norm(d)
rel_asym_vx(d) = norm(d[2:end, :] .+ reverse(d[2:end, :], dims=1)) / norm(d[2:end, :])

@testset "Fomo v0.2.0 verify (v4)" begin

    # ══ 屏蔽几何：nx=601，检波器只在中部，1.2s < 边界污染最早到达 ~1.32s ══
    nxs, nzs, nts = 601, 201, 1200
    vp, vs, rho = make_model(nxs, nzs)
    sx, sz = [Int((nxs + 1) ÷ 2)], [41]
    rx = collect(150:452)
    rz = fill(41, length(rx))   # 150+452=602: p/vz 配对 r↔602-r

    @testset "模型自身对称性（前置条件）" begin
        @test vp == reverse(vp, dims=1)
        @test rho == reverse(rho, dims=1)
    end

    for bnd in (:sponge, :habc)
        @testset "x 镜像对称·因果屏蔽 [$bnd]" begin
            kw = (; sx, sz, rx, rz, nbc=30, fd_order=4, boundary=bnd, verbose=false)
            res = acoustic2d(vp, rho, dh, dt, nts, f0; kw...)
            @info "[$bnd] acoustic  p: $(rel_sym(res.seis_p))  vz: $(rel_sym(res.seis_vz))  vx: $(rel_asym_vx(res.seis_vx))"
            @test rel_sym(res.seis_p) < 1e-5
            @test rel_sym(res.seis_vz) < 1e-5
            @test rel_asym_vx(res.seis_vx) < 1e-5

            rel = elastic2d(vp, vs, rho, dh, dt, nts, f0; kw...)
            @info "[$bnd] elastic   vz: $(rel_sym(rel.seis_vz))  vx: $(rel_asym_vx(rel.seis_vx))"
            @test rel_sym(rel.seis_vz) < 1e-5
            @test rel_asym_vx(rel.seis_vx) < 1e-5
        end
    end

    # ══ HABC 吸收质量：小域 HABC vs 大域无边界参考 ══
    @testset "HABC 吸收质量（大域参考, nbc=20）" begin
        nxr, nzr, ntr, L = 301, 201, 1500, 270
        # 因果保证: 2L·dh/vmax = 2·270·10/3400 ≈ 1.59s > ntr·dt = 1.5s
        vp2, _, rho2 = make_model(nxr, nzr)
        rx2 = collect(1:nxr)
        h = acoustic2d(vp2, rho2, dh, dt, ntr, f0;
            sx=[151], sz=[41], rx=rx2, rz=fill(41, nxr),
            nbc=20, fd_order=4, boundary=:habc, verbose=false)
        vpB, rhoB = edge_extend(vp2, L), edge_extend(rho2, L)
        ref = acoustic2d(vpB, rhoB, dh, dt, ntr, f0;
            sx=[151 + L], sz=[41 + L], rx=rx2 .+ L, rz=fill(41 + L, nxr),
            nbc=20, fd_order=4, boundary=:sponge, verbose=false)
        @test !any(isnan, h.seis_p)
        resid = norm(h.seis_p .- ref.seis_p) / norm(ref.seis_p)
        @info "HABC 边界残差（vs 无边界真值）: $resid   [CPU float64 标定: 1.245e-2; 旧作用区: 1.336e-2; sponge: 2.87e-2]"
        @test resid < 2.5e-2
    end

    # ══ 功能测试（sponge 路径，逐位确定性）══
    kw0 = (; sx, sz, rx, rz, nbc=30, fd_order=4, boundary=:sponge, verbose=false)

    @testset "多源线性（wavelet repeat 修复）" begin
        a = acoustic2d(vp, rho, dh, dt, nts, f0; kw0..., sx=[200], sz=[41])
        b = acoustic2d(vp, rho, dh, dt, nts, f0; kw0..., sx=[400], sz=[41])
        ab = acoustic2d(vp, rho, dh, dt, nts, f0; kw0..., sx=[200, 400], sz=[41, 41])
        r = norm(ab.seis_p .- (a.seis_p .+ b.seis_p)) / norm(ab.seis_p)
        @info "多源线性残差: $r"
        @test r < 1e-5
    end

    @testset "CFL / 几何越界报错" begin
        @test_throws ArgumentError acoustic2d(vp, rho, dh, 9.0f-3, nts, f0; kw0...)
        @test_throws ArgumentError elastic2d(vp, vs, rho, dh, 9.0f-3, nts, f0; kw0...)
        @test_throws ArgumentError acoustic2d(vp, rho, dh, dt, nts, f0; kw0..., sx=[nxs + 5], sz=[41])
    end

    @testset "自定义子波 = 默认 Ricker（逐位）" begin
        w = ricker_wavelet(f0, dt, nts)
        r0 = acoustic2d(vp, rho, dh, dt, nts, f0; kw0...)
        r1 = acoustic2d(vp, rho, dh, dt, nts, f0; kw0..., wavelet=w)
        @test r1.seis_p == r0.seis_p
    end

end

println("\n完成。")