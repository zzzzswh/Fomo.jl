# Fomo.jl

**基于 GPU 加速的 2D/3D 声波与弹性波波动方程模拟器 Julia 工具包**

> 🌐 **中文** | [English](README.md)

<p align="center">
  <img src="wavefield.gif" alt="弹性波波场模拟" width="600">
</p>

Fomo 是一个基于 CUDA 的二维波动方程模拟器，采用交错网格高阶有限差分方法求解声波方程和弹性波方程。虽然最初为地震波模拟而开发，但同样适用于任何涉及声波或弹性波传播的场景——地震正演，超声检测、无损探伤、医学成像、水声学等。

## 特性

- **声波 & 弹性波 2D/3D** — 同时支持声波（压力-速度）和弹性波（应力-速度）方程
- **🆕 耦合 P-S 势场 2D** — 基于 Li et al. (2018) 的新型求解器，直接传播天然分离的 P 波和 S 波势场（详见[下方说明](#-耦合-p-s-势场求解器)）
- **交错网格有限差分** — 最高支持 10 阶空间精度，有效压制数值频散
- **混合吸收边界条件 (HABC)** — 基于刘洋教授的方法，结合单程波方程与指数衰减吸收层。相比传统 PML，具有更优的吸收效果与更低的计算开销
- **Vacuum 自由表面** — 通过 vacuum 方法（地表以上速度和密度置零）可以在介质里随意设置反射界面
- **GPU 加速** — 基于 [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)，在 NVIDIA GPU 上实现高性能计算
- **内置可视化** — 开箱即用的炮记录绘图与波场动画导出

## 安装

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

## 快速开始

### 声波 2D

```julia
using CUDA
using Fomo

# 网格参数
nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# 速度模型
vp  = fill(2500.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# Vacuum 自由表面（顶行置零）
vp[:, 1]  .= 0.0f0
rho[:, 1] .= 0.0f0

# 震源与检波器
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# 正演模拟
vx, vz, snaps = acoustic2d(vp, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# 可视化
plot_shot(trace_norm(vz, dims=2), "acoustic_vz.png")
plot_wavefield_video(snaps, 50, "acoustic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

### 弹性波 2D

```julia
using CUDA
using Fomo

nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# 弹性模型：vp, vs, rho
vp  = fill(2500.0f0, nx, nz)
vs  = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)

# Vacuum 自由表面
vp[:, 1]  .= 0.0f0
vs[:, 1]  .= 0.0f0
rho[:, 1] .= 0.0f0

# 震源与检波器
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# 正演模拟
vx, vz, snaps = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# 可视化
plot_shot(trace_norm(vz, dims=2), "elastic_vz.png")
plot_wavefield_video(snaps, 50, "elastic_wavefield.mp4",
    fps=10, adaptive_clims=true)
```

### 🆕 耦合 P-S 势场 2D

```julia
using CUDA
using Fomo

nx, nz = 400, 300
dh = 10.0f0
dt = 0.001f0
nt = 2000
f0 = 15.0f0

# 只需 vp 和 vs —— 不需要密度（假设 ρ=1）
vp = fill(2500.0f0, nx, nz)
vs = fill(1200.0f0, nx, nz)
vp[:, 150:end] .= 3500.0f0
vs[:, 150:end] .= 1800.0f0

# 震源与检波器
sx = [nx ÷ 2];       sz = [10]
rx = collect(1:2:nx); rz = fill(10, length(rx))

# 正演模拟 —— 返回天然分离的 P 和 S 势场
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz,
    nbc=100, fd_order=8, snap_interval=50)

# seis_P: P 波势场记录（包含 PP 反射）
# seis_S: S 波势场记录（包含 PS 转换波 —— 仅在 Vs 不连续处产生！）
```

### 示例输出：炮记录

<p align="center">
  <img src="vz.png" alt="炮记录（Vz 分量）" width="600">
</p>

## 🆕 耦合 P-S 势场求解器

### 理论基础

基于 **Li et al. (2018)** *"Elastic reverse time migration using acoustic propagators"* (Geophysics, Vol. 83, No. 5, S399–S408)，在常密度各向同性弹性介质中推导了 P 波和 S 波势场满足的耦合二阶方程：

$$\ddot{P} = P\nabla^2\alpha + 2\nabla\alpha\cdot\nabla P - 2P\nabla^2\beta - 2\nabla\beta\cdot(\nabla\times\mathbf{S}) + \alpha\nabla^2 P + \nabla\cdot\mathbf{f}$$

$$\ddot{\mathbf{S}} = \nabla\beta\cdot\nabla\mathbf{S} - (\nabla\beta)\times(\nabla\times\mathbf{S}) + 2(\nabla\beta)\times(\nabla P) + \beta\nabla^2\mathbf{S} + \nabla\times\mathbf{f}$$

其中 α = V²_P、β = V²_S 为 P、S 波速度的平方，P = ∇·**u** 为 P 波势场（标量），**S** = ∇×**u** 为 S 波势场（矢量；2D 中仅 y 分量 S_y 非零）。

**方程揭示的关键物理：**

1. **模式转换仅在 V_S 不连续处发生** — 若 ∇β = 0，P 波与 S 波完全解耦
2. **V_P 不连续面对 S 波透明** — 纯 V_P 扰动不产生 PS 转换波
3. P 波与 S 波势场**天然分离**，无需 Helmholtz 分解

### 实现：速度-位置分裂与二阶等效 HABC

直接对二阶方程做 leapfrog 时间推进，每步只能施加一次 HABC，等效于一阶 Higdon ABC，斜入射吸收不足。

**核心设计：将 leapfrog 改写为速度-位置分裂格式**，使其与 velocity-stress 交错时间方案结构同构：

| Velocity-Stress (elastic2d) | Velocity-Position (coupled2d) |
|---|---|
| v_x, v_z（质点速度） | dP/dt, dS/dt（势场变化率） |
| τ_xx, τ_zz, τ_xz（应力） | P, S（势场） |
| update_velocity → **HABC** | update_velocity → **HABC** |
| update_stress → **HABC** | update_position → **HABC** |

每步**两次 HABC**，等效于**二阶 Higdon ABC**，吸收效果与传统弹性波求解器一致。

### 对比传统弹性波求解器

| | elastic2d | coupled2d |
|---|---|---|
| 波场数组 (2D) | 10 个 (vx, vz, τ_xx, τ_zz, τ_xz + 备份) | 8 个 (P, S, dP/dt, dS/dt + 备份) |
| 内存占用 | 5n² | 4n² **（减少 20%）** |
| P/S 分离 | 需要 Helmholtz 分解 | **天然内置** |
| 模式转换 | 隐式 | **显式（仅在 ∇β ≠ 0 处）** |
| 成像条件 | 需要额外的相位校正 | **与物理扰动一致** |
| 密度要求 | 任意 | 常密度 (ρ = const) |

## API 参考

### 正演模拟

| 函数 | 说明 |
|---|---|
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)` | 声波方程正演模拟 |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)` | 弹性波方程正演模拟 |
| `coupled2d(vp, vs, dh, dt, nt, f0; ...)` | 🆕 耦合 P-S 势场正演模拟 |

**通用关键字参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `sx, sz` | — | 震源位置（网格索引） |
| `rx, rz` | — | 检波器位置（网格索引） |
| `nbc` | `50` | 吸收边界层网格点数 |
| `fd_order` | `8` | 有限差分阶数（2, 4, 6, 8 或 10） |
| `snap_interval` | `0` | 波场快照间隔（0 = 不保存快照） |

**`coupled2d` 额外关键字参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `v_ref_p` | `min(vp)` | P 场 HABC 参考速度 |
| `v_ref_s` | `min(vs)` | S 场 HABC 参考速度 |

**返回值：**
- `acoustic2d` / `elastic2d` → `(vx_record, vz_record, snapshots)`
- `coupled2d` → `(P_record, S_record, P_snapshots, S_snapshots)`

### 工具函数

| 函数 | 说明 |
|---|---|
| `ricker_wavelet(f0, dt, nt)` | 生成 Ricker 子波 |
| `trace_norm(data; dims)` | 逐道归一化 |
| `plot_shot(data, filename)` | 保存炮记录图 |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)` | 导出波场动画视频 |

## 方法

Fomo 实现了三种波动方程求解器：

**声波 & 弹性波（velocity-stress 格式）** — 标准交错网格方案（Virieux, 1986），最高支持 10 阶 FD 算子，二阶蛙跳时间推进，HABC 吸收边界（刘洋教授）。

**耦合 P-S 势场** — 基于 Li et al. (2018) 推导的 P-S 波耦合二阶势场方程。将二阶时间方程分解为速度-位置分裂格式，使其与 velocity-stress 方案结构同构，从而实现二阶等效 HABC 吸收。该求解器使用正则网格上的中心差分算子（非交错）。

## 环境要求

- Julia ≥ 1.10
- 支持 CUDA 的 NVIDIA GPU
- CUDA.jl ≥ 5.0

## 许可证

MIT

## 参考文献

- **Li, Y. E., Du, Y., Yang, J., Cheng, A., & Fang, X. (2018).** Elastic reverse time migration using acoustic propagators. *Geophysics*, 83(5), S399–S408. doi: [10.1190/GEO2017-0687.1](https://doi.org/10.1190/GEO2017-0687.1)
- **Virieux, J. (1986).** P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.

## 致谢

- 混合吸收边界条件：基于刘洋教授的工作
- 交错网格有限差分格式：Virieux (1986)
- 耦合 P-S 势场方程：Li et al. (2018)