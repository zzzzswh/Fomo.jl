# Fomo.jl

**基于 GPU 加速的地震波方程正演模拟 Julia 工具包**

<p align="center">
  <img src="wavefield.gif" alt="弹性波波场模拟" width="600">
</p>

Fomo 是一个基于 CUDA 的二维地震波正演模拟工具包，采用交错网格高阶有限差分方法求解声波方程和弹性波方程。

## 特性

- **声波 & 弹性波 2D** — 同时支持声波（压力-速度）和弹性波（应力-速度）方程
- **交错网格有限差分** — 最高支持 10 阶空间精度，有效压制数值频散
- **混合吸收边界条件 (HABC)** — 基于刘洋教授的方法，结合单程波方程与指数衰减吸收层
- **Vacuum 自由表面** — 通过 vacuum 方法（地表以上速度和密度置零）模拟真实自由地表条件
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

### 示例输出：炮记录

<p align="center">
  <img src="vz.png" alt="炮记录（Vz 分量）" width="600">
</p>

## API 参考

### 正演模拟

| 函数 | 说明 |
|---|---|
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)` | 声波方程正演模拟 |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)` | 弹性波方程正演模拟 |

**通用关键字参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `sx, sz` | — | 震源位置（网格索引） |
| `rx, rz` | — | 检波器位置（网格索引） |
| `nbc` | `50` | 吸收边界层网格点数 |
| `fd_order` | `8` | 有限差分阶数（2, 4, 6, 8 或 10） |
| `snap_interval` | `0` | 波场快照间隔（0 = 不保存快照） |

**返回值：** `(vx_record, vz_record, snapshots)`

### 工具函数

| 函数 | 说明 |
|---|---|
| `ricker_wavelet(f0, dt, nt)` | 生成 Ricker 子波 |
| `trace_norm(data; dims)` | 逐道归一化 |
| `plot_shot(data, filename)` | 保存炮记录图 |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)` | 导出波场动画视频 |

## 方法

Fomo 采用交错网格速度-应力格式（Virieux, 1986）求解波动方程，主要数值方法包括：

- **空间离散** — 标准交错网格，最高支持 10 阶有限差分算子
- **时间离散** — 二阶蛙跳格式
- **边界条件** — 混合吸收边界条件（HABC），结合单程波方程与指数衰减吸收层（刘洋教授）
- **自由表面** — Vacuum 方法：将自由表面以上的速度和密度置零，自然满足零应力条件

## 环境要求

- Julia ≥ 1.10
- 支持 CUDA 的 NVIDIA GPU
- CUDA.jl ≥ 5.0

## 许可证

MIT

## 致谢

- 混合吸收边界条件：基于刘洋教授的工作
- 交错网格有限差分格式：Virieux (1986)