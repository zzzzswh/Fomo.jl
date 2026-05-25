# Fomo.jl

**可读、可改、可扩展的 Julia GPU 波动方程求解器 —— 例如，内置 P/S 波分离能力。**

> 🌐 **中文** | [English](README.md)

<p align="center">
  <img src="wavefield.gif" alt="弹性波波场模拟" width="600">
</p>

---

## 关于 Fomo

Fomo.jl 是一个用 Julia + CUDA.jl 写的 GPU 波动方程求解器，目标是提供一份可读、可扩展的代码库：一张消费级 NVIDIA 显卡就能跑，性能也还不错。

实现的功能包括 2D/3D 声波与弹性波正演（velocity-stress 格式）、HABC 吸收边界、vacuum 自由表面，以及一个基于 Li et al. (2018) 的耦合 P-S 势场 2D 求解器（常密度各向同性介质下提供天然的 P/S 分离，无需 Helmholtz 后处理）。

### 横向对比

与其他开源包的功能粗略对比：

|                           | SOFI2D/3D    | Devito       | Deepwave        | JUDI.jl          | Fomo.jl                    |
| ------------------------- | ------------ | ------------ | --------------- | ---------------- | -------------------------- |
| 后端                      | Fortran      | DSL→C        | PyTorch + C/CUDA| Julia (via Devito)| Julia + CUDA.jl           |
| 单张消费级 GPU            | ❌            | ⚠️           | ✅               | ⚠️               | ✅                          |
| 源码可读性                | ⚠️ Fortran   | 自动生成     | ❌ C 算子        | 上层包装         | ✅ 全部 Julia               |
| 耦合 P-S 势场求解器       | ❌            | ❌            | ❌               | ❌                | ✅                          |
| 内置 P/S 分离             | 后处理       | 后处理       | 不适用          | 后处理           | ✅（常密度各向同性下）       |
| 吸收边界                  | PML          | PML          | PML             | PML              | HABC                       |

单张 RTX 级 GPU 上的性能：在相近问题规模下，Fomo 比 SOFI2D 快约 60×，比 Deepwave 慢约 1.6×。

---

## 核心特性

- **声波 & 弹性波 2D/3D** —— 交错网格 velocity-stress 格式，最高 10 阶空间精度
- **🆕 耦合 P-S 势场 2D** —— 基于 Li et al. (2018)，将二阶 leapfrog 改写为速度-位置分裂格式，每步可两次施加 HABC（详见 [coupled2d 章节](#-耦合-p-s-势场求解器)）
- **混合吸收边界 (HABC)** —— 结合单程波方程与指数衰减，吸收效果优于 PML，内存开销更低
- **Vacuum 自由表面** —— 通过 vacuum 方法在介质内任意位置设置真实自由界面
- **PyTorch 风格 API** —— 一次函数调用跑完整正演；没有手写时间循环、没有配置文件、没有 MPI 启动器
- **GPU 加速** —— 基于 [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)，算子是纯 Julia 代码，可读可改
- **内置可视化** —— 炮记录与波场视频开箱即用

---

## 快速开始

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

### 弹性波 2D —— 一行函数调用

```julia
using CUDA, Fomo

nx, nz = 400, 300
dh, dt, nt, f0 = 10.0f0, 0.001f0, 2000, 15.0f0

vp  = fill(2500.0f0, nx, nz)
vs  = fill(1200.0f0, nx, nz)
rho = fill(2000.0f0, nx, nz)
vp[:,1] .= 0; vs[:,1] .= 0; rho[:,1] .= 0  # vacuum 自由表面

sx, sz = [nx ÷ 2], [10]
rx, rz = collect(1:2:nx), fill(10, length(1:2:nx))

vx, vz, snaps = elastic2d(vp, vs, rho, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=100, fd_order=8, snap_interval=50)

plot_shot(trace_norm(vz, dims=2), "elastic_vz.png")
plot_wavefield_video(snaps, 50, "elastic.mp4", fps=10, adaptive_clims=true)
```

### 耦合 P-S 势场 —— 天然分离的 P/S 波场

```julia
seis_P, seis_S, snaps_P, snaps_S = coupled2d(vp, vs, dh, dt, nt, f0;
    sx, sz, rx, rz, nbc=100, fd_order=8, snap_interval=50)
# seis_P：纯 P 波势场（只含 PP 反射）
# seis_S：纯 S 波势场（只在 Vs 不连续面产生 PS 转换波）
```

不需要 Helmholtz 分解，不需要后处理。波场在方程层面就被分开了。

---

## 🆕 耦合 P-S 势场求解器

### 背景

弹性波逆时偏移 (RTM) 和全波形反演 (FWI) 中经常需要分离 P 和 S 波场。常见做法是传播完整的 velocity-stress 波场，然后对快照应用 Helmholtz 分解（`∇·` 和 `∇×` 算子）—— 这会增加存储、计算和后处理复杂度。

Li et al. (2018) 提出的耦合 P-S 势场格式可以在方程层面就传播分离的势场，前提是**常密度各向同性介质**这个限制。`coupled2d` 是基于这个格式的一个开源 GPU 实现，我个人目前还没有见到同类开源 GPU 实现，但相关文献很多，也可能是我漏看了。

### 理论

在常密度各向同性弹性介质中，P 波势 P = ∇·**u** 和 S 波势 **S** = ∇×**u** 满足如下耦合二阶方程：

$$\ddot{P} = P\nabla^2\alpha + 2\nabla\alpha\cdot\nabla P - 2P\nabla^2\beta - 2\nabla\beta\cdot(\nabla\times\mathbf{S}) + \alpha\nabla^2 P + \nabla\cdot\mathbf{f}$$

$$\ddot{\mathbf{S}} = \nabla\beta\cdot\nabla\mathbf{S} - (\nabla\beta)\times(\nabla\times\mathbf{S}) + 2(\nabla\beta)\times(\nabla P) + \beta\nabla^2\mathbf{S} + \nabla\times\mathbf{f}$$

其中 α = V²ₚ，β = V²ₛ。这组方程揭示的物理：

1. **模式转换只发生在 V_S 不连续处** —— 若 ∇β = 0，P 和 S 完全解耦
2. **V_P 不连续对 S 波透明** —— 纯 V_P 扰动不产生 PS 转换波
3. P 和 S 在方程层面就**天然分离**，不需要 Helmholtz 分解

### 吸收边界的处理

实现这组方程时，吸收边界是一个绕不过去的问题。

直接对二阶方程做 leapfrog，每个时间步只能施加一次 HABC —— 这等价于一阶 Higdon 吸收边界，斜入射吸收效果较差。

一个可行的做法是把 leapfrog 改写为速度-位置分裂格式，使其与标准 velocity-stress 格式结构同构：

| velocity-stress (`elastic2d`)        | velocity-position (`coupled2d`)            |
| ------------------------------------ | ------------------------------------------ |
| `v_x, v_z`（质点速度）               | `dP/dt, dS/dt`（势场时间导数）             |
| `τ_xx, τ_zz, τ_xz`（应力）           | `P, S`（势场）                             |
| `update_velocity → HABC`             | `update_velocity → HABC`                   |
| `update_stress → HABC`               | `update_position → HABC`                   |

这样每个时间步可以施加两次 HABC，等效于二阶 Higdon 吸收，吸收质量与 `elastic2d` 一致。这个分裂在 Li et al. 原论文里未涉及，是 `coupled2d` 实现中补上的一步。

### 相对传统弹性波求解器的优势

|                       | elastic2d                          | **coupled2d**                       |
| --------------------- | ---------------------------------- | ----------------------------------- |
| 波场数组（2D）        | 10 个（vx, vz, 3 应力 + 备份）     | 8 个（P, S, dP/dt, dS/dt + 备份）   |
| 内存占用              | 5n²                                | **4n²（–20%）**                     |
| P/S 分离              | 需要 Helmholtz 后处理              | **方程内置**                        |
| 模式转换              | 隐式                               | **显式（只在 ∇β ≠ 0 处）**          |
| RTM 成像条件          | 需要 ad-hoc 相位修正               | **与物理扰动一致**                  |
| 密度                  | 任意                               | 常密度（ρ = const）                 |

---

## API 速查

### 正演模拟

| 函数                                                | 说明                              |
| --------------------------------------------------- | --------------------------------- |
| `acoustic2d(vp, rho, dh, dt, nt, f0; ...)`          | 2D 声波正演                       |
| `elastic2d(vp, vs, rho, dh, dt, nt, f0; ...)`       | 2D 弹性波正演                     |
| `coupled2d(vp, vs, dh, dt, nt, f0; ...)`            | 🆕 2D 耦合 P-S 势场正演           |
| `acoustic3d(vp, rho, dh, dt, nt, f0; ...)`          | 3D 声波正演                       |
| `elastic3d(vp, vs, rho, dh, dt, nt, f0; ...)`       | 3D 弹性波正演                     |

**通用关键字参数：**

| 参数            | 默认值 | 说明                                |
| --------------- | ------ | ----------------------------------- |
| `sx, sz`        | —      | 震源坐标（网格索引）                |
| `rx, rz`        | —      | 接收器坐标（网格索引）              |
| `nbc`           | `50`   | 吸收边界层数                        |
| `fd_order`      | `8`    | 有限差分阶数（2, 4, 6, 8, 10）      |
| `snap_interval` | `0`    | 快照间隔（0 = 不保存）              |

**`coupled2d` 专属：**

| 参数            | 默认值       | 说明                              |
| --------------- | ------------ | --------------------------------- |
| `v_ref_p`       | `min(vp)`    | P 场 HABC 参考速度                |
| `v_ref_s`       | `min(vs)`    | S 场 HABC 参考速度                |
| `smooth_sigma`  | `3.0`        | 介质参数高斯平滑标准差            |

**返回值：**
- `acoustic2d` / `elastic2d` → `(vx_record, vz_record, snapshots)`
- `coupled2d` → `(P_record, S_record, P_snapshots, S_snapshots)`

### 工具函数

| 函数                                                                     | 说明                  |
| ------------------------------------------------------------------------ | --------------------- |
| `ricker_wavelet(f0, dt, nt)`                                             | 生成 Ricker 子波      |
| `trace_norm(data; dims)`                                                 | 道归一化              |
| `plot_shot(data, filename)`                                              | 保存炮记录图          |
| `plot_wavefield_video(snaps, interval, filename; fps, adaptive_clims)`   | 导出波场视频          |

---

## 架构

```
src/
├── Fomo.jl                  # 模块入口
├── acquisition/             # 震源与接收器（2D + 3D）
├── boundary/
│   ├── habc/                # 混合吸收边界（刘洋方法）
│   └── sponge.jl            # Sponge 边界（可选）
├── equations/
│   ├── acoustic2d/          # 2D 声波（速度-压力）
│   ├── acoustic3d/          # 3D 声波
│   ├── elastic2d/           # 2D 弹性波（速度-应力）
│   ├── elastic3d/           # 3D 弹性波
│   └── coupled2d/           # 🆕 耦合 P-S 势场
├── utils/                   # 差分系数、padding、子波
└── visualization/           # 绘图与视频导出
```

加入新方程 = 在 `equations/` 下加一个文件夹，包含 `medium.jl`, `wavefield.jl`, `update_*.jl` 和入口函数。**coupled2d 求解器就是用这个结构在几周内加进来的** —— 这是架构设计的红利。

---

## 方法

Fomo 实现了三类波动方程求解器：

**声波 & 弹性波（velocity-stress）** —— 标准交错网格格式（Virieux, 1986），最高 10 阶 FD 算子，二阶 leapfrog 时间推进，HABC 吸收边界。

**耦合 P-S 势场** —— 基于 Li et al. (2018)。将二阶时间方程改写为与 velocity-stress 同构的速度-位置分裂格式，实现二阶等效 HABC 吸收。空间上使用规则网格中心差分算子。

---

## 环境要求

- Julia ≥ 1.10
- NVIDIA GPU（支持 CUDA）
- CUDA.jl ≥ 5.0

---

## License

MIT

---

## 参考文献

- **Li, Y. E., Du, Y., Yang, J., Cheng, A., & Fang, X. (2018).** Elastic reverse time migration using acoustic propagators. *Geophysics*, 83(5), S399–S408. doi: [10.1190/GEO2017-0687.1](https://doi.org/10.1190/GEO2017-0687.1)
- **Virieux, J. (1986).** P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.
- **Liu, Y., & Sen, M. K. (2010).** A hybrid scheme for absorbing edge reflections in numerical modeling of wave propagation. *Geophysics*, 75(2), A1–A6.

## 致谢

- HABC 公式：刘洋教授
- 交错网格 FD：Virieux (1986)
- 耦合 P-S 势场方程：Li et al. (2018)
- 实现、GPU 移植与速度-位置分裂格式：本项目
