# example/acoustic3d_demo.jl
#
# 3D 声波模拟示例（参考 acoustic2d_demo.jl 扩展）

using Pkg
using Fomo

# ── 模型参数 ──
nx, ny, nz = 101, 101, 101    # 网格大小
dh = 10.0f0                    # 网格间距 (m)
dt = 0.001f0                   # 时间步长 (s)
nt = 500                       # 时间步数
f0 = 15.0f0                    # 震源主频 (Hz)

# ── 均匀速度模型 ──
vp = fill(3000.0f0, nx, ny, nz)
rho = fill(2000.0f0, nx, ny, nz)

# ── 震源：中心位置 ──
sx = [nx ÷ 2]
sy = [ny ÷ 2]
sz = [nz ÷ 2]

# ── 接收器：沿 x 方向排列，y/z 固定在中心 ──
rx = collect(1:nx)
ry = fill(ny ÷ 2, nx)
rz = fill(nz ÷ 2, nx)

# ── 运行 3D 声波模拟 ──
seis_vx, seis_vy, seis_vz, snaps = acoustic3d(
    vp, rho, dh, dt, nt, f0;
    sx=sx, sy=sy, sz=sz,
    rx=rx, ry=ry, rz=rz,
    nbc=50, fd_order=8,
    snap_interval=50,
    snap_plane=:xz,       # 保存 xz 平面切片（y 方向切片）
    snap_index=ny ÷ 2     # 在 y 中心切片
)

println("Seismogram size (vx): ", size(seis_vx))
println("Seismogram size (vy): ", size(seis_vy))
println("Seismogram size (vz): ", size(seis_vz))
println("Number of snapshots: ", length(snaps))
if length(snaps) > 0
    println("Snapshot size: ", size(snaps[1]))
end
