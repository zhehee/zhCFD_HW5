import numpy as np
import matplotlib.pyplot as plt

# 模拟参数
N = 101  # 网格点数（包括边界）
h = 1.0 / (N - 1)  # 网格间距
nu = 0.001  # 运动粘度
dt = 0.0001  # 时间步长
max_iter = 1000000  # 最大迭代次数
max_p_iter = 10000      # 最大压力迭代次数
p_tol = 1e-5         # 压力残差容差
p_residual = 1.0     # 初始化残差
velocity_tol = 1e-7   # 速度收敛容差
check_interval = 100    # 收敛检查间隔
# 初始化场变量
u = np.zeros((N, N))
v = np.zeros((N, N))
p = np.zeros((N, N))

# 设置上边界的水平速度（sin²(πx)）
x = np.linspace(0, 1, N)
u_top = np.sin(np.pi * x) ** 2
u[:, -1] = u_top  # 上边界条件


# 边界掩模（用于强制固定边界条件）
def apply_boundary_conditions(u, v):
    # 上边界
    u[:, -1] = u_top  # 水平速度
    v[:, -1] = 0  # 垂直速度

    # 下边界
    u[:, 0] = 0
    v[:, 0] = 0

    # 左边界
    u[0, :] = 0
    v[0, :] = 0

    # 右边界
    u[-1, :] = 0
    v[-1, :] = 0
    return u, v

# 压力边界条件函数
def apply_pressure_bc(p):
    # Neumann边界条件（法向梯度为零）
    p[0, :] = p[1, :]     # 左边界
    p[-1, :] = p[-2, :]   # 右边界
    p[:, 0] = p[:, 1]     # 下边界
    p[:, -1] = p[:, -2]   # 上边界
    return p


def plot_analysis_results(u, v):
    #可视化分析结果
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    #  流线图
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, u.T, v.T, density=2, color='lightgray')
    plt.title('Vortex Core Locations')
    plt.savefig('1_streamlines.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    #  水平中线速度剖面
    plt.figure(figsize=(10, 4))
    mid_y = N // 2
    absolute_velocityy = np.sqrt(u[mid_y, :] ** 2 + v[mid_y, :] ** 2)
    plt.plot(x, absolute_velocityy, 'r-', linewidth=2)
    plt.xlabel('x coordinate')
    plt.ylabel('Velocity')
    plt.title(f'Horizontal Velocity Profile @ y={y[mid_y]:.2f}')
    plt.grid(True)
    plt.savefig('2_horizontal_profile.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    #  垂直中线速度剖面
    plt.figure(figsize=(6, 8))
    mid_x = N // 2
    absolute_velocityx= np.sqrt(u[:, mid_x] ** 2 + v[:, mid_x] ** 2)
    plt.plot(absolute_velocityx, y, 'b-', linewidth=2)
    plt.ylabel('y coordinate')
    plt.xlabel('Velocity')
    plt.title(f'Vertical Velocity Profile @ x={x[mid_x]:.2f}')
    plt.grid(True)
    plt.savefig('3_vertical_profile.pdf', bbox_inches='tight', dpi=300)
    plt.close()


# 主求解循环
prev_u = np.zeros_like(u)
prev_v = np.zeros_like(v)
converged = False
for iter in range(max_iter):
    # 保存前次速度场用于收敛判断
    if iter % check_interval == 0:
        prev_u[:] = u
        prev_v[:] = v
    # 临时速度场
    u_prev = u.copy()
    v_prev = v.copy()

    # 计算中间速度（扩散项 + 对流项）
    u[1:-1, 1:-1] += dt * (
            nu * (u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] + u_prev[1:-1, 2:] + u_prev[1:-1, :-2] - 4 * u_prev[1:-1,1:-1]) / h ** 2
            - (u_prev[1:-1, 1:-1] * (u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2 * h)
               + v_prev[1:-1, 1:-1] * (u_prev[1:-1, 2:] - u_prev[1:-1, :-2]) / (2 * h))
    )

    v[1:-1, 1:-1] += dt * (
            nu * (v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] + v_prev[1:-1, 2:] + v_prev[1:-1, :-2] - 4 * v_prev[1:-1,1:-1]) / h ** 2
            - (u_prev[1:-1, 1:-1] * (v_prev[2:, 1:-1] - v_prev[:-2, 1:-1]) / (2 * h)
               + v_prev[1:-1, 1:-1] * (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2 * h))
    )

    # 应用速度边界条件
    u, v = apply_boundary_conditions(u, v)


    for p_iter in range(max_p_iter):
        p_old = p.copy()

        p[1:-1, 1:-1] = (
                            (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2])
                            - h ** 2 / (4 * dt) * (
                                    (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h) +
                                    (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
                            )
                        ) / 4
        # 应用边界条件
        p = apply_pressure_bc(p)

        # 计算残差（仅内部点）
        p_residual = np.max(np.abs(p[1:-1, 1:-1] - p_old[1:-1, 1:-1]))

        # 收敛检查
        if p_residual < p_tol:
            print(f"压力场在 {p_iter+1} 次迭代后收敛，残差 = {p_residual:.3e}")
            break

    # 最终收敛检查
    if p_residual > p_tol:
        print(f"警告：压力场未在{max_p_iter}次迭代内收敛，最终残差{p_residual:.3e}")
    # 速度修正
    u[1:-1, 1:-1] -= dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * h)
    v[1:-1, 1:-1] -= dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * h)

    # 最终应用边界条件
    u, v = apply_boundary_conditions(u, v)
    # 收敛性检查
    if iter % check_interval == 0 and iter > 0:
        # 计算速度场变化量
        du_max = np.max(np.abs(u - prev_u))
        dv_max = np.max(np.abs(v - prev_v))


        # 输出监控信息
        print(f"Iter {iter:04d}: Δu={du_max:.2e}, Δv={dv_max:.2e}")

        # 双重收敛标准
        if (du_max < velocity_tol and
                dv_max < velocity_tol):
            print(f"速度场在 {iter} 次迭代后收敛!")
            converged = True
            break


plot_analysis_results(u, v)