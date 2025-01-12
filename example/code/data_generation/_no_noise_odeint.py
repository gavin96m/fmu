import numpy as np
from scipy.integrate import odeint
import os
# 导入车辆模型所需的函数和参数
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks



# 数据保存路径
filename_suffix = 'no_noise_odeint'
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')
os.makedirs(data_dir, exist_ok=True)


# 定义车辆动力学函数
def func_KS(x, t, u, p):
    return vehicle_dynamics_ks(x, u, p)


# 仿真参数
tStart = 0  # 起始时间
tFinal = 10  # 结束时间
dt = 0.001  # 调整时间步长，减少数据量

# 加载车辆参数
p = parameters_vehicle1()

# 多次仿真参数
num_simulations = 100  # 调整仿真次数以避免过多数据
X_all = []
Y_all = []

for _ in range(num_simulations):
    # 初始化状态
    delta0 = np.random.uniform(-5, 5)  # 随机初始转向角
    vel0 = np.random.uniform(5, 20)  # 随机初始速度
    Psi0 = np.random.uniform(-np.pi, np.pi)  # 随机航向角
    sy0 = 0  # 固定纵向位置
    initialState = [0, sy0, delta0, vel0, Psi0]
    x0_KS = init_ks(initialState)

    # 时间数组
    t = np.arange(tStart, tFinal, dt)

    # 随机生成控制输入
    u_delta = np.random.uniform(-0.1, 0.1, len(t))  # 转向角输入的随机序列
    u_v = np.random.uniform(3, 25, len(t))  # 速度输入的随机序列

    # 运行仿真
    x = odeint(func_KS, x0_KS, t, args=([u_delta[0], u_v[0]], p))


    for i in range(1, len(t)):
        u = [u_delta[i], u_v[i]]  # 更新控制输入
        x_next = odeint(func_KS, x[-1], [t[i - 1], t[i]], args=(u, p))

        # 限制航向角 psi 在 [-pi, pi] 范围内
        x_next[1][4] = (x_next[1][4] + np.pi) % (2 * np.pi) - np.pi

        x = np.vstack((x, x_next[1]))

    # 准备数据用于 GPR
    num_samples = min(len(x) - 1, len(u_delta), len(u_v))

    # 输入特征：当前状态和控制输入
    X_state = x[:num_samples]
    X_control = np.array([[u_delta[i], u_v[i]] for i in range(num_samples)])

    # 合并状态和控制输入
    X = np.hstack((X_state, X_control))

    # 输出目标：下一时间步的状态
    Y = x[1:num_samples + 1]

    # 保存当前仿真数据
    X_all.append(X)
    Y_all.append(Y)

# 合并所有仿真的数据
X_all = np.vstack(X_all)
Y_all = np.vstack(Y_all)

# 保存数据到文件
np.save(os.path.join(data_dir, f'X_{filename_suffix}.npy'), X_all)
np.save(os.path.join(data_dir, f'Y_{filename_suffix}.npy'), Y_all)

# 可选：保存 CSV 文件
np.savetxt(os.path.join(data_dir, f'X_{filename_suffix}.csv'), X_all, delimiter=',', header='x_pos,y_pos,delta,v,psi,u_delta,u_v', comments='')
np.savetxt(os.path.join(data_dir, f'Y_{filename_suffix}.csv'), Y_all, delimiter=',', header='x_pos_next,y_pos_next,delta_next,v_next,psi_next', comments='')

print("Data extraction finished")