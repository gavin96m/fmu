import numpy as np
import os
from fmu_conversion.ks_fmu_noise import Model  # Assuming ks_fmu.py is in the same directory

# 仿真参数
t_start = 0  # 起始时间
t_final = 10  # 结束时间
step_size = 0.001  # 调整步长
num_simulations = 100  # 多次仿真的次数


# 数据保存路径
filename_suffix = 'with_noise_from_fmu'
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')
os.makedirs(data_dir, exist_ok=True)


# 初始化保存数据的列表
X_all = []
Y_all = []

for _ in range(num_simulations):
    # 初始化车辆模型
    vehicle = Model()

    # 随机初始化控制输入
    steering_angle = np.random.uniform(-0.1, 0.1)  # 随机初始转向角
    velocity = np.random.uniform(3, 25)  # 随机初始速度
    vehicle.fmi2SetReal([0, 1], [steering_angle, velocity])  # 设置初始控制输入

    # 保存初始状态和输入
    X_current = []
    Y_current = []

    # 进行仿真
    time_points = np.arange(t_start, t_final, step_size)
    for t in time_points:
        vehicle.fmi2DoStep(t, step_size, True)
        current_state = vehicle.state.copy()  # 当前状态
        control_inputs = [vehicle.u[0], vehicle.u[1]]  # 当前控制输入

        # 保存输入特征：当前状态和控制输入
        X_current.append(np.hstack((current_state, control_inputs)))

        # 更新状态为下一个时间步的状态
        if len(X_current) > 1:
            Y_current.append(current_state)

    # 合并当前仿真的数据
    X_current = np.array(X_current[:-1])  # 忽略最后一步的状态
    Y_current = np.array(Y_current)

    X_all.append(X_current)
    Y_all.append(Y_current)

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

