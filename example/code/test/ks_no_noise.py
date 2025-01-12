import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# 定义数据路径和文件后缀
filename_suffix = 'no_noise_from_fmu'
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

# 加载已保存的模型和缩放器
gpr_models = joblib.load(os.path.join(data_dir, 'gpr_ks_multi_output_model.pkl'))
X_scaler = joblib.load(os.path.join(data_dir, 'X_scaler.pkl'))
Y_scaler = joblib.load(os.path.join(data_dir, 'Y_scaler.pkl'))

# 假设初始状态
current_state = np.array([0, 0, 0.1, 15, 0])  # 初始状态 [x_pos, y_pos, delta, speed, heading_angle]
control_input = np.array([0, 5])  # 假设恒定的控制输入

# 初始化轨迹
trajectory = [current_state[:2]]  # 仅记录 x_pos, y_pos

# 仿真参数
total_time = 10  # 总时间
step_size = 0.1  # 时间步长
time_steps = int(total_time / step_size)

# 迭代预测轨迹
for _ in range(time_steps):
    # 标准化当前状态和控制输入
    X_input = np.hstack((X_scaler.transform(current_state.reshape(1, -1)), control_input.reshape(1, -1)))

    # 预测下一个时间点的状态
    predicted_state_scaled = gpr_models.predict(X_input)

    # 反标准化得到原始空间的预测状态
    predicted_state = Y_scaler.inverse_transform(predicted_state_scaled)

    # 更新当前状态为预测的状态
    current_state = predicted_state[0]

    # 记录轨迹点
    trajectory.append(current_state[:2])  # 仅记录 x_pos, y_pos

# 将轨迹转换为数组以便绘图
trajectory = np.array(trajectory)

# 绘制轨迹图
plt.figure(figsize=(10, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='Predicted Vehicle Trajectory')
plt.title('Predicted Vehicle Trajectory over Time')
plt.xlabel('Position X (m)')
plt.ylabel('Position Y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
