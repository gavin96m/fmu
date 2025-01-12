import time
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

start = time.time()

# 设置路径
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

# 仅选择 'ks' 模型进行测试
model_variants = ['ks']
model_path = os.path.join(data_dir, 'gpr_ks_multi_output_model.pkl')
model_data = joblib.load(model_path)
gpr_model = model_data['model']
X_scaler = model_data['X_scaler']
Y_scaler = model_data['Y_scaler']



print("X_scaler mean:", X_scaler.mean_)
print("X_scaler scale:", X_scaler.scale_)
print("Y_scaler mean:", Y_scaler.mean_)
print("Y_scaler scale:", Y_scaler.scale_)


# 加载测试数据
X_test = np.load(os.path.join(data_dir, 'X_no_noise_from_fmu.npy'))

# 设置初始状态和模拟参数
initial_state = [0.0, 0.0, 0.1, 15.0, 0.0]
trajectory = [initial_state[:2]]  # 使用测试数据的初始状态
total_time = 15  # 总模拟时间（秒）
step_size = 0.01  # 时间步长（秒）
time_points = np.arange(0, total_time, step_size)
num_steps = len(time_points)

# 定义控制输入
u_delta = -0.1  # 恒定转向角控制输入
u_v = 15  # 恒定速度控制输入

current_state = initial_state.copy()

for step in range(num_steps):
    # Control inputs (could be time-varying if desired)
    control_inputs = [u_delta, u_v]

    # Prepare input: current state + control inputs
    X_input = np.hstack((current_state, control_inputs)).reshape(1, -1)
    X_input_scaled = X_scaler.transform(X_input)

    # Predict next state
    Y_pred_scaled = gpr_model.predict(X_input_scaled)
    Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)
    next_state = Y_pred[0]

    # Update current state
    current_state = next_state.copy()

    # Save trajectory
    trajectory.append(current_state[:2])

# Convert trajectory to numpy array for plotting
trajectory = np.array(trajectory)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Predicted Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory Predicted by GPR Model')
plt.legend()
plt.grid(True)
plt.show()


end = time.time()
print(f'Total Time: {end - start}')


# import time
# import numpy as np
# import os
# import joblib
# import matplotlib.pyplot as plt
#
# start = time.time()
#
# current_dir = os.path.dirname(__file__)
# data_dir = os.path.join(current_dir, '../data')
#
# model_variants = ['ks']
#
# model_path = os.path.join(data_dir, 'gpr_ks_multi_output_model.pkl')
# model_data = joblib.load(model_path)
# gpr_model = model_data['model']
# X_scaler = model_data['X_scaler']
# Y_scaler = model_data['Y_scaler']
#
#
# X_test = np.load(os.path.join(data_dir, 'X_no_noise_from_fmu.npy'))
#
#
# trajectory = [X_test[0, :2]]  # 使用测试数据的初始状态
# initial_state = [0.0, 0.0, 0.1, 15.0, 0.0, 0.0, 0.0]
# # 定义总时间和时间步长
# total_time = 15  # 总模拟时间（秒）
# step_size = 0.01  # 时间步长（秒）
# time_points = np.arange(0, total_time, step_size)
# num_steps = len(time_points)
#
#
# # 定义控制输入
# u_delta = -0.1  # 恒定转向角控制输入
# u_v = 15  # 恒定速度控制输入
#
# current_state = initial_state.copy()
#
# for step in range(num_steps):
#
#     # Update current_state with control inputs
#     current_state[-2:] = [u_delta, u_v]
#
#     # Transform current_state# 在进行标准化之前，将 current_state 转换为 NumPy 数组
#     X_input = X_scaler.transform(np.array(current_state).reshape(1, -1))
#
#     # Predict next state
#     predicted_state_scaled = gpr_model.predict(X_input)
#     predicted_state = Y_scaler.inverse_transform(predicted_state_scaled)[0]
#
#     # Record trajectory
#     trajectory.append(predicted_state[:2])  # x_pos_next, y_pos_next
#
#     # Prepare current_state for next iteration
#     current_state[:-2] = predicted_state  # 更新状态变量（不包括控制输入）
#
# trajectory = np.array(trajectory)
# plt.figure(figsize=(10, 5))
# plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='Predicted Trajectory')
# plt.title(f'Predicted Vehicle Trajectory for ks')
# plt.xlabel('Position X (m)')
# plt.ylabel('Position Y (m)')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# end = time.time()
# print(f'Total Time: {end-start}')