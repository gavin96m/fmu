import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

# 设置模型文件名后缀和数据路径
model_variants = ['ks', 'ks_with_noise', 'st', 'st_with_noise']


# 用于读取和分析模型的函数
def analyze_model(filename_suffix):
    # 加载模型和缩放器
    model_path = os.path.join(data_dir, f'gpr_{filename_suffix}_multi_output_model.pkl')
    model_data = joblib.load(model_path)
    gpr_model = model_data['model']
    X_scaler = model_data['X_scaler']
    Y_scaler = model_data['Y_scaler']

    # 定义模型名称与测试数据名称的映射
    test_data_mapping = {
        'ks': ('X_no_noise_from_fmu.npy', 'Y_no_noise_from_fmu.npy'),
        'ks_with_noise': ('X_with_noise_from_fmu.npy', 'Y_with_noise_from_fmu.npy'),
        'st': ('X_st_no_noise_from_fmu.npy', 'Y_st_no_noise_from_fmu.npy'),
        'st_with_noise': ('X_st_with_noise_from_fmu.npy', 'Y_st_with_noise_from_fmu.npy'),
    }

    # 使用映射获取测试数据文件名
    X_test_filename, Y_test_filename = test_data_mapping.get(filename_suffix, (None, None))

    if X_test_filename and Y_test_filename:
        # 加载测试数据
        X_test = np.load(os.path.join(data_dir, X_test_filename))
        Y_test = np.load(os.path.join(data_dir, Y_test_filename))
    else:
        print(f"测试数据文件未找到匹配的模型名称: {filename_suffix}")

    # 标准化测试数据
    X_test_scaled = X_scaler.transform(X_test)
    Y_test_scaled = Y_scaler.transform(Y_test)

    # 保存预测结果的文件名
    save_path = os.path.join(data_dir, f'{filename_suffix}_predictions.npz')

    # 检查文件是否已经存在，若存在则直接加载
    if os.path.exists(save_path):
        print(f"加载已有的预测结果: {save_path}")
        data = np.load(save_path)
        Y_pred_scaled = data['Y_pred_scaled']
        Y_pred_orig = data['Y_pred_orig']
        Y_test_orig = data['Y_test_orig']
    else:
        print(f"计算并保存新的预测结果: {save_path}")

        # 分批预测
        batch_size = 5000  # 设置批次大小
        Y_pred_scaled_list = []

        for i in range(0, X_test_scaled.shape[0], batch_size):
            batch_X = X_test_scaled[i:i + batch_size, :]
            Y_pred_batch_scaled = gpr_model.predict(batch_X)
            Y_pred_scaled_list.append(Y_pred_batch_scaled)

        # 合并所有批次的预测结果
        Y_pred_scaled = np.vstack(Y_pred_scaled_list)

        # 反标准化预测和测试数据
        Y_pred_orig = Y_scaler.inverse_transform(Y_pred_scaled)
        Y_test_orig = Y_scaler.inverse_transform(Y_test_scaled)

        # 保存数据到文件
        np.savez(save_path, Y_pred_scaled=Y_pred_scaled, Y_pred_orig=Y_pred_orig, Y_test_orig=Y_test_orig)

    # # 模型预测时的特征数
    # num_features = X_test_scaled.shape[1]
    #
    # # 分批预测
    # batch_size = 5000  # 设置批次大小
    # Y_pred_scaled_list = []
    #
    # for i in range(0, X_test_scaled.shape[0], batch_size):
    #     batch_X = X_test_scaled[i:i + batch_size, :num_features]
    #     Y_pred_batch_scaled = gpr_model.predict(batch_X)
    #     Y_pred_scaled_list.append(Y_pred_batch_scaled)
    #
    # # 合并所有批次的预测结果
    # Y_pred_scaled = np.vstack(Y_pred_scaled_list)
    #
    #
    # # 模型预测
    # # Y_pred_scaled = gpr_model.predict(X_test_scaled)
    # Y_pred_orig = Y_scaler.inverse_transform(Y_pred_scaled)
    # Y_test_orig = Y_scaler.inverse_transform(Y_test_scaled)

    # 评估模型性能
    mse = mean_squared_error(Y_test_orig, Y_pred_orig)
    r2 = r2_score(Y_test_orig, Y_pred_orig)
    mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
    rmse = np.sqrt(mse)

    print(f"模型 {filename_suffix} 误差指标：")
    print(f"  Mean Squared Error (MSE): {mse}")
    print(f"  Root Mean Squared Error (RMSE): {rmse}")
    print(f"  Mean Absolute Error (MAE): {mae}")
    print(f"  R^2 Score: {r2}")

    # 逐个输出变量计算误差
    mse_per_output = mean_squared_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')
    mae_per_output = mean_absolute_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')

    for i in range(Y_pred_orig.shape[1]):
        print(f"  输出变量 {i}: MSE = {mse_per_output[i]}, MAE = {mae_per_output[i]}")

    # 绘制残差图
    residuals = Y_test_orig - Y_pred_orig
    # 采样数据以减少绘图数据点数量，例如每隔100个点取一个
    residuals_sampled = residuals[::100, :]

    # 绘制每个输出变量的残差图
    num_outputs = Y_test_orig.shape[1]
    plt.figure(figsize=(12, 6))
    for i in range(num_outputs):
        plt.plot(residuals_sampled[:, i], linestyle='-', linewidth=0.5, label=f'Output {i}')  # 简化标记
    plt.title(f'Residuals for {filename_suffix}')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # 绘制整体误差分布直方图
    # plt.figure(figsize=(10, 6))
    # plt.hist(residuals.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    # plt.title('Residual Distribution for All Outputs')
    # plt.xlabel('Residual')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()
    #
    # # 为每个输出变量绘制误差分布图
    # num_outputs = Y_test_orig.shape[1]
    # for i in range(num_outputs):
    #     plt.figure(figsize=(8, 5))
    #     plt.hist(residuals[:, i], bins=50, color='green', alpha=0.7, edgecolor='black')
    #     plt.title(f'Residual Distribution for Output Variable {i}')
    #     plt.xlabel('Residual')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.show()


    # 绘制预测轨迹
    trajectory = [X_test[0, :2]]  # 使用测试数据的初始状态

    if filename_suffix in ['ks', 'ks_with_noise']:
        initial_state = [0.0, 0.0, 0.1, 15.0, 0.0, 0.0, 0.0]
    elif filename_suffix in ['st', 'st_with_noise']:
        initial_state = [0.0, 0.0, 0.1, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print("start")
    # 定义总时间和时间步长
    total_time = 15  # 总模拟时间（秒）
    step_size = 0.1  # 时间步长（秒）
    time_points = np.arange(0, total_time, step_size)
    num_steps = len(time_points)

    # 定义控制输入
    u_delta = -0.05  # 恒定转向角控制输入
    u_v = 15  # 恒定速度控制输入

    current_state = initial_state.copy()

    for step in range(num_steps):

        # Update current_state with control inputs
        current_state[-2:] = [u_delta, u_v]

        # Transform current_state# 在进行标准化之前，将 current_state 转换为 NumPy 数组
        X_input = X_scaler.transform(np.array(current_state).reshape(1, -1))

        # Predict next state
        predicted_state_scaled = gpr_model.predict(X_input)
        predicted_state = Y_scaler.inverse_transform(predicted_state_scaled)[0]

        # Record trajectory
        trajectory.append(predicted_state[:2])  # x_pos_next, y_pos_next

        # Prepare current_state for next iteration
        current_state[:-2] = predicted_state  # 更新状态变量（不包括控制输入）

    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 5))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='Predicted Trajectory')
    plt.title(f'Predicted Vehicle Trajectory for {filename_suffix}')
    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Error Distribution Plots误差分布图
    num_outputs = Y_test_orig.shape[1]
    for i in range(num_outputs):
        plt.figure(figsize=(8, 5))
        plt.hist(residuals[:, i], bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.title(f'Residual Distribution for Output Variable {i}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Error Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title('Residuals Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual Value')
    plt.legend([f'Output {i}' for i in range(num_outputs)])
    plt.grid(True)
    plt.show()

    print("-----------------")
    print("done")

# 主程序
if __name__ == "__main__":
    for variant in model_variants:
        analyze_model(variant)


    # model_variants = ['ks', 'ks_with_noise', 'st', 'st_with_noise']
    # for variant in ['ks']:

    # for variant in ['ks', 'ks_with_noise', 'st', 'st_with_noise']:
    #     analyze_model(variant)
