import time
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

start = time.time()

from ks_fmu import Model  # Assuming ks_fmu.py is in the same directory

# 仿真参数
t_start = 0  # 起始时间
t_final = 10  # 结束时间
step_size = 0.001  # 调整步长
num_simulations = 100  # 多次仿真的次数


# 数据保存路径
filename_suffix = 'no_noise_from_fmu'
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
    steering_angle = np.random.uniform(-0.5, 0.5)  # 随机初始转向角
    velocity = np.random.uniform(3, 40)  # 随机初始速度
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


# 设置路径
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

# 加载数据
filename_suffix = 'no_noise_from_fmu'
X = np.load(os.path.join(data_dir, f'X_{filename_suffix}.npy'))
Y = np.load(os.path.join(data_dir, f'Y_{filename_suffix}.npy'))

# 对数据进行子采样，减少训练样本数量
# X_sampled, _, Y_sampled, _ = train_test_split(X, Y, train_size=30000, random_state=42)

# 设置簇的数量，这样会有较少的代表性样本
n_clusters = 10000  # 可以根据内存情况调整
batch_size = 4096  # 可以调整批量大小以提高计算速度
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42, n_jobs=-1)
kmeans.fit(X)


# 获取每个簇的中心点
X_sampled = kmeans.cluster_centers_

# 选择相应的 Y_sampled（可以基于距离最近的点）
Y_sampled = []
for center in X_sampled:
    idx = np.argmin(np.linalg.norm(X - center, axis=1))
    Y_sampled.append(Y[idx])

Y_sampled = np.array(Y_sampled)

print("Data sampling and clustering successful. Proceeding to model training...")

# 数据预处理
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X_sampled)
Y_scaled = Y_scaler.fit_transform(Y_sampled)
print("X_scaler mean:", X_scaler.mean_)
print("X_scaler scale:", X_scaler.scale_)
print("Y_scaler mean:", Y_scaler.mean_)
print("Y_scaler scale:", Y_scaler.scale_)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 三种不同的核函数配置
kernels = [
    C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)),
    # C(1.0, (1e-3, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e5), nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)),
    # C(1.0, (1e-3, 1e5)) * RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
]

# 模型列表
models = []
errors = []
trajectories = []

# 进行三种核函数的训练和评估
for i, kernel in enumerate(kernels):
    print(f"Training model {i + 1} with kernel: {kernel}")
    gpr = MultiOutputRegressor(GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True
    ), n_jobs=-1)
    gpr.fit(X_train, Y_train)
    Y_pred = gpr.predict(X_test)
    Y_pred_orig = Y_scaler.inverse_transform(Y_pred)
    Y_test_orig = Y_scaler.inverse_transform(Y_test)

    # 评估模型性能
    mse = mean_squared_error(Y_test_orig, Y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
    r2 = r2_score(Y_test_orig, Y_pred_orig)
    errors.append((mse, rmse, mae, r2))

    # 保存模型
    models.append(gpr)

    print(f"Model {i + 1} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

    # 进行轨迹模拟
    initial_state = [0.0, 0.0, 0.1, 15.0, 0.0]
    trajectory = [initial_state[:2]]  # 使用测试数据的初始状态
    total_time = 15  # 总模拟时间（秒）
    step_size = 1  # 时间步长（秒）
    time_points = np.arange(0, total_time, step_size)
    num_steps = len(time_points)
    current_state = initial_state.copy()
    u_delta = -0.1  # 恒定轮角控制输入
    u_v = 15  # 恒定速度控制输入

    for step in range(num_steps):
        # Control inputs (could be time-varying if desired)
        control_inputs = [u_delta, u_v]

        # Prepare input: current state + control inputs
        X_input = np.hstack((current_state, control_inputs)).reshape(1, -1)
        X_input_scaled = X_scaler.transform(X_input)

        # Predict next state
        Y_pred_scaled = gpr.predict(X_input_scaled)
        Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)
        next_state = Y_pred[0]

        # Update current state
        current_state = next_state.copy()

        # Save trajectory
        trajectory.append(current_state[:2])

    trajectories.append(np.array(trajectory))

# 作图比较模型性能
metrics = ['MSE', 'RMSE', 'MAE', 'R^2']
plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
    values = [error[i] for error in errors]
    plt.bar(range(1, len(kernels) + 1), values, tick_label=[f'Model {j+1}' for j in range(len(kernels))])
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.title(f'Model Comparison - {metric}')
    plt.grid(True)

plt.tight_layout()
plt.show()

# 轨迹绘制
plt.figure(figsize=(14, 10))
for i, trajectory in enumerate(trajectories):
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Model {i+1} Trajectory')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory Predicted by Different GPR Models')
plt.legend()
plt.grid(True)
plt.show()

# 保存模型和缩放器
model_data = {
    'model': models[0],  # 保存第一个模型
    'X_scaler': X_scaler,
    'Y_scaler': Y_scaler
}

model_path = os.path.join(data_dir, 'gpr_ks_multi_output_model.pkl')
joblib.dump(model_data, model_path)
print(f"多输出 GPR 模型和缩放器已保存到 {model_path}")

end = time.time()
print(f'Total Time: {end - start}')


with open('output.log', 'w') as f:
    print("Your message", file=f)


plt.savefig(os.path.join(data_dir, 'model_comparison.png'))
plt.savefig(os.path.join(data_dir, 'trajectory_plot.png'))
