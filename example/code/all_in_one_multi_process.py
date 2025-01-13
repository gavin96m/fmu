import os
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fmu_conversion.ks_fmu import Model
import time
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultiOutputGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[1], rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# 参数范围设置
# train_simulations_range = range(50, 501, 50)  # 训练集仿真次数范围
train_simulations_range = [50, 100, 200]  # 训练集仿真次数范围
# test_simulations_range = range(11, 101, 10)  # 测试集仿真次数范围
test_simulations_range = [11, 31]# 测试集仿真次数范围
# step_size_range = [0.2, 0.1, 0.05, 0.01]  # 时间步长范围
step_size_range = [0.2, 0.1, 0.05]  # 时间步长范围
# t_start, t_final = 0, 10  # 仿真时间范围
t_start, t_final = 0, 5  # 仿真时间范围
log_file = "simulation_results.txt"  # 日志文件路径
data_dir = os.path.join(os.path.dirname(__file__), './data')
os.makedirs(data_dir, exist_ok=True)

# 检查日志是否已完成某个参数组合
def is_already_completed(train_simulations, test_simulations, step_size):
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.readlines()
            for line in logs:
                if f"Train Simulations: {train_simulations}, Test Simulations: {test_simulations}, Step Size: {step_size}" in line:
                    return True
    return False

# 日志写入函数
def write_log(log_message):
    with open(log_file, "a") as f:
        f.write(log_message + "\n")

# 数据生成函数
def generate_data(num_simulations, step_size, filename):
    X_all, Y_all = [], []
    for _ in range(num_simulations):
        vehicle = Model()
        steering_angle = np.random.uniform(-0.5, 0.5)
        velocity = np.random.uniform(3, 40)
        vehicle.fmi2SetReal([0, 1], [steering_angle, velocity])

        X_current, Y_current = [], []
        time_points = np.arange(t_start, t_final, step_size)
        for t in time_points:
            vehicle.fmi2DoStep(t, step_size, True)
            current_state = vehicle.state.copy()
            control_inputs = [vehicle.u[0], vehicle.u[1]]
            X_current.append(np.hstack((current_state, control_inputs)))
            if len(X_current) > 1:
                Y_current.append(current_state)

        X_all.append(np.array(X_current[:-1]))
        Y_all.append(np.array(Y_current))

    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)

    np.save(os.path.join(data_dir, f'X_{filename}.npy'), X_all)
    np.save(os.path.join(data_dir, f'Y_{filename}.npy'), Y_all)
    return X_all, Y_all

# 主训练和评估函数
def train_and_evaluate(train_simulations, test_simulations, step_size):
    try:
        start_time = time.time()
        # 数据生成
        train_filename = f"train_{train_simulations}_{step_size}"
        test_filename = f"test_{test_simulations}_{step_size}"
        X_train, Y_train = generate_data(train_simulations, step_size, train_filename)
        X_test, Y_test = generate_data(test_simulations, step_size, test_filename)

        # 数据预处理
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        Y_train_scaled = Y_scaler.fit_transform(Y_train)
        X_test_scaled = X_scaler.transform(X_test)
        Y_test_scaled = Y_scaler.transform(Y_test)

        # 转换为 PyTorch 张量
        # Use GPU
        # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).cuda()
        # Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32).cuda()
        # X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).cuda()
        # Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32).cuda()

        # Use CPU
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)


        # 定义 GP 模型
        num_tasks = Y_train_tensor.shape[1]
        # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).cuda()
        # model = MultiOutputGPModel(X_train_tensor, Y_train_tensor, likelihood).cuda()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = MultiOutputGPModel(X_train_tensor, Y_train_tensor, likelihood)
        # 训练模型
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(50):
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = -mll(output, Y_train_tensor)
            loss.backward()
            optimizer.step()

        print(f"Final training loss after 50 steps: {loss.item()}")

        # 评估模型
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            Y_pred = likelihood(model(X_test_tensor)).mean.cpu().numpy()

        # 反标准化结果
        Y_pred_orig = Y_scaler.inverse_transform(Y_pred)
        Y_test_orig = Y_scaler.inverse_transform(Y_test_tensor.cpu().numpy())

        # 计算性能指标
        mse = mean_squared_error(Y_test_orig, Y_pred_orig)
        mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
        r2 = r2_score(Y_test_orig, Y_pred_orig)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 记录结果
        write_log(f"Train Simulations: {train_simulations}, Test Simulations: {test_simulations}, Step Size: {step_size}")
        write_log(f"MSE: {mse}, MAE: {mae}, R²: {r2}, Time: {elapsed_time:.2f} seconds")

        return train_simulations, test_simulations, step_size, mse, mae, r2

    except Exception as e:
        write_log(f"Error with Train Simulations: {train_simulations}, Test Simulations: {test_simulations}, Step Size: {step_size}")
        write_log(str(e))
        return None


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for train_simulations in train_simulations_range:
            for test_simulations in test_simulations_range:
                for step_size in step_size_range:
                    # futures.append(executor.submit(train_and_evaluate, train_simulations, test_simulations, step_size))
                    # 只在日志中没出现过的组合才执行
                    if not is_already_completed(train_simulations, test_simulations, step_size):
                        futures.append(
                            executor.submit(
                                train_and_evaluate,
                                train_simulations,
                                test_simulations,
                                step_size
                            )
                        )
        for future in futures:
            future.result()  # 确保所有任务完成

# # 输出最佳结果
# if best_result:
#     write_log(f"Best Result - Train Simulations: {best_result[0]}, Test Simulations: {best_result[1]}, Step Size: {best_result[2]}")
#     write_log(f"MSE: {best_result[3]}, MAE: {best_result[4]}, R²: {best_result[5]}")
