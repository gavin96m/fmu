import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np

# 加载数据
filename_suffix = 'no_noise_from_fmu'
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

X = np.load(os.path.join(data_dir, f'X_{filename_suffix}.npy'))
Y = np.load(os.path.join(data_dir, f'Y_{filename_suffix}.npy'))

# 对数据进行子采样，减少训练样本数量
X_sampled, _, Y_sampled, _ = train_test_split(X, Y, train_size=5000, random_state=42)

# 数据预处理
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X_sampled)
Y_scaled = Y_scaler.fit_transform(Y_sampled)

# 将数据转换为张量
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# 定义 GPyTorch 模型
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

# 定义模型的似然函数
num_tasks = Y_train.shape[1]  # 任务数量
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
model = MultiOutputGPModel(X_train, Y_train, likelihood)

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
likelihood = likelihood.to(device)
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)

# 训练模型
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, Y_train)
    loss.backward()
    print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}')
    optimizer.step()

# 评估模型
model.eval()
likelihood.eval()

with torch.no_grad():
    Y_pred = likelihood(model(X_test))
    Y_pred_mean = Y_pred.mean.cpu().numpy()

# 反标准化预测结果
Y_pred_orig = Y_scaler.inverse_transform(Y_pred_mean)
Y_test_orig = Y_scaler.inverse_transform(Y_test.cpu().numpy())

# 评估模型性能
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(Y_test_orig, Y_pred_orig)
r2 = r2_score(Y_test_orig, Y_pred_orig)
mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# 逐个输出变量计算误差
mse_per_output = mean_squared_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')
mae_per_output = mean_absolute_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')

for i in range(Y_pred_orig.shape[1]):
    print(f"输出变量 {i}: MSE = {mse_per_output[i]}, MAE = {mae_per_output[i]}")
