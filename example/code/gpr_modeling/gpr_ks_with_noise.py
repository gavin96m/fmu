import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
from tqdm import tqdm

# 加载数据
filename_suffix = 'with_noise_from_fmu'
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

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 调整核函数的参数边界
kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

# 使用 MultiOutputRegressor 包装 GPR 进行多输出建模
gpr = MultiOutputRegressor(GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # 增加重启次数
    normalize_y=False
), n_jobs=-1)

# 训练模型
print("Training the GPR model...")
gpr.fit(X_train, Y_train)

# 模型预测
print("Predicting with the GPR model...")
Y_pred = np.zeros(Y_test.shape)

# 使用进度条进行预测
for i, estimator in enumerate(tqdm(gpr.estimators_)):
    Y_pred[:, i] = estimator.predict(X_test)

# 反标准化预测结果
Y_pred_orig = Y_scaler.inverse_transform(Y_pred)
Y_test_orig = Y_scaler.inverse_transform(Y_test)

# 保存模型和缩放器
model_data = {
    'model': gpr,
    'X_scaler': X_scaler,
    'Y_scaler': Y_scaler
}

# 保存模型
model_path = os.path.join(data_dir, 'gpr_ks_with_noise_multi_output_model.pkl')
joblib.dump(model_data, model_path)
print(f"多输出 GPR 模型和缩放器已保存到 {model_path}")


# 评估模型性能
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(Y_test_orig, Y_pred_orig)
r2 = r2_score(Y_test_orig, Y_pred_orig)
mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
rmse = np.sqrt(mse)

# 输出误差指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# 逐个输出变量计算误差
mse_per_output = mean_squared_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')
mae_per_output = mean_absolute_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')

# 输出每个输出变量的误差
for i in range(Y_pred_orig.shape[1]):
    print(f"输出变量 {i}: MSE = {mse_per_output[i]}, MAE = {mae_per_output[i]}")


# 多输出 GPR 模型和缩放器已保存到 D:\Code\fmu\python\fmutest\pythonProject\example\code\gpr_modeling\../data\gpr_ks_with_noise_multi_output_model.pkl
# Mean Squared Error (MSE): 0.006395256920580324
# Root Mean Squared Error (RMSE): 0.07997035025920747
# Mean Absolute Error (MAE): 0.049538805646983694
# R^2 Score: 0.9999579378467416
# 输出变量 0: MSE = 0.009873802641256944, MAE = 0.07861777577909008
# 输出变量 1: MSE = 0.011185899102419005, MAE = 0.0845146855032552
# 输出变量 2: MSE = 9.833291293125658e-07, MAE = 0.0007853414568723225
# 输出变量 3: MSE = 0.010914445927462475, MAE = 0.08293081253135813
# 输出变量 4: MSE = 1.1536026338870614e-06, MAE = 0.0008454129643427315
#
# Process finished with exit code 0
