import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib

# 加载数据

filename_suffix = 'st_with_noise_from_fmu'

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
kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

# 使用 MultiOutputRegressor 包装 GPR 进行多输出建模，移除自定义优化器
gpr = MultiOutputRegressor(GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # 减少重启次数
    normalize_y=False
), n_jobs=-1) # 使用所有可用的 CPU 核

# 训练模型
gpr.fit(X_train, Y_train)

# 模型预测
Y_pred = gpr.predict(X_test)

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
model_path = os.path.join(data_dir, 'gpr_st_with_noise_multi_output_model.pkl')
joblib.dump(model_data, model_path)
print(f"多输出 GPR 模型已保存到 {model_path}")

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


# 多输出 GPR 模型已保存到 D:\Code\fmu\python\fmutest\pythonProject\example\code\gpr_modeling\../data\gpr_st_with_noise_multi_output_model.pkl
# Mean Squared Error (MSE): 0.00011493502702789537
# Root Mean Squared Error (RMSE): 0.010720775486311397
# Mean Absolute Error (MAE): 0.0063524452557598724
# R^2 Score: 0.9999962301807475
# 输出变量 0: MSE = 0.00028199638585387735, MAE = 0.012624910296841877
# 输出变量 1: MSE = 0.00032066603813420976, MAE = 0.01339608605463149
# 输出变量 2: MSE = 1.0914830051170095e-06, MAE = 0.0008351485939837748
# 输出变量 3: MSE = 9.961703035524304e-05, MAE = 0.00799416208328806
# 输出变量 4: MSE = 9.692956334653922e-07, MAE = 0.0007873055515479641
# 输出变量 5: MSE = 9.919710789389109e-05, MAE = 0.008023493412690019
# 输出变量 6: MSE = 1.0078483194639885e-06, MAE = 0.0008060107973359214
#
# Process finished with exit code 0