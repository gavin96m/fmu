import numpy as np
import joblib
import os

# 数据文件夹路径
filename_suffix = 'no_noise_from_fmu'
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

# 加载训练数据
X = np.load(os.path.join(data_dir, f'X_{filename_suffix}.npy'))  # 输入数据
Y = np.load(os.path.join(data_dir, f'Y_{filename_suffix}.npy'))  # 输出目标

# 加载保存的 GPR 模型
model_path = os.path.join(data_dir, 'gpr_ks_multi_output_model.pkl')
gpr_models = joblib.load(model_path)

# 构造 X_test，使用原始的第一行作为初始状态
initial_state = X[0, :5]  # 车辆的初始状态：x_pos, y_pos, delta, v, psi
control_input = X[0, 5:]  # 控制输入：u_delta, u_v
print(initial_state)
print("---")
print(control_input)

# 多步预测
num_steps = Y.shape[0]  # 使用与原始数据相同的步数
predictions = []

current_state = initial_state

for _ in range(num_steps):
    # 构造输入数据 (current_state + control_input)
    X_test = np.hstack((current_state, control_input)).reshape(1, -1)

    # 初始化一个列表来存储每个输出变量的预测值
    next_state = []

    # 对于每个输出变量，使用对应的 GPR 模型进行预测
    for i, gpr in enumerate(gpr_models):
        y_pred = gpr.predict(X_test)
        next_state.append(y_pred[0])  # 取出标量

    # 将预测的状态添加到预测列表中
    predictions.append(next_state)

    # 更新 current_state 为下一时间步的状态
    current_state = next_state

# 将预测结果转换为 (num_steps, n_outputs) 的形状
predictions = np.array(predictions)

# 保存预测结果
predicted_data_path = os.path.join(data_dir, 'Y_predicted.npy')
np.save(predicted_data_path, predictions)

# 保存为 CSV 文件
np.savetxt(os.path.join(data_dir, 'Y_predicted.csv'), predictions, delimiter=',',
           header='x_pos_next,y_pos_next,delta_next,v_next,psi_next', comments='')

# 进行对比
print(f"真实数据: \n{Y[:5]}")  # 打印前5行真实数据进行对比
print(f"预测数据: \n{predictions[:5]}")
