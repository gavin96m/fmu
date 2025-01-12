import pandas as pd
import numpy as np
import os


filename_suffix = 'no_noise_from_fmu'
# filename_suffix = 'st_no_noise_from_fmu'
# filename_suffix = 'no_noise_odeint'
# filename_suffix = 'no_noise_solve_ivp'

# 设置 pandas 打印选项
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)        # 设置每行的宽度

# 加载数据

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')
X_all = np.load(os.path.join(data_dir, f'X_{filename_suffix}.npy'))
Y_all = np.load(os.path.join(data_dir, f'Y_{filename_suffix}.npy'))

# 将数据转换为 DataFrame 方便查看统计信息
X_df = pd.DataFrame(X_all, columns=['x_pos', 'y_pos', 'delta', 'v', 'psi', 'u_delta', 'u_v'])
Y_df = pd.DataFrame(Y_all, columns=['x_pos_next', 'y_pos_next', 'delta_next', 'v_next', 'psi_next'])

# X_df = pd.DataFrame(X_all, columns=['x_pos', 'y_pos', 'delta', 'v', 'psi', 'raw_rate', 'slip_angle' , 'u_delta', 'u_v'])
# Y_df = pd.DataFrame(Y_all, columns=['x_pos_next', 'y_pos_next', 'delta_next', 'v_next', 'psi_next', 'raw_rate_next', 'slip_angle_next'])


# 查看统计信息
print(X_df.describe())
print(Y_df.describe())
