import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 假设已经导入了以下模块和类
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks

# 动力学模型的函数
def func_KS(x, t, u, p):
    return vehicle_dynamics_ks(x, u, p)

class VehicleKSModel:
    def __init__(self):
        self.params = parameters_vehicle1()
        # 0 (x-position)：车辆在全球坐标系下的初始横向位置（单位：米）。
        # 0 (y-position)：车辆在全球坐标系下的初始纵向位置（单位：米）。
        # 0.1 (steering angle or delta)：车辆前轮的转向角度，通常以弧度为单位（该值为0.1弧度，约等于5.7度）。
        # 15 (velocity or speed)：车辆初始的纵向速度，通常以米/秒为单位。
        # 0 (heading angle or yaw angle, ps
        self.state = init_ks([0, 0, 0.1, 15, 0])  # 初始状态
        self.t_step = 0.05  # 减小时间步长以提高精度

    def fmi2SetReal(self, steering_angle, velocity):
        self.u = [steering_angle, velocity]

    def fmi2DoStep(self, current_time, step_size):
        t = np.arange(current_time, current_time + step_size, self.t_step)
        state_trajectory = odeint(func_KS, self.state, t, args=(self.u, self.params))
        self.state = state_trajectory[-1]
        # print(f"Time: {current_time} to {current_time + step_size}, State: {self.state}")  # 打印每步状态
        return state_trajectory

# 使用示例
vehicle = VehicleKSModel()
vehicle.fmi2SetReal(-0.05, 15)  # 设置初始转向角度和速度
trajectory = []
total_time = 15  # 减少总时间以分析每步结果
step_size = 0.5  # 减小步长
time_points = np.arange(0, total_time, step_size)
print(time_points)
for t in time_points:
    states = vehicle.fmi2DoStep(t, step_size)
    print(type(states))
    print(len(states[0]))
    trajectory.extend(states)

trajectory = np.array(trajectory)
plt.figure(figsize=(10, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='Vehicle Trajectory')
plt.title('Vehicle Trajectory over Time')
plt.xlabel('Position X (m)')
plt.ylabel('Position Y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

