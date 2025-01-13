import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
# 假设已经导入了以下模块和类
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks

STEERING_ANGLE_REF = 0
VELOCITY_REF = 1

# 动力学模型的函数
def func_KS(x, t, u, p):
    return vehicle_dynamics_ks(x, u, p)

class Model:
    def __init__(self) -> None:
        self.params = parameters_vehicle1()
        self.state_trajectory = []
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.delta = 0.1
        self.speed = 15.0
        self.heading_angle = 0.0

        # 0 (x-position)：车辆在全球坐标系下的初始横向位置（单位：米）。
        # 0 (y-position)：车辆在全球坐标系下的初始纵向位置（单位：米）。
        # 0.1 (steering angle or delta)：车辆前轮的转向角度，通常以弧度为单位（该值为0.1弧度，约等于5.7度）。
        # 15 (velocity or speed)：车辆初始的纵向速度，通常以米/秒为单位。
        # 0 (heading angle or yaw angle, ps
        self.state = init_ks([self.x_pos, self.y_pos, self.delta, self.speed, self.heading_angle])  # 初始状态
        self.t_step = 0.1  # 减小时间步长以提高精度
        self.u = [0.0, 0.0] # Initialize control inputs

        # print("start")
        self.reference_to_attribute = {
            0: "x_pos",
            1: "y_pos",
            2: "delta",
            3: "speed",
            4: "heading_angle",
        }



        self._update_outputs()

    def fmi2DoStep(self, current_time, step_size, no_step_prior):
        num_steps = max(int(step_size / self.t_step), 1)
        t = np.linspace(current_time, current_time + step_size, num_steps + 1)
        # 调用动力学模型进行积分
        state_trajectory = odeint(func_KS, self.state, t, args=(self.u, self.params))
        self.state = state_trajectory[-1]
        self.state_trajectory = state_trajectory  # 保存完整的状态轨迹
        self._update_outputs()
        return Fmi2Status.ok

    def fmi2EnterInitializationMode(self):
        # return Fmi2Status.ok

    def fmi2ExitInitializationMode(self):
        self._update_outputs()
        return Fmi2Status.ok

    def fmi2SetupExperiment(self, start_time, stop_time, tolerance):
        return Fmi2Status.ok

    def fmi2SetReal(self, references, values):
        for ref, val in zip(references, values):
            if ref == STEERING_ANGLE_REF:
                self.u[0] = val
            elif ref == VELOCITY_REF:
                self.u[1] = val
            else:
                raise ValueError(f"Unknown reference: {ref}")

    def fmi2SetInteger(self, references, values):
        return self._set_value(references, values)

    def fmi2SetBoolean(self, references, values):
        return self._set_value(references, values)

    def fmi2SetString(self, references, values):
        return self._set_value(references, values)

    def fmi2GetReal(self, references):
        return self._get_value(references)

    def fmi2GetInteger(self, references):
        return self._get_value(references)

    def fmi2GetBoolean(self, references):
        return self._get_value(references)

    def fmi2GetString(self, references):
        return self._get_value(references)

    def fmi2Reset(self):
        return Fmi2Status.ok

    def fmi2Terminate(self):
        return Fmi2Status.ok

    def fmi2ExtSerialize(self):

        bytes = pickle.dumps(
            (
                self.x_pos,
                self.y_pos,
                self.delta,
                self.speed,
                self.heading_angle,
            )
        )
        return Fmi2Status.ok, bytes

    def fmi2ExtDeserialize(self, bytes) -> int:
        (
            x_pos,
            y_pos,
            delta,
            speed,
            heading_angle,
        ) = pickle.loads(bytes)

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.delta = delta
        self.speed = speed
        self.heading_angle = heading_angle
        self._update_outputs()

        return Fmi2Status.ok

    def _set_value(self, references, values):

        for r, v in zip(references, values):
            setattr(self, self.reference_to_attribute[r], v)

        return Fmi2Status.ok

    def _get_value(self, references):

        values = []

        for r in references:
            values.append(getattr(self, self.reference_to_attribute[r]))

        return Fmi2Status.ok, values

    # def _update_outputs(self):
    #     self.x_pos = self.x_pos
    #     self.y_pos = self.y_pos
    #     self.delta = self.delta
    #     self.speed = self.speed
    #     self.heading_angle = self.heading_angle
    def _update_outputs(self):
        # 从 self.state 中提取当前的状态值
        self.x_pos = self.state[0]
        self.y_pos = self.state[1]
        self.delta = self.state[2]
        self.speed = self.state[3]
        self.heading_angle = self.state[4]


class Fmi2Status:
    """Represents the status of the FMU or the results of function calls.

    Values:
        * ok: all well
        * warning: an issue has arisen, but the computation can continue.
        * discard: an operation has resulted in invalid output, which must be discarded
        * error: an error has ocurred for this specific FMU instance.
        * fatal: an fatal error has ocurred which has corrupted ALL FMU instances.
        * pending: indicates that the FMu is doing work asynchronously, which can be retrived later.

    Notes:
        FMI section 2.1.3

    """

    ok = 0
    warning = 1
    discard = 2
    error = 3
    fatal = 4
    pending = 5


if __name__ == "__main__":
    # 使用示例
    vehicle = Model()
    vehicle.fmi2SetReal([STEERING_ANGLE_REF, VELOCITY_REF], [-0.05, 15])
    trajectory = []
    total_time = 15  # 减少总时间以分析每步结果
    step_size = 0.001 # 减小步长
    time_points = np.arange(0, total_time, step_size)
    for t in time_points:
        vehicle.fmi2DoStep(t, step_size,True)
        trajectory.extend(vehicle.state_trajectory)

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
