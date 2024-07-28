import control
import fym
import numpy as np
from fym.utils.rot import quat2angle


class LQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        pos_trim, vel_trim, quat_trim, omega_trim = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9
        A_FW, B_FW = env.plant.lin_model(self.x_trims_FW, self.u_trims_FW, ptrb)

        self.K_FW, *_ = control.lqr(A_FW, B_FW[:, 6:], env.Q, env.R)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        K = np.vstack((np.zeros((6, 12)), self.K_FW))
        ctrls = -K @ (x - self.x_trims_FW) + self.u_trims_FW

        controller_info = {
            "posd": self.x_trims_FW[:3],
            "veld": self.x_trims_FW[3:6],
            "angd": self.x_trims_FW[6:9],
            "ang": ang,
        }

        return ctrls, controller_info
