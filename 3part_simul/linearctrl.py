import control
import fym
import numpy as np
from fym.utils.rot import quat2angle
import pyperclip


class LinearCtrl(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()

        # HV
        self.x_trims_HV, self.u_trims_fixed_HV = env.plant.get_trim_fixed(
            fixed={"h": 5, "VT": 0}
        )
        self.u_trims_vtol_HV = env.plant.get_trim_vtol(
            fixed={"x_trims": self.x_trims_HV, "u_trims_fixed": self.u_trims_fixed_HV}
        )
        pos_trim, vel_trim, quat_trim, omega_trim = self.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((self.u_trims_vtol_HV, *self.u_trims_fixed_HV))

        ptrb = 1e-9
        A_HV, B_HV = env.plant.lin_model(self.x_trims_HV, self.u_trims_HV, ptrb)

        self.Q_HV = np.diag([1, 1, 10, 1, 1, 10, 100, 100, 100, 1, 1, 1])
        self.R_HV = 100 * np.diag([1, 1, 1, 1, 1, 1])

        self.K_HV, *_ = control.lqr(A_HV, B_HV[:, :6], self.Q_HV, self.R_HV)
        K = "[\n"
        K += "\n".join("    [" + ", ".join(f"{num:.8e}" for num in row) + "]," for row in self.K_HV)
        K += "\n]"
        pyperclip.copy(K)





    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        x = np.vstack((pos, vel, ang, omega))

        posd = np.zeros((3, 1))
        veld = np.zeros((3, 1))
        angd = np.deg2rad(np.vstack((0, 0, 0)))
        omegad = np.zeros((3, 1))

        x_ref = np.vstack((posd, veld, angd, omegad))
        _ctrls = -self.K_HV @ (x - x_ref)
        ctrls = self.u_trims_HV + np.vstack((_ctrls, np.zeros((5, 1))))

        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "ang": ang,
            "omegad": omegad,
            "ang": ang,
            "K": self.K_HV,
        }

        return ctrls, controller_info
