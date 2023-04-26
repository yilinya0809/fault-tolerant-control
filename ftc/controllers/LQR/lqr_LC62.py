import fym
import numpy as np
from fym.utils.rot import quat2angle


class LQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        (pos_trim, vel_trim, quat_trim, omega_trim) = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.plant.u_trims_vtol
        pusher_trim, dels_trim = env.plant.u_trims_fixed

        self.x_trims = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9

        A, B = env.plant.lin_model(self.x_trims, self.u_trims, ptrb)
        self.K, *_ = fym.agents.LQR.clqr(A, B, env.Q, env.R)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = np.zeros((3, 1))
        omegad = np.zeros((3, 1))
        x_ref = np.vstack((posd, veld, angd, omegad))

        ctrls0 = -self.K.dot(x - x_ref) + self.u_trims
        ctrls = env.plant.saturate(ctrls0)

        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info


class LQR_FMController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        # VTOL mode
        (pos_trim, vel_trim, quat_trim, omega_trim) = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_HV
        pusher_trim, dels_trim = env.u_trims_fixed_HV

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((rotor_trim, pusher_trim, dels_trim))
        self.FM_trims_HV = env.plant.get_FM(
            pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims_HV
        )

        ptrb = 1e-9
        A_HV, B_HV = env.plant.lin_model_FM(self.x_trims_HV, self.FM_trims_HV, ptrb)
        self.K_HV, *_ = fym.agents.LQR.clqr(A_HV, B_HV, env.Q_HV, env.R_HV)

        # FW mode
        (pos_trim, vel_trim, quat_trim, omega_trim) = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))
        self.FM_trims_FW = env.plant.get_FM(
            pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims_FW
        )

        A_FW, B_FW = env.plant.lin_model_FM(self.x_trims_FW, self.FM_trims_FW, ptrb)
        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW, env.Q_FW, env.R_FW)


    def get_control(self, t, env):
        mode = env.get_ref(t, "mode")[0]

        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))
        
        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = np.zeros((3, 1))
        omegad = np.zeros((3, 1))
        x_ref = np.vstack((posd, veld, angd, omegad))

        # desired virtual input FM
        if mode == "VTOL":
            K = self.K_HV
            x_trims = self.x_trims_HV
            u_trims = self.u_trims_HV
            FM_trims = self.FM_trims_HV
        elif mode == "FW":
            K = self.K_FW
            x_trims = self.x_trims_FW
            u_trims = self.u_trims_FW
            FM_trims = self.FM_trims_FW

        FM_ctrl = -K.dot(x - x_ref) + FM_trims
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return FM_ctrl, controller_info
