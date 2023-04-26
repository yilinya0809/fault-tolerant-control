import fym
import numpy as np
from fym.utils.rot import quat2angle

class LQR_PIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()

        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_HV
        pusher_trim, dels_trim = env.u_trims_fixed_HV

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((rotor_trim, pusher_trim, dels_trim))
        self.FM_trims_HV = env.plant.get_FM(pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims_HV)


        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))
        self.FM_trims_FW = env.plant.get_FM(pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims_FW)

        ptrb = 1e-9

        A_HV, B_HV = env.plant.lin_model_FM(self.x_trims_HV, self.FM_trims_HV, ptrb)
        A_FW, B_FW = env.plant.lin_model_FM(self.x_trims_FW, self.FM_trims_FW, ptrb)

        self.K_HV, *_ = fym.agents.LQR.clqr(A_HV, B_HV, env.Q_HV, env.R_HV)
        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW, env.Q_FW, env.R_FW)

    def get_control(self, t, env):
        # mode from high level controller - char type 
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
            FM_ctrls = -self.K_HV.dot(x - x_ref) + self.FM_trims_HV
        elif mode == "FW":
            FM_ctrls = -self.K_FW.dot(x - x_ref) + self.FM_trims_FW

        # control effectiveness matrix B
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        r = 0.0338 # tq_r / th_r
        self.B_r2FM = np.array((
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
            [-dy2, dy1, dy1, -dy2, -dy2, dy1],
            [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
            [-r, r, -r, r, r, -r],
        ))

        p = 0.0835 # tq_p / th_p
        self.B_p2FM = np.array((
            [1, 1],
            [0, 0],
            [0, 0],
            [p, -p],
            [0, 0],
            [0, 0],
        ))
        

        if mode == "VTOL":
            pos = self.x_trims_HV[0:3]
            vel = self.x_trims_HV[3:6]
        elif mode == "FW":
            pos = self.x_trims_FW[0:3]
            vel = self.x_trims_FW[3:6]

        rho = env.plant.get_rho(-pos[2])
        VT = np.linalg.norm(vel)
        qbar = 0.5 * rho * VT**2
        alp = np.arctan2(vel[2], vel[0])
        CL, CD, CM = env.plant.aero_coeff(alp)
        S, S2, St = env.plant.S, env.plant.S2, env.plant.St
        b, c, d = env.plant.b, env.plant.c, env.plant.d

        Cy_del_R, Cll_del_A, Cll_del_R = (
            env.plant.Cy_del_R,
            env.plant.Cll_del_A,
            env.plant.Cll_del_R,
        )
        Cm_del_A, Cm_del_E, Cn_del_A, Cn_del_R = (
            env.plant.Cm_del_A,
            env.plant.Cm_del_E,
            env.plant.Cn_del_A,
            env.plant.Cn_del_R,
        )

        self.B_dels2FM = qbar * np.array(
            (
                [0, 0, 0],
                [0, 0, S * Cy_del_R],
                [0, 0, 0],
                [S2 * b * Cll_del_A, 0, S * b * Cll_del_R],
                [S * c * Cm_del_A, S * c * Cm_del_E, 0],
                [St * d * Cn_del_A, 0, St * d * Cn_del_R],
            )
        )

        B_u2FM = np.hstack((self.B_r2FM, self.B_p2FM, self.B_dels2FM))
        B_const = qbar * S * np.vstack((-CD, 0, -CL, 0, c*CM, 0))

        # Control Allocation
        ctrls = np.linalg.pinv(B_u2FM).dot(FM_ctrls - B_const)
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info






    
