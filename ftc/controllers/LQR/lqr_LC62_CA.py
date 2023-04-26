import fym
import numpy as np
import cvxpy as cp
from fym.utils.rot import quat2angle

def constrained_lasso(b, nu_d):
    _, dim_u = b.shape
    u = cp.Variable((dim_u,1))
    objective = cp.Minimize(cp.norm(u, 1))
    constraints = [
        nu_d == b @ u,
    ]
    prob = cp.Problem(objective, constraints)
    _ = prob.solve()
    return (u.value, prob.status)


def penalty_lasso(b, nu_d, penalty_weight=1e3):
    _, dim_u = b.shape
    u = cp.Variable((dim_u,1))
    objective = cp.Minimize(
        cp.sum_squares(nu_d - b @ u)
        + penalty_weight*cp.norm(u, 1)
    )
    constraints = []
    prob = cp.Problem(objective, constraints)
    _ = prob.solve(solver=cp.ECOS)
    return u.value



class LQR_binaryController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        # VTOL mode
        (pos_trim, vel_trim, quat_trim, omega_trim) = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_HV
        pusher_trim, dels_trim = env.u_trims_fixed_HV

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9
        A_HV, B_HV = env.plant.lin_model(self.x_trims_HV, self.u_trims_HV, ptrb)
        self.K_HV, *_ = fym.agents.LQR.clqr(A_HV, B_HV[:, :6], env.Q_HV, env.R_HV)

        # FW mode
        (pos_trim, vel_trim, quat_trim, omega_trim) = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))

        A_FW, B_FW = env.plant.lin_model(self.x_trims_FW, self.u_trims_FW, ptrb)
        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW[:, 6:], env.Q_FW, env.R_FW)

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

        if mode == "VTOL":
            w_r = 1
            x_trims = self.x_trims_HV
            u_trims = self.u_trims_HV
        elif mode == "FW":
            w_r = 0
            x_trims = self.x_trims_FW
            u_trims = self.u_trims_FW

        K = np.concatenate((w_r * self.K_HV, (1 - w_r) * self.K_FW), axis=0)

        ctrls0 = -K.dot(x - x_ref) + u_trims
        ctrls = env.plant.saturate(ctrls0)
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info


class LQR_PIController(fym.BaseEnv):
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

    def control_effectiveness(self, t, env):
        # control effectiveness matrix B
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        r1 = 130  # r1 = th_r / rcmds
        r2 = 0.0338  # r2 = tq_r / th_r
        B_r2FM = r1 * np.array(
            (
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-r2, r2, -r2, r2, r2, -r2],
            )
        )

        p1 = 70  # p1 = th_p / pcmds
        p2 = 0.0835  # p2 = tq_p / th_p
        B_p2FM = p1 * np.array(
            (
                [1, 1],
                [0, 0],
                [0, 0],
                [p2, -p2],
                [0, 0],
                [0, 0],
            )
        )

        mode = env.get_ref(t, "mode")[0]
        if mode == "VTOL":
            x_trims = self.x_trims_HV
        elif mode == "FW":
            x_trims = self.x_trims_FW

        pos = x_trims[0:3]
        vel = x_trims[3:6]
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

        B_dels2FM = qbar * np.array(
            (
                [0, 0, 0],
                [0, 0, S * Cy_del_R],
                [0, 0, 0],
                [S2 * b * Cll_del_A, 0, S * b * Cll_del_R],
                [S * c * Cm_del_A, S * c * Cm_del_E, 0],
                [St * d * Cn_del_A, 0, St * d * Cn_del_R],
            )
        )

        B = np.hstack((B_r2FM, B_p2FM, B_dels2FM))
        return B

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
            K = self.K_HV
            x_trims = self.x_trims_HV
            u_trims = self.u_trims_HV
            FM_trims = self.FM_trims_HV
        elif mode == "FW":
            K = self.K_FW
            x_trims = self.x_trims_FW
            u_trims = self.u_trims_FW
            FM_trims = self.FM_trims_FW

        FM_ctrls = -K.dot(x - x_ref)

        # control effectiveness matrix
        B_u2FM = self.control_effectiveness(t, env)

        ctrls0 = np.linalg.pinv(B_u2FM).dot(FM_ctrls) + u_trims
        ctrls = env.plant.saturate(ctrls0)
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info


class LQR_L1normController(fym.BaseEnv):
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
            

    
    def control_effectiveness(self, t, env):
        # control effectiveness matrix B
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        r1 = 130  # r1 = th_r / rcmds
        r2 = 0.0338  # r2 = tq_r / th_r
        B_r2FM = r1 * np.array(
            (
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-r2, r2, -r2, r2, r2, -r2],
            )
        )

        p1 = 70  # p1 = th_p / pcmds
        p2 = 0.0835  # p2 = tq_p / th_p
        B_p2FM = p1 * np.array(
            (
                [1, 1],
                [0, 0],
                [0, 0],
                [p2, -p2],
                [0, 0],
                [0, 0],
            )
        )

        # mode = env.get_ref(t, "mode")[0]
        # if mode == "VTOL":
        #     x_trims = self.x_trims_HV
        # elif mode == "FW":
        #     x_trims = self.x_trims_FW

        pos, vel, quat, omega = env.plant.observe_list()
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

        B_dels2FM = qbar * np.array(
            (
                [0, 0, 0],
                [0, 0, S * Cy_del_R],
                [0, 0, 0],
                [S2 * b * Cll_del_A, 0, S * b * Cll_del_R],
                [S * c * Cm_del_A, S * c * Cm_del_E, 0],
                [St * d * Cn_del_A, 0, St * d * Cn_del_R],
            )
        )

        B = np.hstack((B_r2FM, B_p2FM, B_dels2FM))
        return B

    def allocation(self, B, nu_d):
        """
        L1 norm optimization-based control allocation

        Input:
            small pertubation of virtual input nu_d (6x1)
            control effectiveness B (6x11)
        Output:
            small pertubation of control input commands (11x1)
        """
        # u = self.constrained_lasso(B, nu_d)
        u = penalty_lasso(B, nu_d)
        return u

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = np.zeros((3, 1))
        omegad = np.zeros((3, 1))
        x_ref = np.vstack((posd, veld, angd, omegad))

        mode = env.get_ref(t, "mode")[0]
        if mode == "VTOL":
            K = self.K_HV
            x_trims = self.x_trims_HV
            u_trims = self.u_trims_HV
        elif mode == "FW":
            K = self.K_FW
            x_trims = self.x_trims_FW
            u_trims = self.u_trims_FW

        nu_d = -K.dot(x - x_ref)
        B = self.control_effectiveness(t, env)
        del_u = self.allocation(B, nu_d)
        ctrls0 = u_trims + del_u
        ctrls = env.plant.saturate(ctrls0)
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
            # "ctrls0": ctrls0,
        }

        return ctrls, controller_info
