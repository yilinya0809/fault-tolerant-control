import fym
import numpy as np
from fym.utils.rot import angle2dcm, quat2angle, quat2dcm
from numpy import cos, sin, tan


# Transition mode controller
class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.m, self.g = env.plant.m, env.plant.g
        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        cr, self.cr_th = 0.0338, 128  # tq / th, th / rcmds
        self.cp_th = 70
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
                [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
                [-cr, cr, -cr, cr, cr, -cr],
            )
        )
        self.K1 = np.diag((0, 100))
        self.K2 = np.diag((10, 20))
        self.K3 = np.diag((10, 100, 10))
        self.K4 = np.diag((10, 10, 10))

    def get_control(self, t, env):
        # current state
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        R = quat2dcm(quat)
        dpos = R.T @ vel
        theta = ang[1, 0]

        # desired state
        zd, veld, thetad = env.get_ref(t)
        angd = np.vstack((0, thetad, 0))
        omegad = np.zeros((3, 1))
        Rd = angle2dcm(0, thetad, 0)
        dpos_d = Rd.T @ veld

        # eliminate y-axis
        pos = np.vstack((pos[0], pos[2]))
        posd = np.vstack((0, zd))
        dpos = np.vstack((dpos[0], dpos[2]))
        dpos_d = np.vstack((dpos_d[0], dpos_d[2]))

        # virtual control input - Fr, Fp
        f1 = np.vstack((0, self.g))
        g1 = np.array([[sin(theta), -cos(theta)], [cos(theta), sin(theta)]]) / (-self.m)

        nu1 = np.linalg.inv(g1) @ (
            -f1 - self.K1 @ (pos - posd) - self.K2 @ (dpos - dpos_d)
        )

        Frd = nu1[0]
        Fpd = nu1[1]

        f2 = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)
        g2 = env.plant.Jinv
        Mrd = np.linalg.inv(g2) @ (
            -f2 - self.K3 @ (ang - angd) - self.K4 @ (omega - omegad)
        )

        # control input
        th_r = np.linalg.pinv(self.B_r2f) @ np.vstack((-Frd, Mrd))
        rcmds = th_r / self.cr_th
        if not np.isclose(np.linalg.norm(self.B_r2f@ th_r- np.vstack((-Frd, Mrd))), 0):
            print(np.linalg.norm(self.B_r2f@ th_r- np.vstack((-Frd, Mrd))))
            breakpoint()

        th_p = Fpd / 2
        pcmds = th_p / self.cp_th * np.ones((2, 1))

        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        controller_info = {
            "posd": np.vstack((0, 0, zd)),
            "veld": veld,
            "Frd": Frd,
            "Fpd": Fpd,
            "th_r": th_r,
            "th_p": th_p,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info


# FW mode controller
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

        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW[:, 6:], env.Q, env.R)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        K = np.vstack((np.zeros((6, 12)), self.K_FW))
        ctrls = -K @ (x - self.x_trims_FW) + self.u_trims_FW

        controller_info = {
            "posd": self.x_trims_FW[0:3],
            "veld": self.x_trims_FW[3:6],
            "Frd": np.array([0.0]),
            "Fpd": env.plant.B_Pusher(ctrls[6:8])[0],
            "angd": self.x_trims_FW[6:9],
            "omegad": np.vstack((0, 0, 0)),
            "ang": ang,
        }

        return ctrls, controller_info
