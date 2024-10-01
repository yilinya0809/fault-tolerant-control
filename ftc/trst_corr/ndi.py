import fym
import numpy as np
from fym.utils.rot import angle2dcm, quat2angle, quat2dcm
from numpy import cos, sin, tan


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
        self.K2 = np.diag((10, 10))
        self.K3 = np.diag((10, 50, 10))
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

        th_p = Fpd / 2
        pcmds = th_p / self.cp_th * np.ones((2, 1))

        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        controller_info = {
            "Frd": Frd,
            "Fpd": Fpd,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info
