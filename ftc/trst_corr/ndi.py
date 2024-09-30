""" References
[1] F. L. Lewis, A. Das, and K. Subbarao, “Dynamic inversion with zero-dynamics stabilisation for quadrotor control,” IET Control Theory Appl., vol. 3, no. 3, pp. 303–314, Mar. 2009, doi: 10.1049/iet-cta:20080002.
"""

import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan


class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
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
        self.K1 = np.diag((40, 30, 30))
        self.K2 = np.diag((40, 30, 30))

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        zd, Vxd, Vzd, Frd, Fpd, thetad = env.get_ref(t)
        angd = np.vstack((0, thetad, 0))
        omegad = np.zeros((3, 1))
        f = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)
        g = env.plant.Jinv
        nu = np.vstack(
            (
                -Frd,
                np.linalg.inv(g)
                @ (-f - self.K1 @ (ang - angd) - self.K2 @ (omega - omegad)),
            )
        )

        # control input
        th_r = np.linalg.pinv(self.B_r2f) @ nu
        rcmds = th_r / self.cr_th

        th_p = Fpd / 2
        pcmds = th_p / self.cp_th * np.ones((2, 1))

        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        controller_info = {
            "Frd": -Frd,
            "Fpd": Fpd,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info
