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
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        self.r1 , r2 = 130, 0.0338  # th_r/rcmds, tq_r/th_r
        self.B_r2FM = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-r2, r2, -r2, r2, r2, -r2],
            )
        )
        self.p1, p2 = 70, 0.0835 # th_p/pcmds, tq_p/th_p
        # self.B_p2FM = np.array(
        #     (
        #         [1, 1],
        #         [0, 0],
        #         [0, 0],
        #         [p2, -p2],
        #         [0, 0],
        #         [0, 0],
        #     )
        # )

        self.mg = env.plant.m * env.plant.g
        self.ang_lim = np.deg2rad(30)
        # self.W = np.diag((200/(self.ang_lim), 1/self.p1))
        self.W = np.diag((600/(self.ang_lim), 1/self.p1))
        self.eo_int = fym.BaseSystem(np.zeros((2, 1)))

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang0 = np.vstack(quat2angle(quat)[::-1])
        # small angle assumption
        ang_min, ang_max = -self.ang_lim, self.ang_lim
        ang = np.clip(ang0, ang_min, ang_max)

        posd, veld = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control
        Objective: horizontal position (x, y) tracking control
        States:
            pos[0:2]: horizontal position
            posd[0:2]: desired horizontal position
        """
        xo, xod = pos[0:2], posd[0:2]
        xo_dot, xod_dot = vel[0:2], veld[0:2]
        eo, eo_dot = xo - xod, xo_dot - xod_dot
        eo_int = self.eo_int.state
         
        Ko1 = 0.01 * np.diag((0, 4))
        Ko2 = 0.01 * np.diag((22, 1))
        Ko3 = 0.001 * np.diag((12, 0))

        # outer-loop virtual control input
        nuo = (-Ko1 @ eo - Ko2 @ eo_dot - Ko3 @ eo_int) * env.plant.m
        phi = nuo[1] / self.mg
        
        # control effectiveness vector
        bo = np.vstack((-self.mg, 1)) 
        Winv = np.linalg.inv(self.W)
        Pw = Winv @ bo @ np.linalg.inv(bo.T @ Winv @ bo)
        uo = nuo[0] * Pw
        theta = uo[0]
        th_p = 0.5 * uo[1] * np.ones((2, 1))

        angd = np.vstack((phi, theta, 0))

        """ inner-loop control
        Objective: vertical position (z) and angle (phi, theta, psi) tracking control
        States:
            pos[2]: vertical position
            posd[2]: desired vertical position
            ang: Euler angle
            angd: desired Euler angle
        """
        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((veld[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        Ki1 = 10 * np.diag((30, 1, 30, 1))
        Ki2 = 10 * np.diag((2, 1, 20, 1))
        f = np.vstack(
            (
                env.plant.g,
                -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0),
            )
        )
        g = np.zeros((4, 4))
        g[0, 0] = quat2dcm(quat).T[2, 2] / env.plant.m
        g[1:4, 1:4] = env.plant.Jinv

        # inner-loop virtual control input
        nui = np.linalg.inv(g) @ (-f - Ki1 @ ei - Ki2 @ ei_dot)
        th_r = np.linalg.pinv(self.B_r2FM) @ nui
        rcmds = th_r / self.r1
        pcmds = th_p / self.p1
        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        self.eo_int.dot = eo_dot
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "ang": ang,
            "Fpd": uo[1],
        }

        return ctrls, controller_info
