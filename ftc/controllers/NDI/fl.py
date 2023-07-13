""" References
[1] F. L. Lewis, A. Das, and K. Subbarao, “Dynamic inversion with zero-dynamics stabilisation for quadrotor control,” IET Control Theory Appl., vol. 3, no. 3, pp. 303–314, Mar. 2009, doi: 10.1049/iet-cta:20080002.
"""

import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan


class FLController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")
        """ outer-loop control
        Objective: Internal dynamics (x,y) tracking control
        States:
            pos[0:2]: horizontal position (x,y)
            posd[0:2]: desired horizontal position (xd, yd)
        """
        xy = pos[0:2]
        xyd = posd[0:2]
        xy_dot = vel[0:2]
        xyd_dot = posd_dot[0:2]
        xyd_2dot = np.zeros((2,1))
        e, e_dot = xy - xyd, xy_dot - xyd_dot

        C1 = 1 * np.diag((1,1))
        C2 = 1 * np.diag((1,1))

        # outer loop virtual control input
        v_o = (xyd_2dot - C1 @ e - C2 @ e_dot) / env.plant.g

        # angd
        angd = np.vstack((v_o[1], -v_o[0], 0))


        """ inner-loop control
        Objective: vertical position (z) and angle (phi, theta, psi) tracking control
        States:
            pos[2]: vertical position
            posd[2]: desired vertical position
            ang: Euler angle
            angd: desired Euler angle
        """
        y1 = np.vstack((pos[2], ang))
        yd = np.vstack((posd[2], angd))
        y1_dot = np.vstack((vel[2], omega))
        yd_dot = np.vstack((posd_dot[2], 0, 0, 0))
        yd_2dot = np.zeros((4,1))

        e1 = y1 - yd
        e1_dot = y1_dot - yd_dot

        M_h = np.vstack((env.plant.g, -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)))
        E_h = np.zeros((4,4))
        E_h[0,0] = quat2dcm(quat).T[2,2] / env.plant.m
        E_h[1:4, 1:4] = env.plant.Jinv

        # Gain tuning
        K1 = 1 * np.diag((10,10,10,10))
        K2 = 1 * np.diag((10,10,10,10))

        # Feedback Linearized virtual input
        v_h = yd_2dot - K1 @ e1 - K2 @ e1_dot

        # Desired inner-loop input
        u_h = np.linalg.inv(E_h) @ (-M_h + v_h)
        forces = u_h
        forces[0] = -forces[0]

        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
        }

        return forces, controller_info
