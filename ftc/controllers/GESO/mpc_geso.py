import casadi as ca
import fym
import numpy as np
from fym.utils.rot import quat2angle

from ftc.models.LC62S import LC62


class GESOController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        cr, self.cr_th = 0.0338, 130  # tq / th, th / rcmds
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
                [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
                [-cr, cr, -cr, cr, cr, -cr],
            )
        )
        self.cp_th = 70
        self.ang_lim = env.ang_lim
        self.tau1 = 0.05
        # self.tau2 = 0.012
        self.lpf_ang = fym.BaseSystem(np.zeros((3, 1)))
        # self.lpf_r = fym.BaseSystem(np.zeros((6, 1)))
        # self.lpf_p = fym.BaseSystem(np.zeros((2, 1)))
        

        self.obsv = fym.BaseSystem(np.zeros((6, 1)))
        self.L = np.vstack((40 * np.eye(3), 400 * np.eye(3)))


    def get_control(self, t, env, action):
        _, _, quat, omega = env.plant.observe_list()
        ang0 = np.vstack(quat2angle(quat)[::-1])
        ang = np.clip(ang0, -self.ang_lim, self.ang_lim)

        Frd, Fpd, thetad = np.ravel(action)
        # Frd = - env.plant.g * env.plant.m
        # Fpd = 0
        # thetad = 0.5 * np.sin(t)

        angd = np.vstack((0, thetad, 0))
        ang_f = self.lpf_ang.state
        omegad = self.lpf_ang.dot = -(ang_f - angd) / self.tau1

        f = -np.cross(omega, env.plant.J @ omega, axis=0)
        # f = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)

        K1 = np.diag((1, 100, 1))
        K2 = np.diag((1, 100, 1))

        nui_N = -f + (-K1 @ (ang - angd) - K2 @ (omega - omegad))
        # nui_N = env.plant.J @ (-f + (-K1 @ (ang - angd) - K2 @ (omega - omegad)))
        dhat = self.obsv.state[3:]
        nui_E = -dhat
        nui = nui_N + nui_E
        self.u_star = nui + f
        # self.u_star = nui + env.plant.J @ f
        nu = np.vstack((Frd, nui))
        th_r = np.linalg.pinv(self.B_r2f) @ nu
        rcmds = th_r / self.cr_th

        th_p = Fpd / 2
        pcmds = th_p / self.cp_th * np.ones((2, 1))
        
        # rcmds_f = self.lpf_r.state
        # self.lpf_r.dot = -(rcmds_f - rcmds) / self.tau2
        
        # pcmds_f = self.lpf_p.state
        # self.lpf_p.dot = -(pcmds_f - pcmds) / self.tau2

        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))
        # ctrls = np.vstack((rcmds_f, pcmds, dels))


        controller_info = {
            "Frd": Frd,
            "Fpd": Fpd,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info

    def set_dot(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        y = env.plant.J @ omega
        
        A = np.eye(6, 6, 3)
        B = np.zeros((6, 3))
        B[0:3, 0:3] = np.eye(3)
        C = np.zeros((3, 6))
        C[0:3, 0:3] = np.eye(3)

        x_hat = self.obsv.state
        y_hat = C @ x_hat

        self.obsv.dot = A @ x_hat + B @ self.u_star + self.L @ (y - y_hat)
