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
        self.tau = 0.05
        # self.tau1 = 0.05
        # self.tau2 = 0.01
        self.lpf_ang = fym.BaseSystem(np.zeros((3, 1)))
        # self.lpf_r = fym.BaseSystem(np.zeros((6, 1)))
        # self.lpf_p = fym.BaseSystem(np.zeros((2, 1)))
        
        """ Extended State Observer """
        # self.obsv = fym.BaseSystem(np.zeros((6, 1)))
        # self.L = np.vstack((40 * np.eye(3), 400 * np.eye(3)))
        n = 3  # (n-1)-order derivative of disturbance
        l = 3  # observer output dimension
        self.obsv = fym.BaseSystem(np.zeros((l * (n + 1), 1)))
        self.B = np.zeros((l * (n + 1), l))
        self.C = np.zeros((l * (n + 1), l)).T
        self.B[0:l, 0:l] = np.eye(l)
        self.C[0:l, 0:l] = np.eye(l)
        self.A = np.eye(l * (n + 1), l * (n + 1), l)
        wb = 20
        llist = []
        if n == 3:
            clist = np.array([4, 6, 4, 1])
            llist = clist * wb ** np.array([1, 2, 3, 4])
        elif n == 2:
            clist = np.array([3, 3, 1])
            llist = clist * wb ** np.array([1, 2, 3])
        elif n == 1:
            clist = np.array([2, 1])
            llist = clist * wb ** np.array([1, 2])
        L = []
        for lval in llist:
            L.append(lval * np.eye(l))
        self.L = np.vstack(L)

        self.K1 = np.diag((20, 400, 20))
        self.K2 = np.diag((20, 200, 20))



    def get_control(self, t, env, action):
        _, _, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        # ang = np.clip(ang0, -self.ang_lim, self.ang_lim)

        Frd, Fpd, thetad = np.ravel(action)
        # Frd = - env.plant.g * env.plant.m
        # Fpd = 0
        # thetad = 0.5 * np.sin(t)

        angd = np.vstack((0, thetad, 0))
        ang_f = self.lpf_ang.state
        # omegad = self.lpf_ang.dot = -(ang_f - angd) / self.tau
        self.lpf_ang.dot = -(ang_f - ang) / self.tau
        omegad = np.zeros((3, 1))

        # f = -np.cross(omega, env.plant.J @ omega, axis=0)
        f = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)
        g = env.plant.Jinv
        
        # K1 = np.diag((20, 40, 20))
        # K2 = np.diag((10, 20, 10))

        nui_N = np.linalg.inv(g) @ (-f - self.K1 @ (ang - angd) - self.K2 @ (omega - omegad))
        dhat = self.obsv.state[3:6]
        nui_E = -dhat

        """ GESO control """
        nui = nui_N + nui_E

        """ NDI control """
        # nui = nui_N

        self.u_star = nui + np.linalg.inv(g) @ f

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
        self.set_dot(t, env)


        controller_info = {
            "Frd": Frd,
            "Fpd": Fpd,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
            "dhat": dhat,
        }

        return ctrls, controller_info

    def set_dot(self, t, env):
        _, _, _, omega = env.plant.observe_list()
        y = env.plant.J @ omega
        
        x_hat = self.obsv.state
        y_hat = self.C @ x_hat

        self.obsv.dot = self.A @ x_hat + self.B @ self.u_star + self.L @ (y - y_hat)


class NDIController(GESOController):
    def get_control(self, t, env, action):
        _, _, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        # ang = np.clip(ang0, -self.ang_lim, self.ang_lim)

        Frd, Fpd, thetad = np.ravel(action)
        # Frd = - env.plant.g * env.plant.m
        # Fpd = 0
        # thetad = 0.5 * np.sin(t)

        angd = np.vstack((0, thetad, 0))
        ang_f = self.lpf_ang.state
        # omegad = self.lpf_ang.dot = -(ang_f - angd) / self.tau1
        self.lpf_ang.dot = -(ang_f - ang) / self.tau
        omegad = np.zeros((3, 1))

        # f = -np.cross(omega, env.plant.J @ omega, axis=0)
        f = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)
        g = env.plant.Jinv

        # K1 = np.diag((20, 60, 20))
        # K2 = np.diag((10, 30, 10))

        nui_N = np.linalg.inv(g) @ (-f - self.K1 @ (ang - angd) - self.K2 @ (omega - omegad))
        dhat = self.obsv.state[3:6]
        nui_E = -dhat

        """ GESO control """
        # nui = nui_N + nui_E

        """ NDI control """
        nui = nui_N

        self.u_star = nui + np.linalg.inv(g) @ f

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
        self.set_dot(t, env)


        controller_info = {
            "Frd": Frd,
            "Fpd": Fpd,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
            "dhat": dhat,
        }

        return ctrls, controller_info


    
# class NDIController(fym.BaseEnv):
#     def __init__(self, env):
#         super().__init__()
#         self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
#         self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
#         cr, self.cr_th = 0.0338, 130  # tq / th, th / rcmds
#         self.B_r2f = np.array(
#             (
#                 [-1, -1, -1, -1, -1, -1],
#                 [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
#                 [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
#                 [-cr, cr, -cr, cr, cr, -cr],
#             )
#         )
#         self.cp_th = 70
#         self.ang_lim = env.ang_lim
#         self.tau = 0.05
#         # self.tau1 = 0.05
#         # self.tau2 = 0.01
#         self.lpf_ang = fym.BaseSystem(np.zeros((3, 1)))
#         # self.lpf_r = fym.BaseSystem(np.zeros((6, 1)))
#         # self.lpf_p = fym.BaseSystem(np.zeros((2, 1)))
        
#         """ Extended State Observer """
#         # self.obsv = fym.BaseSystem(np.zeros((6, 1)))
#         # self.L = np.vstack((40 * np.eye(3), 400 * np.eye(3)))
#         n = 1  # (n-1)-order derivative of disturbance
#         l = 3  # observer output dimension
#         self.obsv = fym.BaseSystem(np.zeros((l * (n + 1), 1)))
#         self.B = np.zeros((l * (n + 1), l))
#         self.C = np.zeros((l * (n + 1), l)).T
#         self.B[0:l, 0:l] = np.eye(l)
#         self.C[0:l, 0:l] = np.eye(l)
#         self.A = np.eye(l * (n + 1), l * (n + 1), l)
#         wb = 20
#         llist = []
#         if n == 3:
#             clist = np.array([4, 6, 4, 1])
#             llist = clist * wb ** np.array([1, 2, 3, 4])
#         elif n == 2:
#             clist = np.array([3, 3, 1])
#             llist = clist * wb ** np.array([1, 2, 3])
#         elif n == 1:
#             clist = np.array([2, 1])
#             llist = clist * wb ** np.array([1, 2])
#         L = []
#         for lval in llist:
#             L.append(lval * np.eye(l))
#         self.L = np.vstack(L)
#         self.K1 = np.diag((20, 60, 20))
#         self.K2 = np.diag((10, 30, 10))

#     def get_control(self, t, env, action):
#         _, _, quat, omega = env.plant.observe_list()
#         ang = np.vstack(quat2angle(quat)[::-1])
#         # ang = np.clip(ang0, -self.ang_lim, self.ang_lim)

#         Frd, Fpd, thetad = np.ravel(action)
#         # Frd = - env.plant.g * env.plant.m
#         # Fpd = 0
#         # thetad = 0.5 * np.sin(t)

#         angd = np.vstack((0, thetad, 0))
#         ang_f = self.lpf_ang.state
#         # omegad = self.lpf_ang.dot = -(ang_f - angd) / self.tau1
#         self.lpf_ang.dot = -(ang_f - ang) / self.tau
#         omegad = np.zeros((3, 1))

#         # f = -np.cross(omega, env.plant.J @ omega, axis=0)
#         f = -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)
#         g = env.plant.Jinv

#         # K1 = np.diag((20, 60, 20))
#         # K2 = np.diag((10, 30, 10))

#         nui_N = np.linalg.inv(g) @ (-f - self.K1 @ (ang - angd) - self.K2 @ (omega - omegad))
#         dhat = self.obsv.state[3:6]
#         nui_E = -dhat

#         """ GESO control """
#         # nui = nui_N + nui_E

#         """ NDI control """
#         nui = nui_N

#         self.u_star = nui + np.linalg.inv(g) @ f

#         nu = np.vstack((Frd, nui))
#         th_r = np.linalg.pinv(self.B_r2f) @ nu
#         rcmds = th_r / self.cr_th

#         th_p = Fpd / 2
#         pcmds = th_p / self.cp_th * np.ones((2, 1))
        
#         # rcmds_f = self.lpf_r.state
#         # self.lpf_r.dot = -(rcmds_f - rcmds) / self.tau2
        
#         # pcmds_f = self.lpf_p.state
#         # self.lpf_p.dot = -(pcmds_f - pcmds) / self.tau2

#         dels = np.zeros((3, 1))
#         ctrls = np.vstack((rcmds, pcmds, dels))
#         # ctrls = np.vstack((rcmds_f, pcmds, dels))
#         self.set_dot(t, env)


#         controller_info = {
#             "Frd": Frd,
#             "Fpd": Fpd,
#             "angd": angd,
#             "omegad": omegad,
#             "ang": ang,
#             "dhat": dhat,
#         }

#         return ctrls, controller_info

#     def set_dot(self, t, env):
#         _, _, _, omega = env.plant.observe_list()
#         y = env.plant.J @ omega
        
#         x_hat = self.obsv.state
#         y_hat = self.C @ x_hat

#         self.obsv.dot = self.A @ x_hat + self.B @ self.u_star + self.L @ (y - y_hat)
