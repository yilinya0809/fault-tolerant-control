import fym
import numpy as np
from fym.utils.rot import angle2dcm, quat2angle, quat2dcm
from numpy import cos, sin, tan


# Transition mode controller
class Corr_NDIController(fym.BaseEnv):
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
        
    def get_control(self, t, env):
        xd, zd, veld, thetad, mode = env.get_ref(t)
        if mode == "FTC":
            self.K1 = np.diag((10, 100)) # K4
            self.K2 = np.diag((10, 50)) # K3
            self.K3 = np.diag((10, 200, 10)) # K2
            self.K4 = np.diag((10, 20, 10)) # K1
        elif mode == "BTC":
            self.K1 = np.diag((10, 100)) # K4
            self.K2 = np.diag((10, 50)) # K3
            self.K3 = np.diag((10, 5000, 10)) # K2
            self.K4 = np.diag((10, 100, 10)) # K1
   
        # current state
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        R = quat2dcm(quat)
        dpos = R.T @ vel
        theta = ang[1, 0]

        # desired state
        angd = np.vstack((0, thetad, 0))
        omegad = np.zeros((3, 1))
        Rd = angle2dcm(0, thetad, 0)
        dpos_d = Rd.T @ veld

        # eliminate y-axis
        pos = np.vstack((pos[0], pos[2]))
        posd = np.vstack((xd, zd))
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

        th_p = Fpd / 2
        pcmds = th_p / self.cp_th * np.ones((2, 1))

        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        controller_info = {
            "posd": np.vstack((xd, 0, zd)),
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


# FW, HV mode controller
class LQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        # FW
        pos_trim, vel_trim, quat_trim, omega_trim = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9
        A_FW, B_FW = env.plant.lin_model(self.x_trims_FW, self.u_trims_FW, ptrb)

        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW[:, 6:], env.Q_FW, env.R_FW)
        
        # HV
        pos_trim, vel_trim, quat_trim, omega_trim = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_HV
        pusher_trim, dels_trim = env.u_trims_fixed_HV

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((rotor_trim, pusher_trim, dels_trim))

        A_HV, B_HV = env.plant.lin_model(self.x_trims_HV, self.u_trims_HV, ptrb)

        self.K_HV, *_ = fym.agents.LQR.clqr(A_HV, B_HV[:, :6], env.Q_HV, env.R_HV)


    def get_control(self, t, env):
        xd, zd, _, _, mode = env.get_ref(t)
        if mode == "FW":
            w_r = 0
            x_ref = np.vstack((xd, 0, zd, self.x_trims_FW[3:]))
            u_ref = self.u_trims_FW
        elif mode == "HV":
            w_r = 1
            x_ref = np.vstack((xd, 0, zd, self.x_trims_HV[3:]))
            u_ref = self.u_trims_HV

        K = np.concatenate((w_r * self.K_HV, (1 - w_r) * self.K_FW), axis=0)
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        # K = np.vstack((np.zeros((6, 12)), self.K_FW))
        ctrls = -K @ (x - x_ref) + u_ref 

        controller_info = {
            "posd": np.vstack((xd, 0, zd)),
            "veld": x_ref[3:6],
            "angd": x_ref[6:9],
            "omegad": np.vstack((0, 0, 0)),
            "ang": ang,
            "Frd": env.plant.B_VTOL(ctrls[0:6], np.zeros((3, 1)))[0],
            "Fpd": env.plant.B_Pusher(ctrls[6:8])[0],
        }

        return ctrls, controller_info


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

        xd, zd, veld, _, mode = env.get_ref(t)
        posd = np.vstack((xd, 0, zd))
 
        if mode == "FTC":
            Ko1 = 0.01 * np.diag((0, 4))
            Ko2 = 0.01 * np.diag((22, 1))
            Ko3 = 0.001 * np.diag((12, 0))
            Ki1 = 10 * np.diag((30, 1, 30, 1))
            Ki2 = 10 * np.diag((2, 1, 20, 1))
        elif mode == "BTC":
            Ko1 = 0.01 * np.diag((0, 4))
            Ko2 = 0.01 * np.diag((4, 1))
            Ko3 = 0.001 * np.diag((2, 0))
            Ki1 = np.diag((200, 10, 500, 1))
            Ki2 = np.diag((100, 10, 200, 1))


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
            "omegad": np.zeros((3, 1)),
            "Frd": env.plant.B_VTOL(ctrls[0:6], np.zeros((3, 1)))[0],
            "Fpd": env.plant.B_Pusher(ctrls[6:8])[0],

        }

        return ctrls, controller_info
