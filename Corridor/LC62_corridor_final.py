import os

import fym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fym.utils.rot import angle2dcm, angle2quat, quat2angle, quat2dcm
from mpl_toolkits.mplot3d import axes3d
from numpy import cos, sin
from scipy.interpolate import interp1d

from ftc.models.LC62R import LC62R
from ftc.models.LC62S import LC62
from ftc.utils import safeupdate


class LC62_corridor(fym.BaseEnv):
    temp_y = 0.0
    # Aircraft paramters
    dx1 = 0.9325 + 0.049  # dx1 = 0.9815
    dx2 = 0.0725 - 0.049  # dx2 = 0.0235
    dx3 = 1.1725 - 0.049  # dx3 = 1.1235
    dy1 = 0.717 + temp_y
    dy2 = 0.717 + temp_y
    dyp = 2.3 / 2
    dzp = 0.0645

    Ixx = 1.3 * 8.094  # [kg * m^2]
    Iyy = 1.3 * 9.125  # [kg * m^2]
    Izz = 1.3 * 16.8615  # [kg * m^2]
    Ixz = -1.3 * 0.308  # [kg * m^2]

    J = np.array(
        [
            [Ixx, 0, Ixz],
            [0, Iyy, 0],
            [Ixz, 0, Izz],
        ]
    )
    Jinv = np.linalg.inv(J)

    g = 9.81  # [m / sec^2]
    m = 41.97  # [kg]

    S1 = 0.2624  # [m^2]
    S2 = 0.5898  # [m^2]
    S = S1 + S2
    St = 0.06894022  # [m^2]
    c = 0.551  # Main wing chord length [m]
    b = 1.1  # Main wing half span [m]
    d = 0.849  # Moment arm length [m]

    inc = 0  # Wing incidence angle

    # Ele Pitch Moment Coefficient [1 / rad]
    Cm_del_A = 0.001156
    Cm_del_E = -0.676

    # Rolling Moment Coefficient [1 / rad]
    Cll_beta = -0.0518
    Cll_p = -0.4624
    Cll_r = 0.0218
    Cll_del_A = -0.0369 * 5
    Cll_del_R = 0.0026

    # Yawing Moment Coefficient [1 / rad]
    Cn_beta = 0.0866
    Cn_p = -0.0048
    Cn_r = -0.0723
    Cn_del_A = -0.000385
    Cn_del_R = -0.0190

    # Y-axis Force Coefficient [1 / rad]
    Cy_beta = -1.1269
    Cy_p = 0.0
    Cy_r = 0.2374
    Cy_del_R = 0.0534

    """
    cmd: 0 ~ 1
    th_p: pusher thrust [N]
    tq_p: pusher torque [N * m]
    """
    tables = {
        "alp": np.deg2rad(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])),  # [rad]
        "CL": np.array(
            [0.1931, 0.4075, 0.6112, 0.7939, 0.9270, 1.0775, 0.9577, 1.0497, 1.0635]
        ),
        "CD": np.array(
            [0.0617, 0.0668, 0.0788, 0.0948, 0.1199, 0.1504, 0.2105, 0.2594, 0.3128]
        ),
        "Cm": np.array(
            [
                0.0406,
                0.0141,
                -0.0208,
                -0.0480,
                -0.2717,
                -0.4096,
                -0.1448,
                -0.2067,
                -0.2548,
            ]
        ),
        "cmd": np.array(
            [
                0,
                0.2,
                0.255,
                0.310,
                0.365,
                0.420,
                0.475,
                0.530,
                0.585,
                0.640,
                0.695,
                0.750,
            ]
        ),
        "th_p": np.array(
            [
                0,
                1.39,
                4.22,
                7.89,
                12.36,
                17.60,
                23.19,
                29.99,
                39.09,
                46.14,
                52.67,
                59.69,
            ]
        ),
        "tq_p": np.array(
            [0, 0.12, 0.35, 0.66, 1.04, 1.47, 1.93, 2.50, 3.25, 3.83, 4.35, 4.95]
        ),
        "th_r": [-19281, 36503, -992.75, 0],
        "tq_r": [-6.3961, 12.092, -0.3156, 0],
    }

    control_limits = {
        "cmd": (0, 1),
        "dela": np.deg2rad((-10, 10)),
        "dele": np.deg2rad((-10, 10)),
        "delr": np.deg2rad((-10, 10)),
    }
    ENV_CONFIG = {
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.pos = fym.BaseSystem(env_config["init"]["pos"])
        self.vel = fym.BaseSystem(env_config["init"]["vel"])
        self.quat = fym.BaseSystem(env_config["init"]["quat"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])

        self.e3 = np.vstack((0, 0, 1))
        self.plant = LC62()

    def B_Pusher(self, Fp):
        Fx = Fp
        Fy = Fz = 0
        l = 0
        m = -self.dzp * (Fp)
        n = 0
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Fuselage(self, dels, pos, vel, omega):
        rho = self.get_rho(-pos[2])
        u, v, w = np.ravel(vel)
        p, q, r = np.ravel(omega)
        VT = np.linalg.norm(vel)
        alp = np.arctan2(w, u)
        if np.isclose(VT, 0):
            beta = 0
        else:
            beta = np.arcsin(v / VT)
        qbar = 0.5 * rho * VT**2
        Cy_p, Cy_r, Cy_beta = self.Cy_p, self.Cy_r, self.Cy_beta
        Cll_p, Cll_r, Cll_beta = (
            self.Cll_p,
            self.Cll_r,
            self.Cll_beta,
        )
        Cn_p, Cn_r, Cn_beta = (
            self.Cn_p,
            self.Cn_r,
            self.Cn_beta,
        )
        Cy_del_R = self.Cy_del_R
        Cll_del_R, Cll_del_A = self.Cll_del_R, self.Cll_del_A
        Cm_del_E, Cm_del_A = self.Cm_del_E, self.Cm_del_A
        Cn_del_R, Cn_del_A = self.Cn_del_R, self.Cn_del_A
        S, S2, St, b, c, d = self.S, self.S2, self.St, self.b, self.c, self.d
        dela, dele, delr = dels
        CL, CD, CM = self.aero_coeff(alp)
        Fx = -qbar * S * CD
        Fy = qbar * S * (p * Cy_p + r * Cy_r + beta * Cy_beta + delr * Cy_del_R)
        Fz = -qbar * S * CL
        l = (
            qbar
            * b
            * (
                S * (p * Cll_p + r * Cll_r + beta * Cll_beta + delr * Cll_del_R)
                + S2 * dela * Cll_del_A
            )
        )
        m = qbar * c * S * (CM + dele * Cm_del_E + dela * Cm_del_A)
        n = (
            qbar
            * d
            * St
            * (p * Cn_p + r * Cn_r + beta * Cn_beta + delr * Cn_del_R + dela * Cn_del_A)
        )
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Gravity(self, quat):
        l = m = n = 0
        return np.vstack((quat2dcm(quat) @ (self.m * self.g * self.e3), l, m, n))

    def get_rho(self, altitude):
        pressure = 101325 * (1 - 2.25569e-5 * altitude) ** 5.25616
        temperature = 288.14 - 0.00649 * altitude
        return pressure / (287 * temperature)

    def aero_coeff(self, alp):
        _CL = interp1d(
            self.tables["alp"],
            self.tables["CL"],
            kind="linear",
            fill_value="extrapolate",
        )
        _CD = interp1d(
            self.tables["alp"],
            self.tables["CD"],
            kind="linear",
            fill_value="extrapolate",
        )
        _Cm = interp1d(
            self.tables["alp"],
            self.tables["Cm"],
            kind="linear",
            fill_value="extrapolate",
        )
        CL = _CL(alp)
        CD = _CD(alp)
        Cm = _Cm(alp)
        return np.vstack((CL, CD, Cm))

    def get_corr(
        self,
        z0={
            "Fr": 0.0,
            "Fp": 0.0,
            "q": 0.0,
        },
        height=10,
        grid={"VT": np.arange(0, 40, 1), "theta": np.deg2rad(np.arange(-20, 20, 1))},
        method="SLSQP",
        options={"disp": False, "ftol": 1e-10},
        eps=1e-3,
    ):
        z0 = list(z0.values())
        grid = list(grid.values())
        VT_range, theta_range = grid
        bounds = (
            (0, 6 * self.plant.th_r_max),
            (0, 2 * self.plant.th_p_max),
            (0, 10),
        )
        n = np.size(VT_range)
        m = np.size(theta_range)

        cost = np.ones((n, m))
        success = np.zeros((n, m))
        Fr = np.zeros((n, m))
        Fp = np.zeros((n, m))
        q = np.zeros((n, m))
        acc = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                theta = theta_range[j]
                VT = VT_range[i]
                fixed = (height, VT, theta)
                result = scipy.optimize.minimize(
                    self._cost_fixed,
                    z0,
                    args=(fixed,),
                    bounds=bounds,
                    method=method,
                    options=options,
                )
                cost[i][j] = result.fun
                if np.linalg.norm(cost[i][j]) < eps:
                    (
                        self.Fr,
                        self.Fp,
                        self.q,
                    ) = result.x
                    z0 = {
                        "Fr": self.Fr,
                        "Fp": self.Fp,
                        "q": self.q,
                    }
                    z0 = list(z0.values())
                    dels = np.zeros((3, 1))
                    pos = np.vstack((0, 0, -height))
                    vel = np.vstack((VT * np.cos(theta), 0, VT * np.sin(theta)))
                    quat = np.vstack(angle2quat(0, theta, 0))
                    omega = np.zeros((3, 1))

                    FM = (
                        self.B_Pusher(self.Fp)
                        + self.B_Fuselage(dels, pos, vel, omega)
                        + self.B_Gravity(quat)
                    )
                    R = quat2dcm(quat)
                    F = R.T @ (FM[:3] + np.vstack((0, 0, -self.Fr)))

                    a_x = F[0] / self.m
                    acc[i][j] = a_x[0]

                    success[i][j] = 1
                    Fr[i][j] = self.Fr
                    Fp[i][j] = self.Fp
                    q[i][j] = self.q
                    print(f"vel: {VT:.1f}, theta: {np.rad2deg(theta):.1f}, success")
                else:
                    print(
                        f"vel: {VT:.1f}, theta: {np.rad2deg(theta):.1f}, cost: {cost[i][j]:.3f}"
                    )
                    success[i][j] = np.NaN
                    Fr[i][j] = np.NaN
                    Fp[i][j] = np.NaN
                    q[i][j] = np.NaN
                    acc[i][j] = np.NaN

        Trst_corr = VT_range, theta_range, cost, success, acc, Fr, Fp, q
        return Trst_corr

    def _cost_fixed(self, z, fixed):
        h, VT, theta = fixed
        Fr, Fp, q = z
        X = np.vstack((-h, VT * np.cos(theta), VT * np.sin(theta), theta))
        U = np.vstack((Fr, Fp, q))

        dX = self.plant.deriv(X, U)

        dels = np.zeros((3, 1))
        pos = np.vstack((0, 0, -h))
        vel = np.vstack((VT * np.cos(theta), 0, VT * np.sin(theta)))
        quat = np.vstack(angle2quat(0, theta, 0))
        omega = np.zeros((3, 1))

        FM = (
            self.B_Pusher(Fp)
            + self.B_Fuselage(dels, pos, vel, omega)
            + self.B_Gravity(quat)
        )
        R = quat2dcm(quat)
        F = R.T @ (FM[:3] + np.vstack((0, 0, -Fr)))

        # pos_trim = np.vstack((0, 0, -h))
        # vel_trim = np.vstack((vel * cos(theta), 0, vel * sin(theta)))
        # quat_trim = np.vstack(angle2quat(0, theta, 0))
        # omega_trim = np.vstack((0, 0, 0))
        # rcmds = np.vstack((rotor1, rotor2, rotor3, rotor4, rotor5, rotor6))
        # pcmds = np.vstack((pusher1, pusher2))
        # dels = np.vstack((0, 0, 0))

        #         FM_Rotor = self.B_VTOL(rcmds, omega_trim)
        #         FM_Pusher = self.B_Pusher(pcmds)
        #         FM_Fuselage = self.B_Fuselage(dels, pos_trim, vel_trim, omega_trim)
        #         FM_Gravity = self.B_Gravity(quat_trim)
        #         FM = FM_Fuselage + FM_Pusher + FM_Gravity + FM_Rotor

        #         dpos, dvel, dquat, domega = self.deriv(
        #             pos_trim, vel_trim, quat_trim, omega_trim, FM
        #         )

        # dxs = np.vstack((dpos[0] - VT*cos(alp), dpos[1], dpos[2] - VT*sin(alp), dvel[0] - acc, dvel[1:3], domega))
        # x1 = dpos[0] - vel * cos(theta)
        # x2 = dpos[1]
        # x3 = dpos[2] - vel * sin(theta)
        # x4 = (np.sign(dvel[0]) - 1) * dvel[0]
        # x5 = dvel[1]
        # x6 = (np.sign(dvel[2]) + 1) * dvel[2]
        # x7 = domega
        # x1 = (np.sign(FM[0]) - 1) * FM[0]
        # x2 = FM[2]
        # x3 = FM[4]
        # x4 = (np.sign(dpos[2] ** 2 - Vz_max * dpos[2]) + 1) * (
        #     dpos[2] ** 2 - Vz_max * dpos[2]
        # )

        x1 = dX[0]
        x2 = dX[3]
        x3 = (np.sign(F[0]) - 1) * F[0]
        # x3 = (np.sign(F[2]) + 1) * F[2]
        x4 = F[2]
        dxs = np.vstack((x1, x2, x3, x4))
        weight = np.diag([1, 1, 1, 1])
        cost = dxs.T @ weight @ dxs
        return cost

    def cl_plot(self):
        alp = np.deg2rad(np.arange(-5, 10, 0.1))
        aero_coeff = self.aero_coeff(alp)
        CL = aero_coeff[0, :]

        """ Figure 1 - Trim """
        fig = plt.figure()
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.rad2deg(alp), CL)
        ax.set_xlabel("AoA")
        ax.set_ylabel("CL")

        plt.show()


if __name__ == "__main__":
    system = LC62_corridor()
    height = 50
    grid = {"VT": np.arange(0, 45.1, 0.5), "theta": np.deg2rad(np.arange(-30, 30, 0.2))}
    # grid = {"VT": np.arange(0, 40, 2), "theta": np.deg2rad(np.arange(-30, 30, 2))}

    Trst_corr = system.get_corr(
        z0={"Fr": system.m * system.g, "Fp": 0.0, "q": 0.0},
        height=height,
        grid=grid,
    )
    VT_corr, theta_corr, cost, success, acc, Fr, Fp, q = Trst_corr
    np.savez(
        os.path.join(
            "Corridor/data/corr2.npz",
            # "corr_init_r{0:.1f}_p{1:.1f}.npz".format(r0[i], p0[j]),
        ),
        VT_corr=VT_corr,
        theta_corr=theta_corr,
        cost=cost,
        success=success,
        acc=acc,
        Fr=Fr,
        Fp=Fp,
        q=q,
    )
