import numpy as np
import scipy
from casadi import *

from ftc.utils import linearization


class LC62:
    h_ref = 10
    VT_ref = 45

    def __init__(self):

        self.g = 9.81  # [m / sec^2]
        self.m = 41.97  # [kg]
        S1 = 0.2624  # [m^2]
        S2 = 0.5898  # [m^2]
        self.S = S1 + S2
        self.tables = {
            "alp": np.deg2rad(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])),  # [rad]
            "CL": np.array(
                [0.1931, 0.4075, 0.6112, 0.7939, 0.9270, 1.0775, 0.9577, 1.0497, 1.0635]
            ),
            "CD": np.array(
                [0.0617, 0.0668, 0.0788, 0.0948, 0.1199, 0.1504, 0.2105, 0.2594, 0.3128]
            ),
        }
        self.th_r_max = 159.2089
        self.th_p_max = 91.5991
        X_trim, U_trim = self.get_trim(fixed={"h": self.h_ref, "VT": self.VT_ref})
        X_hover, U_hover = self.get_trim(fixed={"h": self.h_ref, "VT": 0})

        self.X_trim = X_trim[1:]
        self.U_trim = U_trim

    def deriv_original(self, x, u):
        g = self.g
        m = self.m
        pos, vel = x[:2], x[2:]
        Fr, Fp, theta = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage(vel)
        dpos = vertcat(
            cos(theta) * vel[0] + sin(theta) * vel[1],
            -sin(theta) * vel[0] + cos(theta) * vel[1],
        )
        dvel = vertcat((Fp + Fx) / m - g * sin(theta), (Fr + Fz) / m + g * cos(theta))
        return vertcat(dpos, dvel)

    def deriv(self, x, u):
        g = self.g
        m = self.m
        z, vel, theta = x[0], x[1:3], x[3]
        Fr, Fp, q = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage(vel)
        dz = -sin(theta) * vel[0] + cos(theta) * vel[1]
        dvel = vertcat(
            (Fp + Fx) / m - g * sin(theta) - q * vel[1],
            (Fr + Fz) / m + g * cos(theta) + q * vel[0],
        )
        dtheta = q
        return vertcat(dz, dvel, dtheta)

    def B_Fuselage(self, vel):
        S = self.S
        rho = 1.225
        u, w = vel[0], vel[1]
        VT = norm_2(vel)
        alp = arctan2(w, u)
        qbar = 0.5 * rho * VT**2
        CL, CD = self.aero_coeff(alp)
        Fx = qbar * S * (CL * sin(alp) - CD * cos(alp))
        Fz = -qbar * S * (CL * cos(alp) + CD * sin(alp))
        return Fx, Fz

    def aero_coeff(self, alp):
        clgrid = interpolant(
            "CLGRID", "bspline", [self.tables["alp"]], self.tables["CL"]
        )
        cdgrid = interpolant(
            "CDGRID", "bspline", [self.tables["alp"]], self.tables["CD"]
        )
        CL = clgrid(alp)
        CD = cdgrid(alp)

        return CL, CD

    def get_trim(
        self,
        z0={
            "alpha": 0.0,
            "Fr": 0,
            "Fp": 0.5,
        },
        fixed={"h": 10, "VT": 45},
        method="SLSQP",
        options={"disp": False, "ftol": 1e-10},
    ):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            np.deg2rad((0, 20)),
            [0, 0],
            [0, 1000],
        )
        result = scipy.optimize.minimize(
            self._trim_cost,
            z0,
            args=(fixed,),
            bounds=bounds,
            method=method,
            options=options,
        )

        h, VT = fixed
        if np.isclose(VT, 0):
            alp, Fr, Fp = 0, self.m * self.g, 0
        else:
            alp, Fr, Fp = result.x
        pos_trim = np.vstack((0, -h))
        vel_trim = np.vstack((VT * cos(alp), VT * sin(alp)))

        x_trim = np.vstack((pos_trim, vel_trim))
        u_trim = np.vstack((Fr, Fp, alp))
        return x_trim, u_trim

    def _trim_cost(self, z, fixed):
        h, VT = fixed
        alp, Fr, Fp = z
        pos_trim = np.vstack((0, -h))
        vel_trim = np.vstack((VT * cos(alp), VT * sin(alp)))

        x_trim = np.vstack((pos_trim, vel_trim))
        u_trim = np.vstack((Fr, Fp, alp))

        dx = self.deriv_original(x_trim, u_trim)
        dvel = dx[2:]
        weight = np.diag([1, 1])
        return dvel.T @ weight @ dvel


if __name__ == "__main__":
    sys = LC62()
    # print(sys.A)
    # print(sys.B)

    Xdot1 = sys.derivnox(sys.X_trim, sys.U_trim, q=0)
    # Xdot2 = sys.deriv_lin(np.zeros((3,1)), np.zeros((3,1)), q=0)
    breakpoint()
    # print(Xdot1, Xdot2)
