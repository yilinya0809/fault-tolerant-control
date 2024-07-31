import fym
import numpy as np
import scipy
from casadi import *
from fym.utils.rot import quat2angle

from ftc.models.LinearLC62 import LinearLC62
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

        self.model_FW = LinearLC62()
        x_trims, u_trims_FW = self.model_FW.get_trim_fixed(
            fixed={"h": self.h_ref, "VT": self.VT_ref}
        )
        theta_trim = quat2angle(x_trims[2])[1]
        Fp_trim = self.model_FW.B_Pusher(u_trims_FW[0])[0, 0]
        Fr_trim = 0
        self.X_trim = np.vstack(
            (
                -self.h_ref,
                self.VT_ref * np.cos(theta_trim),
                self.VT_ref * np.sin(theta_trim),
            )
        )
        self.U_trim = np.vstack((Fr_trim, Fp_trim, theta_trim))
        self.X_hover = np.vstack((-self.h_ref, 0, 0))
        self.U_hover = np.vstack((self.m * self.g, 0, 0))

    def aero_coeff_lin(self, alp):
        CL = 0.2764 * alp - 0.0779
        CD = 1.0075 * alp - 0.0121
        return CL, CD

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

    def B_aero(self, VT, alp):
        S = self.S
        rho = 1.225
        qbar = 0.5 * rho * VT**2
        CL, CD = self.aero_coeff(alp)
        L = qbar * S * CL
        D = qbar * S * CD
        return L, D

    def deriv(self, X, U):
        g = self.g
        m = self.m
        W = self.m * self.g
        z, V, gamma = X[0], X[1], X[2]
        Fr, Fp, alp = U[0], U[1], U[2]
        L, D = self.B_aero(V, alp)

        dz = -V * sin(gamma)
        dV = (-D + Fp * cos(alp) - Fr * sin(alp) - W * sin(gamma)) / self.m
        # if is_equal(V, 0, 1):
        #     dgamma = 0
        # else:
        #     dgamma = (L + Fp * sin(alp) + Fr * cos(alp) - W *cos(gamma))/(self.m * V)
        dgamma = (L + Fp * sin(alp) + Fr * cos(alp) - W * cos(gamma)) / (self.m * V)

        return vertcat(dz, dV, dgamma)

    def get_trim(
        self,
        z0={
            "alpha": 0.0,
            "gamma": 0.0,
            "Fr": 0,
            "Fp": 0.5,
        },
        fixed={"h": 50, "VT": 45},
        method="SLSQP",
        options={"disp": False, "ftol": 1e-10},
    ):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            np.deg2rad((0, 20)),
            np.deg2rad((-20, 20)),
            [0, 1000],
            [0, 180],
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
            alp, gamma, Fr, Fp = 0, 0, self.m * self.g, 0
        else:
            alp, gamma, Fr, Fp = result.x

        X_trim = np.vstack((-h, VT, gamma))
        U_trim = np.vstack((Fr, Fp, alp))
        return X_trim, U_trim

    def _trim_cost(self, z, fixed):
        h, VT = fixed
        alp, gamma, Fr, Fp = z
        # pos_trim = np.vstack((0, -h))
        # vel_trim = np.vstack((VT * cos(gamma), -VT * sin(gamma)))

        X_trim = np.vstack((-h, VT, gamma))
        U_trim = np.vstack((Fr, Fp, alp))
        theta = gamma + alp

        dX = self.deriv(X_trim, U_trim)
        # Vth = np.vstack((dX[1], gamma))
        weight = np.diag([1, 1, 1])
        return dX.T @ weight @ dX


if __name__ == "__main__":
    sys = LC62()
    # print(sys.A)
    # print(sys.B)

    Xdot1 = sys.deriv(sys.X_trim, sys.U_trim)
    Xdot2 = sys.deriv(sys.X_hover, sys.U_hover)
    breakpoint()

    # Xdot2 = sys.deriv_lin(np.zeros((3,1)), np.zeros((3,1)), q=0)
    # print(Xdot1, Xdot2)
