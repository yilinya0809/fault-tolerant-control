import scipy
from casadi import *


class LC62:
    g = 9.81  # [m / sec^2]
    m = 41.97  # [kg]
    S1 = 0.2624  # [m^2]
    S2 = 0.5898  # [m^2]
    S = S1 + S2
    tables = {
        "alp": np.deg2rad(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])),  # [rad]
        "CL": np.array(
            [0.1931, 0.4075, 0.6112, 0.7939, 0.9270, 1.0775, 0.9577, 1.0497, 1.0635]
        ),
        "CD": np.array(
            [0.0617, 0.0668, 0.0788, 0.0948, 0.1199, 0.1504, 0.2105, 0.2594, 0.3128]
        ),
    }
    th_r_max = 159.2089
    th_p_max = 91.5991

    def deriv(self, x, u):
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

    def derivnox(self, x, u, q):
        g = self.g
        m = self.m
        z, vel = x[0], x[1:]
        Fr, Fp, theta = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage(vel)
        dz = -sin(theta) * vel[0] + cos(theta) * vel[1]
        dvel = vertcat(
            (Fp + Fx) / m - g * sin(theta) - q * vel[1],
            (Fr + Fz) / m + g * cos(theta) + q * vel[0],
        )
        return vertcat(dz, dvel)

    def deriv_ex(self, x, u):
        g = self.g
        m = self.m
        z, Vx, Vz = x[0], x[1], x[2]
        Fr_dot, Fp_dot, q = x[3], x[4], x[5]
        Fr, Fp, theta = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage(np.vstack((Vx, Vz)))
        dz = -sin(theta) * Vx + cos(theta) * Vz
        dvel = vertcat(
            (Fp + Fx) / m - g * sin(theta) - q * Vz,
            (Fr + Fz) / m + g * cos(theta) + q * Vx,
        )
        return vertcat (dz, dvel, Fr_dot, Fp_dot, q)
        

    def B_Fuselage(self, vel):
        S = self.S
        rho = 1.225
        u, w = vel[0], vel[1]
        VT = norm_2(vel)
        alp = arctan2(w, u)
        qbar = 0.5 * rho * VT**2
        CL, CD = self.aero_coeff(alp)
        Fx = qbar * S * (CL * sin(alp) - CD * cos(alp))
        Fz = -qbar * S * (CL*cos(alp) + CD * sin(alp))
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

        dx = self.deriv(x_trim, u_trim)
        dvel = dx[2:]
        weight = np.diag([1, 1])
        return dvel.T @ weight @ dvel
