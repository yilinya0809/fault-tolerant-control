import numpy as np
import scipy
from casadi import *

from ftc.utils import linearization

class LC62:
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
        X_trim, U_trim = self.get_trim(fixed={"h": 50, "VT": 45})
        X_hover, U_hover = self.get_trim(fixed={"h": 50, "VT": 0})
        # self.A, self.B = self.deriv_mat(X_hover[1:], U_hover, q=0)

        self.X_trim = X_trim[1:]
        self.U_trim = U_trim


        # self.A , self.B = self.linearize(fixed={"h": 50, "VT": 0}, q=0)
    # def linearization(self, statefunc, states, ctrls, ptrb):
    #     n = np.size(states)
    #     m = np.size(ctrls)
    #     A = DM(n, n)
    #     B = DM(n, m)

    #     f = statefunc.full()
        
    #     for i in np.arange(n):
    #         ptrb_x = array(n, 1)
    #         ptrb_x[i] = ptrb
    #         x_ptrb = states + ptrb_x

    #         dfdx = (f(x_ptrb, ctrls) - f(states, ctrls)) / ptrb
    #         for j in np.arange(n):
    #             A[j, i] = dfdx[j]
    #     for i in np.arange(m):
    #         ptrbvec_u = array((m, 1))
    #         ptrbvec_u[i] = ptrb
    #         u_ptrb = ctrls + ptrbvec_u

    #         dfdu = (f(states, u_ptrb) - f(states, ctrls)) / ptrb
    #         for j in np.arange(n):
    #             B[j, i] = dfdu[j]

    #     return A, B

        
        
                
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
        pos, vel = x[:2], x[2:]
        Fr, Fp, theta = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage_lin(vel)
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

    def derivnoxq(self, x, u):
        g = self.g
        m = self.m
        z, vel = x[0], x[1:]
        Fr, Fp, theta = u[0], u[1], u[2]
        Fx, Fz = self.B_Fuselage(vel)
        dz = -sin(theta) * vel[0] + cos(theta) * vel[1]
        dvel = vertcat(
            (Fp + Fx) / m - g * sin(theta),
            (Fr + Fz) / m + g * cos(theta),
        )
        return vertcat(dz, dvel)


    
    def deriv_mat(self, x, u, q):
        f = self.derivnoxq
        ptrb = 1e-6
        A0, B = linearization(f, x, u, ptrb)
        A = A0 + np.array((
            [0,0,0],
            [0,0,-1],
            [0,1,0],
        )) * q
        
        return A, B

    def deriv_lin(self, x, u, q):
        AA, BB = self.deriv_mat(x, u, q)
        # n = MX.size(x)[0]
        # m = MX.size(u)[0]

        # A = np.zeros((n, n))
        # B = np.zeros((n, m))
        
        
        Xdot = AA @ x + BB @ u
        return Xdot
       
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


    def B_Fuselage_lin(self):
        S = self.S
        rho = 1.225
        u, w = self.X_trim[1], self.X_trim[2]
        alp = self.U_trim[2]
        # u, w = vel[0], vel[1]
        VT = norm_2(vel)
        VT = 45
        # alp = arctan2(w, u)
        qbar = 0.5 * rho * VT**2
        
        # CL, CD = self.aero_coeff_lin(alp)
        # Fx = qbar * S * (CL * sin(alp) - CD * cos(alp))
        # Fz = -qbar * S * (CL*cos(alp) + CD * sin(alp))
        
        CL = 0.2764 * alp - 0.0779
        CD = 1.0075 * alp - 0.0121
        Fx = qbar * S * (-0.0779 * alp -1.0075 * alp + 0.0121)
        Fz = qbar * S * (0.2764 * alp - 0.0779 + 1.0075 * alp)

        return Fx, Fz

    def aero_coeff_lin(self, alp):
        # clgrid = interpolant(
        #     "CLGRID", "bspline", [self.tables["alp"]], self.tables["CL"]
        # )
        # cdgrid = interpolant(
        #     "CDGRID", "bspline", [self.tables["alp"]], self.tables["CD"]
        # )
        # CL = clgrid(alp)
        # CD = cdgrid(alp)

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

    def get_trim(
        self,
        z0={
            "alpha": 0.0,
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

    def linearize(self, fixed={"h": 50, "VT": 0}, q=0):
        h, VT = list(fixed.values())
        # hovering setpoint
        z = -h
        Vx = VT
        Vz = 0
        Fr = -self.m * self.g
        Fp = 0
        theta = 0

        X = np.vstack((z, Vx, Vz))
        U = np.vstack((Fr, Fp, theta))
        f = self.derivnoxq
        ptrb = 1e-6
        A0, B0 = linearization(f, X, U, ptrb)
        A = A0 + np.array(([0, 0, 0], [0, 0, -1], [0, 1, 0])) * q
        B = B0

        return A, B





if __name__ == "__main__":
    sys = LC62()
    # print(sys.A)
    # print(sys.B)
    
    Xdot1 = sys.derivnox(sys.X_trim, sys.U_trim, q=0)
    # Xdot2 = sys.deriv_lin(np.zeros((3,1)), np.zeros((3,1)), q=0)
    breakpoint()
    # print(Xdot1, Xdot2)
