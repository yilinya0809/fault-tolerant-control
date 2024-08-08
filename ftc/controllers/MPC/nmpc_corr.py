import casadi as ca
import fym
import numpy as np
from ftc.models.LC62S import LC62
from fym.utils.rot import quat2angle

from poly_corr import boundary, poly, weighted_poly

Trst_corr = np.load("corr.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc_corr"]
theta_corr = Trst_corr["theta_corr"]


class MPC_Corr:
    def __init__(self, env):
        self.plant = LC62()

        self.Fr_max = 6 * self.plant.th_r_max
        self.Fp_max = 2 * self.plant.th_p_max
        self.theta_max = env.ang_lim
        self.z_eps = 2

        self.step_horizon = 0.2  # time between steps in seconds
        self.N = 5  # number of look ahead steps

        z_init, vx_init, vz_init, theta_init, _ = env.observation()
        self.state_init = ca.DM([z_init, vx_init, vz_init])
        self.control_init = ca.DM([-self.plant.m * self.plant.g, 0, theta_init])

        VT_target = VT_corr[0]
        theta_target = np.deg2rad(theta_corr[0, 0])
        vx_target = VT_target * np.cos(theta_target)
        vz_target = VT_target * np.sin(theta_target)
        self.Fr_target = 0
        self.Fp_target = self.Fp_max

        self.z_target = -50

        self.state_target = ca.DM([self.z_target, vx_target, vz_target])
        self.control_target = ca.DM([self.Fr_target, self.Fp_target, theta_target])
        self.n_states = self.state_init.numel()
        self.n_controls = self.control_init.numel()
        self.args = self.constraints()

    def DM2Arr(self, dm):
        return np.array(dm.full())

    def set_ref(self, t, tf, v0, vf):
        VT_ref = v0 + (vf - v0) / tf * t
        upper_bound, lower_bound = boundary(VT_corr)
        degree = 3
        upper, lower, central = poly(degree, VT_corr, upper_bound, lower_bound)
        weighted = weighted_poly(degree, VT_corr, vf, upper, lower)

        theta_ref = np.deg2rad(weighted(VT_ref))

        vx_target = VT_ref * np.cos(theta_ref)
        vz_target = VT_ref * np.sin(theta_ref)
        self.state_target = ca.DM([self.z_target, vx_target, vz_target])
        self.control_target = ca.DM([self.Fr_target, self.Fp_target, theta_ref])
        return np.array(self.state_target.full())

    def constraints(self):
        N = self.N
        n_states = self.n_states
        n_controls = self.n_controls

        lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
        ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

        lbx[0 : n_states * (N + 1) : n_states] = self.z_target - self.z_eps  # z min
        ubx[0 : n_states * (N + 1) : n_states] = self.z_target + self.z_eps  # z max
        lbx[1 : n_states * (N + 1) : n_states] = 0  # Vx min
        ubx[1 : n_states * (N + 1) : n_states] = ca.inf  # Vx max
        lbx[2 : n_states * (N + 1) : n_states] = -ca.inf  # Vz min
        ubx[2 : n_states * (N + 1) : n_states] = ca.inf  # Vz max

        lbx[n_states * (N + 1) :: n_controls] = -self.Fr_max  # Fr min
        ubx[n_states * (N + 1) :: n_controls] = 0  # Fr max
        lbx[n_states * (N + 1) + 1 :: n_controls] = 0  # Fp min
        ubx[n_states * (N + 1) + 1 :: n_controls] = self.Fp_max  # Fp max
        lbx[n_states * (N + 1) + 2 :: n_controls] = -self.theta_max  # theta min
        ubx[n_states * (N + 1) + 2 :: n_controls] = self.theta_max  # theta max

        args = {
            "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
            "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
            "lbx": lbx,
            "ubx": ubx,
        }
        return args

    def get_action(self):
        agent_info = {
            "Xd": self.state_target,
            "Ud": self.control_target,
            "qd": 0,
        }
        return self.control_init, agent_info

    def solve_mpc(self, obs):
        z, vx, vz, theta, q = obs
        Fr, Fp, _ = ca.vertsplit(self.control_init)
        state_init = ca.DM([z, vx, vz])
        control_init = ca.DM([Fr, Fp, theta])

        step_horizon = self.step_horizon
        N = self.N
        n_states = self.n_states
        n_controls = self.n_controls

        z = ca.MX.sym("z")
        vx = ca.MX.sym("vx")
        vz = ca.MX.sym("vz")
        states = ca.vertcat(z, vx, vz)

        Fr = ca.MX.sym("Fr")
        Fp = ca.MX.sym("Fp")
        theta = ca.MX.sym("theta")
        controls = ca.vertcat(Fr, Fp, theta)

        X = ca.MX.sym("X", n_states, N + 1)
        U = ca.MX.sym("U", n_controls, N)
        P = ca.MX.sym("P", 2 * n_states + n_controls)

        #         Q = ca.diagcat(300, 300, 300)
        #         R = ca.diagcat(0.01, 0.1, 200000)

        Q = ca.diagcat(1, 1, 1)
        R = ca.diagcat(0, 0, 0)

        # Xdot = self.plant.deriv_lin(states, controls, q)
        Xdot = self.plant.derivnox(states, controls, q)

        f = ca.Function("f", [states, controls], [Xdot])
        cost_fn = 0  # cost function
        g = X[:, 0] - P[:n_states]  # constraints in the equation

        # runge kutta
        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            st_err = st - P[n_states : 2 * n_states]
            con_err = con - P[2 * n_states :]
            cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err
            st_next = X[:, k + 1]
            k1 = f(st, con)
            k2 = f(st + step_horizon / 2 * k1, con)
            k3 = f(st + step_horizon / 2 * k2, con)
            k4 = f(st + step_horizon * k3, con)
            st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        opt_params = ca.vertcat(P.reshape((-1, 1)))
        nlp_prob = {"f": cost_fn, "x": opt_variables, "g": g, "p": opt_params}

        opts = {
            "ipopt": {
                "max_iter": 1000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }

        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

        u0 = ca.repmat(control_init, 1, N)
        X0 = ca.repmat(state_init, 1, N + 1)  # initial state full

        self.args["p"] = ca.vertcat(
            state_init,  # current state
            self.state_target,  # target state
            self.control_target,  # target control
        )
        self.args["x0"] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1),
            ca.reshape(u0, n_controls * N, 1),
        )
        sol = solver(
            x0=self.args["x0"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            p=self.args["p"],
        )

        X0 = ca.reshape(sol["x"][: n_states * (N + 1)], n_states, N + 1)
        u = ca.reshape(sol["x"][n_states * (N + 1) :], n_controls, N)

        self.control_init = u[:, 0]
