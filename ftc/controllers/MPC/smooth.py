import casadi as ca
import fym
import numpy as np
from fym.utils.rot import quat2angle

from ftc.models.LC62S import LC62


class MPC:
    def __init__(self, env):
        self.plant = LC62()

        self.Fr_max = 6 * self.plant.th_r_max
        self.Fp_max = 2 * self.plant.th_p_max
        self.theta_max = env.ang_lim
        self.z_eps = 2

        self.step_horizon = 0.1  # time between steps in seconds
        self.N = 10  # number of look ahead steps

        z_init, vx_init, vz_init, theta_init, _ = env.observation()

        X_trim, U_trim = self.plant.get_trim(fixed={"h": 10, "VT": 45})
        _, self.z_target, vx_target, vz_target = X_trim.ravel()
        Fr_target, Fp_target, theta_target = U_trim.ravel()

        self.control_init = ca.DM([-self.plant.m * self.plant.g, 0, theta_init])
        self.state_init = ca.DM([z_init, vx_init, vz_init])
        self.state_target = ca.DM([self.z_target, vx_target, vz_target])
        self.control_target = ca.DM([Fr_target, Fp_target, theta_target])
        self.n_states = self.state_init.numel()
        self.n_controls = self.control_init.numel()
        self.args = self.constraints()

        self.Q = 10 * ca.diagcat(1, 1, 1) # Gain matrix for X
        self.R = 0.0 * ca.diagcat(0, 0, 1000) # Gain matrix for U



    def DM2Arr(self, dm):
        return np.array(dm.full())

    def constraints(self):
        N = self.N
        n_states = self.n_states
        n_controls = self.n_controls

        lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
        ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

        lbx[0 : n_states * (N + 1) : n_states] = self.z_target - 2  # z min
        ubx[0 : n_states * (N + 1) : n_states] = self.z_target + 1
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

        lbg = ca.DM.zeros(((n_states + n_controls) * (N + 1), 1))
        ubg = ca.DM.zeros(((n_states + n_controls) * (N + 1), 1))

        lbg[n_states * (N + 1):] = -ca.inf
        ubg[n_states * (N + 1):] = ca.inf

        args = {
            "lbg": lbg,
            "ubg": ubg,
            "lbx": lbx,
            "ubx": ubx,
        }
        return args

    def get_action(self):
        agent_info = {
            "Xd": self.state_target,
            "Ud": self.control_target,
            "qd": 0
        }
        
        return self.control_init, agent_info

    def solve_mpc(self, obs):
        z, vx, vz, theta, q = obs
        Fr, Fp, _ = np.ravel(self.control_init)
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
        P = ca.MX.sym("P", 2 * (n_states + n_controls))

        Q = ca.diagcat(50, 50, 20)
        R = 0.005 * ca.diagcat(1, 0, 10000)
        S = 0.0 * ca.diagcat(1, 1, 0)
        # Q = self.Q
        # R = self.R

        Xdot = self.plant.derivnox(states, controls, q)
        f = ca.Function("f", [states, controls], [Xdot])

        cost_fn = 0  # cost function
        g = X[:, 0] - P[:n_states]  # constraints in the equation
        u_diff = U[:, 0] - P[2 * n_states + n_controls:]

        for k in range(N):
            # discrete dynamics runge kutta
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k + 1]
            k1 = f(st, con)
            k2 = f(st + step_horizon / 2 * k1, con)
            k3 = f(st + step_horizon / 2 * k2, con)
            k4 = f(st + step_horizon * k3, con)
            st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

            # cost function
            st_err = st - P[n_states : 2 * n_states]
            con_err = con - P[2 * n_states : 2 * n_states + n_controls]
            con_diff = con - P[2 * n_states + n_controls:]
            cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err + con_diff.T @ S @ con_diff 
            u_diff = ca.vertcat(u_diff, con_diff)

        opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        opt_params = ca.vertcat(P.reshape((-1, 1)))
        opt_g = ca.vertcat(g, u_diff)
        nlp_prob = {"f": cost_fn, "x": opt_variables, "g": opt_g, "p": opt_params}

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
            self.control_target, # target control
            self.control_init
        )
        self.args["x0"] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1)
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

