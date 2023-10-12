import casadi as ca
import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


class MPCController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.step_horizon = 0.05  # time between steps in seconds
        self.N = 10              # number of look ahead steps
        self.sim_time = 20       # simulation time

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

        z_init = -10
        Vx_init = 0
        Vz_init = 0

        X_trim, U_trim = self.plant.get_trim_mpc(fixed={"h": 10, "VT": 45})
        z_target, Vx_target, Vz_target = np.ravel(X_trim)
        Fr_target, Fp_target, theta_target = np.ravel(U_trim)

        self.n_states = 3
        self.n_controls = 3
        self.state_init = ca.DM([z_init, Vx_init, Vz_init])        
        self.state_target = ca.DM([z_target, Vx_target, Vz_target]) 
        self.control_init = ca.DM([env.plant.m * env.plant.g, 0, 0])
        self.control_target = ca.DM([Fr_target, Fp_target, theta_target])



    def nlp_solver(self, env):
        z = ca.MX.sym('z')
        Vx = ca.MX.sym('Vx')
        Vz = ca.MX.sym('Vz')
        states = ca.vertcat(
            z,
            Vx,
            Vz,
        )
        n_states = states.numel()

        Fr = ca.MX.sym('Fr')
        Fp = ca.MX.sym('Fp')
        theta = ca.MX.sym('theta')
        controls = ca.vertcat(
            Fr,
            Fp,
            theta,
        )
        n_controls = controls.numel()

        X = ca.MX.sym('X', n_states, self.N + 1)
        U = ca.MX.sym('U', n_controls, self.N)
        P = ca.MX.sym('P', 2 * n_states + n_controls)
        Q = ca.diagcat(200, 100, 100)
        R = ca.diagcat(0.01, 0.1, 200)

        Xdot = env.deriv_mpc(states, controls)
        f = ca.Function('f', [states, controls], [Xdot])

        cost_fn = 0  # cost function
        g = X[:, 0] - P[:n_states]  # constraints in the equation

        # runge kutta
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            st_err = st - P[n_states:2*n_states]
            con_err = con - P[2*n_states:]
            cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err
            st_next = X[:, k+1]
            k1 = f(st, con)
            k2 = f(st + self.step_horizon/2*k1, con)
            k3 = f(st + self.step_horizon/2*k2, con)
            k4 = f(st + self.step_horizon * k3, con)
            st_next_RK4 = st + (self.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)), 
            U.reshape((-1, 1))
        )

        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver

    def constraints(self, env):
        n_states, n_controls = self.n_states, self.n_controls

        lbg = ca.DM.zeros((n_states*(self.N+1), 1)),  # constraints lower bound
        ubg = ca.DM.zeros((n_states*(self.N+1), 1)),  # constraints lower bound

        lbx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))
        ubx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))

        z_eps = 2
        lbx[0: n_states*(self.N+1): n_states] = z_target - z_eps     # z min
        ubx[0: n_states*(self.N+1): n_states] = z_target + z_eps     # z max
        lbx[1: n_states*(self.N+1): n_states] = 0                    # Vx min
        ubx[1: n_states*(self.N+1): n_states] = ca.inf               # Vx max
        lbx[2: n_states*(self.N+1): n_states] = -ca.inf              # Vz min
        ubx[2: n_states*(self.N+1): n_states] = ca.inf               # Vz max


        Fr_max = 6 * env.plant.th_r_max # 955.2534
        Fp_max = 2 * env.plant.th_p_max # 183.1982
        lbx[n_states*(self.N+1)::n_controls] = 0                     # Fr min
        ubx[n_states*(self.N+1)::n_controls] = Fr_max                # Fr max
        lbx[n_states*(self.N+1)+1::n_controls] = 0                   # Fp min
        ubx[n_states*(self.N+1)+1::n_controls] = Fp_max              # Fp max
        lbx[n_states*(self.N+1)+2::n_controls] = -np.deg2rad(50)     # theta min
        ubx[n_states*(self.N+1)+2::n_controls] = np.deg2rad(50)      # theta min

        return lbg, ubg, lbx, ubx

    def solve_mpc(self, env, t):
        args = {
            'lbg', 'ubg', 'lbx', 'ubx' = self.constraints(env)
        }

        t0 = 0
        t = ca.DM(t0)

        u0 = ca.repmat(control_init, 1, N)
        X0 = ca.repmat(state_init, 1, N+1)         # initial state full

        mpc_iter = 0
        cat_states = DM2Arr(X0)
        cat_controls = DM2Arr(u0[:, 0])
        times = np.array([[0]])




