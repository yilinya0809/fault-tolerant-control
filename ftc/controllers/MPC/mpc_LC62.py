import casadi as ca
import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan

from ftc.controllers.MPC.dyn_LC62 import LC62

ref_plant = LC62()

def shift_timestep(step_horizon, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    # t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )
    return next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())

def constraints(spec):
    _, _, N, n_states, n_controls = spec

    lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
    ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

    z_eps = 2
    lbx[0: n_states*(N+1): n_states] = -10 - z_eps     # z min
    ubx[0: n_states*(N+1): n_states] = -10 + z_eps     # z max
    lbx[1: n_states*(N+1): n_states] = 0                    # Vx min
    ubx[1: n_states*(N+1): n_states] = ca.inf               # Vx max
    lbx[2: n_states*(N+1): n_states] = -ca.inf              # Vz min
    ubx[2: n_states*(N+1): n_states] = ca.inf               # Vz max

    Fr_max = 6 * ref_plant.th_r_max # 955.2534
    Fp_max = 2 * ref_plant.th_p_max # 183.1982
    lbx[n_states*(N+1)::n_controls] = 0                     # Fr min
    ubx[n_states*(N+1)::n_controls] = Fr_max                # Fr max
    lbx[n_states*(N+1)+1::n_controls] = 0                   # Fp min
    ubx[n_states*(N+1)+1::n_controls] = Fp_max              # Fp max
    lbx[n_states*(N+1)+2::n_controls] = -np.deg2rad(50)     # theta min
    ubx[n_states*(N+1)+2::n_controls] = np.deg2rad(50)      # theta min

    args = {
        'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
        'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
        'lbx': lbx,
        'ubx': ubx
    }
    return args


def solve_mpc(X_init, X_target, U_init, U_target, spec, args):
    step_horizon, tf, N, n_states, n_controls = spec

    z = ca.MX.sym('z')
    Vx = ca.MX.sym('Vx')
    Vz = ca.MX.sym('Vz')
    states = ca.vertcat(
        z,
        Vx,
        Vz,
    )

    Fr = ca.MX.sym('Fr')
    Fp = ca.MX.sym('Fp')
    theta = ca.MX.sym('theta')
    controls = ca.vertcat(
        Fr,
        Fp,
        theta,
    )

    X = ca.MX.sym('X', n_states, N + 1)
    U = ca.MX.sym('U', n_controls, N)
    P = ca.MX.sym('P', 2 * n_states + n_controls)
    Q = ca.diagcat(100, 100, 100)
    R = ca.diagcat(0.01, 0.1, 100)

    Xdot = ref_plant.deriv(states, controls)
    f = ca.Function('f', [states, controls], [Xdot])

    cost_fn = 0  # cost function
    g = X[:, 0] - P[:n_states]  # constraints in the equation

    # runge kutta
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_err = st - P[n_states:2*n_states]
        con_err = con - P[2*n_states:]
        cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err
        st_next = X[:, k+1]
        k1 = f(st, con)
        k2 = f(st + step_horizon/2*k1, con)
        k3 = f(st + step_horizon/2*k2, con)
        k4 = f(st + step_horizon * k3, con)
        st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
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


    u0 = ca.repmat(U_init, 1, N)
    X0 = ca.repmat(X_init, 1, N+1)
    cat_states = DM2Arr(X0)

    mpc_iter = 0
    while (ca.norm_2(X_init - X_target > 1e-1) and (mpc_iter * step_horizon < tf)):
        args['p'] = ca.vertcat(
            X_init,     # current state
            X_target,   # target state
            U_target,   # target control
        )
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        X_init, u0 = shift_timestep(step_horizon, X_init, u, f)

        cat_states = np.dstack((
           cat_states,
           DM2Arr(X0)
        ))

        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
       
        mpc_iter = mpc_iter + 1

    Xd = cat_states[:, 0]
    Ud = u0[:, 0]
    breakpoint()
    return Xd, Ud


class MPCController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.J = env.plant.J
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
        self.p1, p2 = 70, 0.0835
        self.ang_lim = np.deg2rad(50)
        self.mg = env.plant.m * env.plant.g

        step_horizon = 0.1
        tf = 20
        N = 10
        n_states = 3
        n_controls = 3
        self.spec = (step_horizon, tf, N, n_states, n_controls)

        z_init = -10
        Vx_init = 0
        Vz_init = 0
        self.X_init = ca.DM([z_init, Vx_init, Vz_init])

        Fr_init = self.mg
        Fp_init = 0
        theta_init = 0
        self.U_init = ca.DM([Fr_init, Fp_init, theta_init])

        X_trim, U_trim = ref_plant.get_trim(fixed={"h": 10, "VT": 45})
        self.X_target = ca.DM(X_trim)
        self.U_target = ca.DM(U_trim)


    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang0 = np.vstack(quat2angle(quat)[::-1])
        ang_min, ang_max = -self.ang_lim, self.ang_lim
        ang = np.clip(ang0, ang_min, ang_max)

        args = constraints(self.spec)
        X_init = ca.DM([pos[2], vel[0], vel[2]]) # Current state
        if np.isclose(t % 0.1, 0):
            self.Xd, self.Ud = solve_mpc(X_init, self.X_target, self.U_init, self.U_target, self.spec, args)


        z_d, Vx_d, Vz_d = np.ravel(self.Xd)
        Fr, Fp, theta_d = np.ravel(self.Ud)

        angd = np.vstack((0, theta_d, 0))
        omegad = np.zeros((3, 1))
        
        f = -np.linalg.inv(self.J) @ (np.cross(omega, self.J @ omega, axis=0))
        g = np.linalg.inv(self.J)

        Ki1 = 1 * np.diag((10, 100, 10))
        Ki2 = 1 * np.diag((10, 100, 10))
        Mr = np.linalg.inv(g) @ (-f - Ki1 @ (ang - angd) - Ki2 @ (omega - omegad))


#         H = np.array([
#             [1, 0, tan(theta)],
#             [0, 1, 0],
#             [0, 0, 1/cos(theta)]
#         ])

#         Omega_dot = np.vstack((0, theta_dot, 0))
#         w_d = np.linalg.inv(H) @ Omega_dot
#         w_dot_d = 0

#         Mr_d = env.plant.J @ w_dot_d + np.cross(w_d, env.plant.J @ w_d)

        nu = np.vstack((Fr, Mr))
        th_r = np.linalg.pinv(self.B_r2FM) @ nu
        rcmds = th_r / self.r1
        
        th_p = Fp / 2
        pcmds = th_p / self.p1 * np.ones((2, 1))
        
        dels = np.zeros((3, 1))
        ctrls = np.vstack((rcmds, pcmds, dels))

        controller_info = {
            "z_d": z_d,
            "Vx_d": Vx_d,
            "Vz_d": Vz_d,
            "Fr_d": Fr,
            "Fp_d": Fp,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info

        


         

