from time import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.LC62X import LC62


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


plant = LC62()

step_horizon = 0.1  # time between steps in seconds
N = 10  # number of look ahead steps
sim_time = 10  # simulation time

# x_init = 0
z_init = -50.0
V_init = 0.1
gamma_init = 0

""" Get trim """
X_trim, U_trim = plant.get_trim(fixed={"h": 50, "VT": 45})
z_target, V_target, gamma_target = X_trim.ravel()
Fr_target, Fp_target, alp_target = U_trim.ravel()

# state symbolic variables
# x = ca.MX.sym("x")
z = ca.MX.sym("z")
V = ca.MX.sym("V")
gamma = ca.MX.sym("gamma")
states = ca.vertcat(z, V, gamma)
n_states = states.numel()

# control symbolic variables
Fr = ca.MX.sym("Fr")
Fp = ca.MX.sym("Fp")
alp = ca.MX.sym("alp")
controls = ca.vertcat(Fr, Fp, alp)
n_controls = controls.numel()

X = ca.MX.sym("X", n_states, N + 1)
U = ca.MX.sym("U", n_controls, N)
P = ca.MX.sym("P", 2 * n_states + n_controls)

# Q = 10 * ca.diagcat(1, 20, 500) # Gain matrix for X
# R = 0.01 * ca.diagcat(1, 1, 500) # Gain matrix for U
Q = 10 * ca.diagcat(1, 20, 500) # Gain matrix for X
R = 0.01 * ca.diagcat(1, 1, 100) # Gain matrix for U



Xdot = plant.deriv(states, controls)
f = ca.Function("f", [states, controls], [Xdot])

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation

# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    st_error = st - P[n_states : 2 * n_states]
    con_error = con - P[2 * n_states :]
    cost_fn = cost_fn + st_error.T @ Q @ st_error + con_error.T @ R @ con_error
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
        "max_iter": 100,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-6,
    },
    "print_time": 0,
}

solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

z_eps = 2
lbx[0 : n_states * (N + 1) : n_states] = z_target - 2  # z min
ubx[0 : n_states * (N + 1) : n_states] = z_target + 2  # z max
lbx[1 : n_states * (N + 1) : n_states] = 0  # V min
ubx[1 : n_states * (N + 1) : n_states] = ca.inf  # V max
lbx[2 : n_states * (N + 1) : n_states] = -np.deg2rad(30)  
ubx[2 : n_states * (N + 1) : n_states] = np.deg2rad(30)
# lbx[2 : n_states * (N + 1) : n_states] = -np.deg2rad(40)  
# ubx[2 : n_states * (N + 1) : n_states] = np.deg2rad(40)


Fr_max = 6 * plant.th_r_max  # 955.2534
Fp_max = 2 * plant.th_p_max  # 183.1982
lbx[n_states * (N + 1) :: n_controls] = 0 # Fr min
ubx[n_states * (N + 1) :: n_controls] = Fr_max  # Fr max
lbx[n_states * (N + 1) + 1 :: n_controls] = 0 # Fp min
ubx[n_states * (N + 1) + 1 :: n_controls] = Fp_max  # Fp max
lbx[n_states * (N + 1) + 2 :: n_controls] = -np.deg2rad(30)  # alp min
ubx[n_states * (N + 1) + 2 :: n_controls] = np.deg2rad(30)  # alp min
# lbx[n_states * (N + 1) + 2 :: n_controls] = -np.deg2rad(45)  # alp min
# ubx[n_states * (N + 1) + 2 :: n_controls] = np.deg2rad(45)  # alp min


args = {
    "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
    "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
    "lbx": lbx,
    "ubx": ubx,
}

t0 = 0
state_init = ca.DM([z_init, V_init, gamma_init])
state_target = ca.DM([z_target, V_target, gamma_target])
control_init = ca.DM([plant.m * plant.g, 0, 0])
control_target = ca.DM([Fr_target, Fp_target, alp_target])
t = ca.DM(t0)

# u0 = ca.DM.zeros((n_controls, N))  # initial control
u0 = ca.repmat(control_init, 1, N)
X0 = ca.repmat(state_init, 1, N + 1)  # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
# cat_controls = DM2Arr(u0[:, 0])
cat_controls = DM2Arr(u0)
times = np.array([[0]])


if __name__ == "__main__":
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (
        mpc_iter * step_horizon < sim_time
    ):
        t1 = time()
        args["p"] = ca.vertcat(
            state_init, state_target, control_target
        )  # current state, target state, target control
        # optimization variable current state
        args["x0"] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1)
        )

        sol = solver(
            x0=args["x0"],
            lbx=args["lbx"],
            ubx=args["ubx"],
            lbg=args["lbg"],
            ubg=args["ubg"],
            p=args["p"],
        )

        X0 = ca.reshape(sol["x"][: n_states * (N + 1)], n_states, N + 1)
        u = ca.reshape(sol["x"][n_states * (N + 1) :], n_controls, N)

        cat_states = np.dstack((cat_states, DM2Arr(X0)))
        # cat_controls = np.dstack((cat_controls, DM2Arr(u[:, 0])))
        cat_controls = np.dstack((cat_controls, DM2Arr(u)))
        t = np.vstack((t, t0))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))

        t2 = time()
        print(mpc_iter)
        print(t2 - t1)
        times = np.vstack((times, t2 - t1))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print("\n")
    print("Total time: ", main_loop_time - main_loop)
    print("avg iteration time: ", np.array(times).mean() * 1000, "ms")
    print("final error: ", ss_error)

    np.save("time.npy", t)
    np.save("cat_states.npy", cat_states)
    np.save("cat_controls.npy", cat_controls)

    """Fig 1. States"""
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, cat_states[0, 0, :])
    ax.set_ylabel(r"$z$ [m]")
    ax.set_xlabel("Time [s]")
    ax.set_ylim([-55, -45])
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_states[1, 0, :])
    ax.set_ylabel(r"$V$ [m/s]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, np.rad2deg(cat_states[2, 0, :]))
    ax.set_ylabel(r"$\gamma$ [deg]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Fig 2. Controls """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, cat_controls[0, 0, :])
    ax.set_ylabel(r"$F_r$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_controls[1, 0, :])
    ax.set_ylabel(r"$F_p$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, np.rad2deg(cat_controls[2, 0, :]))
    ax.set_ylabel(r"$\alpha$ [deg]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    plt.show()
