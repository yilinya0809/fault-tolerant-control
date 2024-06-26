from time import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from poly_corr import boundary, poly, weighted_poly

import ftc
from ftc.models.LC62S import LC62

Trst_corr = np.load("corr.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc_corr"]
theta_corr = Trst_corr["theta_corr"]


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


plant = LC62()

step_horizon = 0.2  # time between steps in seconds
N = 5  # number of look ahead steps
sim_time = 10  # simulation time

# x_init = 0
z_init = z_target = -50.0
vx_init = 0
vz_init = 0

# X_trim, U_trim = plant.get_trim(fixed={"h": 10, "VT": 45})
# _, z_target, vx_target, vz_target = X_trim.ravel()
# Fr_target, Fp_target, theta_target = U_trim.ravel()

VT_target = VT_corr[0]
theta_target = np.deg2rad(theta_corr[0, 0])
Vx_target = VT_target * np.cos(theta_target)
Vz_target = VT_target * np.sin(theta_target)
Fr_target = Fp_target = 0


# state symbolic variables
# x = ca.MX.sym("x")
z = ca.MX.sym("z")
vx = ca.MX.sym("vx")
vz = ca.MX.sym("vz")
# states = ca.vertcat(x, z, vx, vz)
states = ca.vertcat(z, vx, vz)
n_states = states.numel()

# control symbolic variables
Fr = ca.MX.sym("Fr")
Fp = ca.MX.sym("Fp")
theta = ca.MX.sym("theta")
controls = ca.vertcat(Fr, Fp, theta)
n_controls = controls.numel()

X = ca.MX.sym("X", n_states, N + 1)
U = ca.MX.sym("U", n_controls, N)
P = ca.MX.sym("P", 2 * n_states + n_controls)

# weights matrix: state (Q_x, Q_z, Q_Vx, Q_Vz), control (R_Fr, R_Fp, R_theta)
# Q = 10 * ca.diagcat(10, 1, 1) # Gain matrix for X
Q = 10 * ca.diagcat(1, 1, 1)  # Gain matrix for X
R = 0.0 * ca.diagcat(0, 0, 1000)  # Gain matrix for U


# Q = ca.diagcat(100, 100, 100)
# R = ca.diagcat(0.01, 0.1, 100)

Xdot = plant.derivnox(states, controls, q=0.0)
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
        "max_iter": 1000,
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
ubx[0 : n_states * (N + 1) : n_states] = z_target + 1  # z max
# ubx[0 : n_states * (N + 1) : n_states] = 0 # z max
lbx[1 : n_states * (N + 1) : n_states] = 0  # Vx min
ubx[1 : n_states * (N + 1) : n_states] = ca.inf  # Vx max
lbx[2 : n_states * (N + 1) : n_states] = -ca.inf  # Vz min
ubx[2 : n_states * (N + 1) : n_states] = ca.inf  # Vz max

Fr_max = 6 * plant.th_r_max  # 955.2534
Fp_max = 2 * plant.th_p_max  # 183.1982
lbx[n_states * (N + 1) :: n_controls] = -Fr_max  # Fr min
ubx[n_states * (N + 1) :: n_controls] = 0  # Fr max
lbx[n_states * (N + 1) + 1 :: n_controls] = 0  # Fp min
ubx[n_states * (N + 1) + 1 :: n_controls] = Fp_max  # Fp max
lbx[n_states * (N + 1) + 2 :: n_controls] = -np.deg2rad(45)  # theta min
ubx[n_states * (N + 1) + 2 :: n_controls] = np.deg2rad(45)  # theta min

args = {
    "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
    "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
    "lbx": lbx,
    "ubx": ubx,
}

t0 = 0
state_init = ca.DM([z_init, vx_init, vz_init])
state_target = ca.DM([z_target, Vx_target, Vz_target])
control_init = ca.DM([-plant.m * plant.g, 0, 0])
control_target = ca.DM([Fr_target, Fp_target, theta_target])
t = ca.DM(t0)

# u0 = ca.DM.zeros((n_controls, N))  # initial control
u0 = ca.repmat(control_init, 1, N)
X0 = ca.repmat(state_init, 1, N + 1)  # initial state full
# X_ref = ca.repmat(state_target, 1, N + 1)
# u_ref = ca.repmat(control_target, 1, N)


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_states_ref = DM2Arr(state_target)
cat_controls_ref = DM2Arr(control_target)
# cat_controls = DM2Arr(u0[:, 0])
cat_controls = DM2Arr(u0)
times = np.array([[0]])


def set_ref(t, tf, v0, vf):
    VT_ref = v0 + (vf - v0) / tf * t

    upper_bound, lower_bound = boundary(VT_corr)
    degree = 3
    upper, lower, central = poly(degree, VT_corr, upper_bound, lower_bound)
    weighted = weighted_poly(degree, VT_corr, vf, upper, lower)

    theta_ref = np.deg2rad(weighted(VT_ref))

    Vx_target = VT_ref * np.cos(theta_ref)
    Vz_target = VT_ref * np.sin(theta_ref)
    states_target = ca.DM([z_target, Vx_target, Vz_target])
    control_target = ca.DM([0, 0, theta_ref])
    return states_target, control_target


if __name__ == "__main__":
    main_loop = time()  # return time in sec
    while mpc_iter * step_horizon < sim_time:
        t1 = time()

        current_time = mpc_iter * step_horizon
        state_target, control_target = set_ref(
            current_time, sim_time, VT_corr[0], VT_corr[-1]
        )

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

        # u_ref = ca.reshape(sol["p"][2 * n_states * N + 1 :], n_controls, N)
        X0 = ca.reshape(sol["x"][: n_states * (N + 1)], n_states, N + 1)
        u = ca.reshape(sol["x"][n_states * (N + 1) :], n_controls, N)

        cat_states = np.dstack((cat_states, DM2Arr(X0)))
        cat_states_ref = np.dstack((cat_states_ref, DM2Arr(state_target)))
        # cat_controls = np.dstack((cat_controls, DM2Arr(u[:, 0])))
        cat_controls = np.dstack((cat_controls, DM2Arr(u)))
        cat_controls_ref = np.dstack((cat_controls_ref, DM2Arr(control_target)))
        t = np.vstack((t, t0))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))

        t2 = time()
        # print(mpc_iter)
        # print(t2 - t1)
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
    np.save("cat_states_ref.npy", cat_states_ref)
    np.save("cat_controls.npy", cat_controls)

    """Fig 1. States"""
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, cat_states[0, 0, :], "k-")
    ax.plot(t, cat_states_ref[0, 0, :], "r--")
    ax.set_ylabel(r"$z$ [m]")
    ax.set_xlabel("Time [s]")
    ax.set_ylim([-60, -40])
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_states[1, 0, :], "k-")
    ax.plot(t, cat_states_ref[1, 0, :], "r--")
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, cat_states[2, 0, :], "k-")
    ax.plot(t, cat_states_ref[2, 0, :], "r--")
    ax.set_ylabel(r"$v_z$ [m/s]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Fig 2. Controls """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, -cat_controls[0, 0, :], "k-")
    ax.plot(t, cat_controls_ref[0, 0, :], "r--")
    ax.set_ylabel(r"$F_r$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_controls[1, 0, :], "k-")
    ax.plot(t, cat_controls_ref[1, 0, :], "r--")
    ax.set_ylabel(r"$F_p$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, np.rad2deg(cat_controls[2, 0, :]), "k-")
    ax.plot(t, np.rad2deg(cat_controls_ref[2, 0, :]), "r--")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    plt.show()
