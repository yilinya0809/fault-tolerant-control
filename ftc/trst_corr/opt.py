import h5py
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

from ftc.models.LC62_opt import LC62
from ftc.trst_corr.poly_corr import boundary

""" Pre-processing - Transition Corridor """
Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
VT_corr = Trst_corr["VT_corr"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
success = Trst_corr["success"]
upper_bound, lower_bound = boundary(Trst_corr)

mask = lower_bound > np.min(lower_bound)
VT_filtered = VT_corr[mask]
lower_bound_filtered = lower_bound[mask]

deg = 3
upper = np.polyfit(VT_corr, upper_bound, deg)
lower = np.polyfit(VT_filtered, lower_bound_filtered, deg)


def casadi_polyval(coeffs, x):
    value = 0
    deg = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        value += coeff * x ** (deg - i)
    return value


def upper_func(vel):
    value = casadi_polyval(upper, vel)
    return value


def lower_func(vel):
    value_min = np.min(lower_bound)
    value_low = casadi_polyval(lower, vel)
    value = if_else(vel < VT_filtered[0], value_min, value_low)
    return value


upper_data = np.zeros((np.size(VT_corr), 1))
lower_data = np.zeros((np.size(VT_corr), 1))
for i in range(np.size(VT_corr)):
    upper_data[i] = upper_func(VT_corr[i])
    lower_data[i] = lower_func(VT_corr[i])


plant = LC62()

""" Get trim """
x_trim, u_trim = plant.get_trim()

""" Optimization """
N = 200  # number of control intervals

opti = Opti()  # Optimization problem

# ---- decision variables ---------
X = opti.variable(3, N + 1)  # state trajectory
z = X[0, :]
vx = X[1, :]
vz = X[2, :]
U = opti.variable(3, N)  # control trajectory (throttle)
Fr = U[0, :]
Fp = U[1, :]
theta = U[2, :]
T = opti.variable()

# ---- objective          ---------
W_t = 1000
W_z = 50000
W_u = diag([1, 10, 500000])

cost = W_t * T

dt = T / N
for k in range(N):  # loop over control intervals
    # Runge-Kutta 4 integration
    # if k > 0.7 * N:
    #     W_z = 500000
    #     W_u = diag([10, 10, 500000])
    cost += U[:, k].T @ W_u @ U[:, k] * dt
    q = 0.0
    k1 = plant.derivq(X[:, k], U[:, k], q)
    k2 = plant.derivq(X[:, k] + dt / 2 * k1, U[:, k], q)
    k3 = plant.derivq(X[:, k] + dt / 2 * k2, U[:, k], q)
    k4 = plant.derivq(X[:, k] + dt * k3, U[:, k], q)

    # k1 = plant.deriv(X[:, k], U[:, k])
    # k2 = plant.deriv(X[:, k] + dt / 2 * k1, U[:, k])
    # k3 = plant.deriv(X[:, k] + dt / 2 * k2, U[:, k])
    # k4 = plant.deriv(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # zdot = x_next[0] - X[0, k]
    # cost += W_z * zdot ** 2
    cost += W_z * (X[0, k] - x_trim[1]) ** 2

    # Transition Corridor
    theta_k = U[2, k]
    VT_k = norm_2(X[1:3, k])
    opti.subject_to(opti.bounded(lower_func(VT_k), theta_k, upper_func(VT_k)))


# opti.minimize(T)
opti.minimize(cost)

Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max
theta_max = np.deg2rad(30)
# ---- input constraints --------
opti.subject_to(opti.bounded(0, Fr, Fr_max))
opti.subject_to(opti.bounded(0, Fp, Fp_max))
opti.subject_to(opti.bounded(-theta_max, theta, theta_max))

# ---- state constraints --------
z_eps = 1
opti.subject_to(opti.bounded(x_trim[1] - z_eps, z, x_trim[1] + z_eps))
# opti.subject_to(opti.bounded(0, T, 20))
opti.subject_to(T >= 0)

# ---- boundary conditions --------
opti.subject_to(z[0] == x_trim[1])
opti.subject_to(vx[0] == 0)
opti.subject_to(vz[0] == 0)
opti.subject_to(Fr[0] == plant.m * plant.g)
opti.subject_to(Fp[0] == 0)
opti.subject_to(theta[0] == np.deg2rad(0))

opti.subject_to(z[-1] == x_trim[1])
opti.subject_to(vx[-1] == x_trim[2])
opti.subject_to(vz[-1] == x_trim[3])

u_eps = 0.5
opti.subject_to(opti.bounded(0, Fr[-1], 10))
opti.subject_to(opti.bounded(u_trim[1] * (1 - u_eps), Fp[-1], u_trim[1] * (1 + u_eps)))
opti.subject_to(
    opti.bounded(u_trim[2] * (1 - u_eps), theta[-1], u_trim[2] * (1 + u_eps))
)

# opti.subject_to(Fr[-1] == u_trim[0])
# opti.subject_to(Fp[-1] == u_trim[1])
# opti.subject_to(theta[-1] == u_trim[2])


with h5py.File("ftc/trst_corr/opt.h5", "r") as f:
    tf_init = f["tf"][()]
    X_init = f["X"][:]
    U_init = f["U"][:]
# cost = f["cost"]

# ---- initial values for solver ---
opti.set_initial(T, tf_init)
opti.set_initial(z, X_init[0, :])
opti.set_initial(vx, X_init[1, :])
opti.set_initial(vz, X_init[2, :])
opti.set_initial(Fr, U_init[0, :])
opti.set_initial(Fp, U_init[1, :])
opti.set_initial(theta, U_init[2, :])
# opti.set_initial(T, 20)
# opti.set_initial(z, x_trim[1])
# opti.set_initial(vx, x_trim[2] / 2)
# opti.set_initial(vz, x_trim[3] / 2)
# opti.set_initial(Fr, plant.m * plant.g / 2)
# opti.set_initial(Fp, u_trim[1] / 2)
# opti.set_initial(theta, u_trim[2] / 2)


# ---- solve NLP              ------
p_opts = {"expand": False}
s_opts = {
    "tol": 1e-1,
    "acceptable_tol": 1e-1,
    "acceptable_iter": 15,
    "max_iter": 2000,
    "max_cpu_time": 1e4,
    "print_level": 5,
}


opti.solver("ipopt", p_opts, s_opts)  # set numerical backend

results = {}


def plot_results(data):
    tspan = linspace(0, data["tf"], N + 1)

    """ States trajectory """
    fig, axs = plt.subplots(3, 1, squeeze=False, sharex=True)
    ax = axs[0, 0]
    ax.plot(tspan, -data["X"][0, :], "k", linewidth=3)
    ax.set_ylabel("$h$, m", fontsize=15)
    ax.set_ylim([9, 11])
    ax.grid()
    ax.set_xlim([0, data["tf"]])

    ax = axs[1, 0]
    ax.plot(tspan, data["X"][1, :], "k", linewidth=3)
    ax.set_ylabel("$V_x^B$, m/s", fontsize=15)
    ax.set_xlim([0, data["tf"]])
    ax.grid()

    ax = axs[2, 0]
    ax.plot(tspan, data["X"][2, :], "k", linewidth=3)
    ax.set_ylabel("$V_z^B$, m/s", fontsize=15)
    ax.set_xlabel("Time, s", fontsize=15)
    ax.set_ylim([-10, 10])
    ax.set_xlim([0, data["tf"]])
    ax.grid()

    fig, axs = plt.subplots(3, 1, squeeze=False, sharex=True)
    ax = axs[0, 0]
    ax.plot(tspan[:-1], data["U"][0, :], "k", linewidth=3)
    ax.plot(tspan[:-1], Fr_max * np.ones((N, 1)), "r--")
    ax.set_ylabel("$F^{rotor}$, N", fontsize=15)
    ax.set_xlim([0, data["tf"]])
    ax.grid()

    ax = axs[1, 0]
    ax.plot(tspan[:-1], data["U"][1, :], "k", linewidth=3)
    ax.plot(tspan[:-1], Fp_max * np.ones((N, 1)), "r--")
    ax.set_ylabel("$F^{pusher}$, N", fontsize=15)
    ax.set_xlim([0, data["tf"]])
    ax.grid()

    ax = axs[2, 0]
    ax.plot(tspan[:-1], np.rad2deg(data["U"][2, :]), "k", linewidth=3)
    ax.plot(tspan[:-1], -30 * np.ones((N, 1)), "r--")
    ax.plot(tspan[:-1], 30 * np.ones((N, 1)), "r--")
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)
    ax.set_xlabel("Time, s", fontsize=15)
    ax.set_ylim([-35, 35])
    ax.set_xlim([0, data["tf"]])
    ax.grid()

    """ VT, theta traj """
    fig, ax = plt.subplots(1, 1)
    VT_traj = np.zeros((N, 1))
    theta_traj = np.zeros((N, 1))
    for i in range(N):
        VT_traj[i] = norm_2(data["X"][1:3, i])
        theta_traj[i] = np.rad2deg(data["U"][2, i])

    ax.plot(VT_traj[1:], theta_traj[1:], "r-", linewidth=5)
    VT, theta = np.meshgrid(VT_corr, theta_corr)
    ax.scatter(VT, theta, s=success.T, c="b")
    ax.set_xlabel("V, m/s", fontsize=15)
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)
    ax.set_title("Dynamic Transition Corridor", fontsize=20)

    #     """ Cost plot """
    #     fig, ax = plt.subplots(1, 1)
    #     ax.plot(range(len(data["cost"])), data["cost"])
    #     ax.set_xlabel("Iteration", fontsize=15)
    #     ax.set_ylabel("Cost", fontsize=15)
    #     ax.grid()

    plt.show()


try:
    sol = opti.solve()  # actual solve
    results["tf"] = sol.value(T)
    results["X"] = sol.value(X)
    results["U"] = sol.value(U)
    stats = opti.stats()

    iter_costs = stats["iterations"]["obj"]
    iter_primal_infeas = stats["iterations"]["inf_pr"]
    iter_dual_infeas = stats["iterations"]["inf_du"]

    cost = []
    for i in range(len(iter_costs)):
        if (
            iter_primal_infeas[i] < s_opts["tol"]
            and iter_dual_infeas[i] < s_opts["tol"]
        ):
            cost.append(iter_costs[i])

    results["cost"] = cost

    with h5py.File("opt.h5", "w") as f:
        f.create_dataset("tf", data=results["tf"])
        f.create_dataset("X", data=results["X"])
        f.create_dataset("U", data=results["U"])
        f.create_dataset("cost", data=results["cost"])
        # f.create_dataset("Fr_max", data=)
        # f.create_dataset("Fp_max", data=Fp_max * np.ones((len(results["tf"], 1)))
    plot_results(results)

except RuntimeError as e:
    print("Solver Failed!")
    results["tf"] = float(opti.debug.value(T))
    results["X"] = opti.debug.value(X)
    results["U"] = opti.debug.value(U)
    stats = opti.stats()
    results["cost"] = stats["iterations"]["obj"]

    t_final = results["tf"]
    z_final = results["X"][0, :]
    vx_final = results["X"][1, :]
    vz_final = results["X"][2, :]
    Fr_final = results["U"][0, :]
    Fp_final = results["U"][1, :]
    theta_final = results["U"][2, :]

    if np.any(z_final < x_trim[1] - z_eps) or np.any(z_final > x_trim[1] + z_eps):
        print("Altitude constraint violated")

    if np.any(Fr_final < 0) or np.any(Fr_final > Fr_max):
        print("Fr constraint violated")
    if np.any(Fp_final < 0) or np.any(Fp_final > Fp_max):
        print("Fp constraint violated")
    if np.any(theta_final < -theta_max) or np.any(theta_final > theta_max):
        print("theta constraint violated")

    if (Fr_final[-1] < u_trim[0] * (1 - u_eps)) or (
        Fr_final[-1] > u_trim[0] * (1 + u_eps)
    ):
        print("Fr terminal condition violated")

    if (Fp_final[-1] < u_trim[1] * (1 - u_eps)) or (
        Fp_final[-1] > u_trim[1] * (1 + u_eps)
    ):
        print("Fp terminal condition violated")

    if (theta_final[-1] < u_trim[2] * (1 - u_eps)) or (
        theta_final[-1] > u_trim[2] * (1 + u_eps)
    ):
        print("theta terminal condition violated")

    plot_results(results)
