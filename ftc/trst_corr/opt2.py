import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from scipy.optimize import curve_fit

import ftc
# consider q as control input
from ftc.models.LC62_optq import LC62
from ftc.trst_corr.poly_corr import boundary

Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
VT_corr = Trst_corr["VT_corr"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
cost = Trst_corr["cost"]
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


# fig, ax = plt.subplots(1, 1)
# VT, theta = np.meshgrid(VT_corr, np.rad2deg(theta_corr))
# ax.scatter(VT, theta, s=success.T, c="b")
# ax.plot(VT_corr, np.rad2deg(upper_data))
# ax.plot(VT_corr, np.rad2deg(lower_data))
# ax.set_xlabel("VT, m/s", fontsize=15)
# ax.set_ylabel(r"$\theta$, deg", fontsize=15)
# ax.set_title("Dynamic Transition Corridor", fontsize=20)

# plt.show()

# breakpoint()

plant = LC62()

h_ref = 10
VT_ref = 45

N = 100

X_trim, U_trim = plant.get_trim(fixed={"h": h_ref, "VT": VT_ref})
X_target = X_trim[1:]
U_target = U_trim

opti = Opti()  # Optimization problem

# ---- decision variables ---------
X = opti.variable(4, N + 1)  # state trajectory
z = X[0, :]
vx = X[1, :]
vz = X[2, :]
theta = X[3, :]

U = opti.variable(3, N)  # control trajectory (throttle)
Fr = U[0, :]
Fp = U[1, :]
q = U[2, :]
T = opti.variable()

# ---- objective          ---------
Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max
q_max = np.deg2rad(50)


W = diag([1, 1, 10000])
# W = diag([1/Fr_max, 1/Fp_max, 1/q_max])
obj = dot(U, W @ U)
opti.minimize(obj)

dt = T / N
for k in range(N):  # loop over control intervals
    # Runge-Kutta 4 integration
    k1 = plant.deriv(X[:, k], U[:, k])
    k2 = plant.deriv(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = plant.deriv(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = plant.deriv(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # Transition Corridor
    theta_k = X[3, k]
    VT_k = norm_2(X[1:3, k])
    opti.subject_to(opti.bounded(lower_func(VT_k), theta_k, upper_func(VT_k)))


Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max

# ---- input constraints --------
opti.subject_to(opti.bounded(0, Fr, Fr_max))
opti.subject_to(opti.bounded(0, Fp, Fp_max))
opti.subject_to(opti.bounded(np.deg2rad(-50), theta, np.deg2rad(50)))

# ---- state constraints --------
z_eps = 2
opti.subject_to(opti.bounded(X_target[0] - z_eps, z, X_target[0] + z_eps))
# opti.subject_to(-z >= 0)
opti.subject_to(opti.bounded(0, T, 50))

# ---- boundary conditions --------
opti.subject_to(z[0] == X_target[0])
opti.subject_to(vx[0] == 0)
opti.subject_to(vz[0] == 0)
opti.subject_to(theta[0] == np.deg2rad(0))
opti.subject_to(Fr[0] == plant.m * plant.g)
opti.subject_to(Fp[0] == 0)
opti.subject_to(q[0] == np.deg2rad(0))

opti.subject_to(z[-1] == X_target[0])
opti.subject_to(vx[-1] == X_target[1])
opti.subject_to(vz[-1] == X_target[2])
opti.subject_to(theta[-1] == X_target[3])
opti.subject_to(Fr[-1] == U_target[0])
opti.subject_to(Fp[-1] == U_target[1])
opti.subject_to(q[-1] == U_target[2])

# ---- initial values for solver ---
opti.set_initial(z, X_target[0])
opti.set_initial(vx, X_target[1] / 2)
opti.set_initial(vz, X_target[2] / 2)
opti.set_initial(theta, X_target[3] / 2)
opti.set_initial(Fr, plant.m * plant.g / 2)
opti.set_initial(Fp, U_target[1] / 2)
opti.set_initial(q, U_target[2] / 2)
opti.set_initial(T, 50)

# ---- solve NLP              ------
opti.solver("ipopt")  # set numerical backend
results = {}


def plot_results(data):
    tspan = linspace(0, data["tf"], N + 1)

    """ States trajectory """
    fig, axs = plt.subplots(3, 2, squeeze=False, sharex=True)
    ax = axs[0, 0]
    ax.plot(tspan, -data["X"][0, :], "k")
    ax.set_ylabel("$h$, m")
    ax.grid()

    ax = axs[1, 0]
    ax.plot(tspan, data["X"][1, :], "k")
    ax.set_ylabel("$V_x$, m/s")
    ax.grid()

    ax = axs[2, 0]
    ax.plot(tspan, data["X"][2, :], "k")
    ax.set_ylabel("$V_z$, m/s")
    ax.set_xlabel("Time, s")
    ax.grid()

    ax = axs[0, 1]
    ax.plot(tspan[:-1], data["U"][0, :], "k")
    ax.set_ylabel("Rotor, N")
    ax.grid()

    ax = axs[1, 1]
    ax.plot(tspan[:-1], data["U"][1, :], "k")
    ax.set_ylabel("Pusher, N")
    ax.grid()

    ax = axs[2, 1]
    ax.plot(tspan, np.rad2deg(data["X"][3, :]), "k")
    ax.set_ylabel(r"$\theta$, m/s")
    ax.set_xlabel("Time, s")
    ax.grid()

    """ VT, theta traj """
    fig, ax = plt.subplots(1, 1)
    VT_traj = np.zeros((N, 1))
    theta_traj = np.zeros((N, 1))
    for i in range(N):
        VT_traj[i] = norm_2(data["X"][1:3, i])
        theta_traj[i] = np.rad2deg(data["X"][3, i])

    ax.plot(VT_traj, theta_traj, "r--")
    VT, theta = np.meshgrid(VT_corr, theta_corr)
    ax.scatter(VT, theta, s=success.T, c="b")
    ax.set_xlabel("VT, m/s", fontsize=15)
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)
    ax.set_title("Dynamic Transition Corridor", fontsize=20)

    """ Cost plot """
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(data["cost"])), data["cost"])
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel("Cost", fontsize=15)


    plt.show()


try:
    sol = opti.solve()  # actual solve
    results["tf"] = sol.value(T)
    results["X"] = sol.value(X)
    results["U"] = sol.value(U)
    stats = opti.stats()
    results["cost"] = stats["iterations"]["obj"]
    plot_results(results)


except RuntimeError as e:
    print("Solver Failed!")
    results["tf"] = float(opti.debug.value(T))
    results["X"] = opti.debug.value(X)
    results["U"] = opti.debug.value(U)
    stats = opti.stats()
    results["cost"] = stats["iterations"]["obj"]
    plot_results(results)