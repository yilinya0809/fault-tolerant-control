import h5py
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from ftc.models.LC62_opt import LC62
from ftc.trst_corr.poly_corr import boundary

plant = LC62()
Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max

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



""" Results of Outer-loop optimal trajectory """
opt_traj = {}
f = h5py.File('ftc/trst_corr/opt.h5', 'r')
opt_traj["tf"] = f.get('tf')[()]
opt_traj["X"] = f.get('X')[:]
opt_traj["U"] = f.get('U')[:]

def plot_results(data):
    N = np.shape(data["U"])[1]
    tspan = np.linspace(0, data["tf"], N + 1)

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


    plt.show()


if __name__ == "__main__":
    plot_results(opt_traj)

