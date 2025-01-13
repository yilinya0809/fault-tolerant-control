import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from fym.utils.rot import angle2dcm
from matplotlib.lines import Line2D

from ftc.models.LC62_opt import LC62
from ftc.trst_corr.poly_corr import boundary

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
)

plant = LC62()
Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max

Trst_corr = np.load("data/corr_safe_cause.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
cost = Trst_corr["cost"]
success = Trst_corr["success"]
Fr = Trst_corr["Fr"]
Fp = Trst_corr["Fp"]
Fx = Trst_corr["Fx"]
Fz = Trst_corr["Fz"]

# Check safety
eta = 0.8
Fr_margin = np.zeros((np.size(VT_corr), 1))
Fp_margin = np.zeros((np.size(VT_corr), 1))


for i in range(np.size(VT_corr)):
    for j in range(np.size(theta_corr)):
        if Fr[i, j] >= eta * Fr_max:
            Fr_margin[i] = theta_corr[j]

    for j in reversed(range(np.size(theta_corr))):
        if Fp[i, j] >= eta * Fp_max:
            Fp_margin[i] = theta_corr[j]

for i in range(np.size(VT_corr)):
    if Fr_margin[i, 0] == 0:
        Fr_margin[i, 0] = np.NaN
    if Fp_margin[i, 0] == 0:
        Fp_margin[i, 0] = np.NaN


# Optimal Trajectory
data = {}
with h5py.File("opt_corr.h5", "r") as f:
    data["tf"] = f["tf"][()]
    data["X"] = f["X"][:]
    data["U"] = f["U"][:]


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


""" Figure 1 """
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

degree = 3
upper_bound, lower_bound = boundary(Trst_corr)

mask = lower_bound > np.min(lower_bound)
VT_filtered = VT_corr[mask]
lower_bound_filtered = lower_bound[mask]

deg = 3
upper = np.polyfit(VT_corr, upper_bound, deg)
lower = np.polyfit(VT_filtered, lower_bound_filtered, deg)

VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT, theta, s=success.T, c="k")
ax.plot(
    VT_corr,
    np.rad2deg(upper_func(VT_corr)),
    "r-",
    label=r"$\mathrm{upper}(V)$",
    linewidth=5,
)
ax.plot(
    VT_corr,
    np.rad2deg(lower_func(VT_corr)),
    "b-",
    label=r"$\mathrm{lower}(V)$",
    linewidth=5,
)
ax.set_xlabel(r"$V, \mathrm{m/s}$", fontsize=20)
ax.set_ylabel(r"$\theta, \mathrm{deg}$", fontsize=20)
ax.set_xlim([0, 45])
ax.set_ylim([-30, 30])
ax.legend(fontsize=20)
fig.tight_layout()

""" Figure 2 """
fig, ax = plt.subplots(1, 1)
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT.T, acc_corr, s=3)
ax.set_xlabel("VT, m/s", fontsize=20)
ax.set_ylabel(r"$a_x, m/s^{2}$", fontsize=20)


""" Figure 3 """
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(VT, theta, acc_corr.T, cmap="plasma", edgecolor="none")
ax.contourf(VT, theta, acc_corr.T, zdir="z", offset=7, cmap="plasma")

""" Figure 4 - Trst 2D """
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
contour = ax.contourf(
    VT, theta, acc_corr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
ax.set_xlabel(r"$V,\, \mathrm{m/s}$", fontsize=20)
ax.set_ylabel(r"$\theta, \mathrm{deg}$", fontsize=20)
ax.set_aspect(0.62)
# ax.set_title("Forward Acceleration Corridor", fontsize=20)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$a_x^I,\, \mathrm{m/s^{2}}$", fontsize=20, labelpad=15)
fig.tight_layout()

""" Figure 5 - Fr, Fp """
# fig, axs = plt.subplots(1, 2, figsize=(18, 5), squeeze=False, sharex=True)
# ax = axs[0, 0]
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
contour = ax.contourf(
    VT, theta, Fr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
# ax.plot(VT_corr, Fr_margin, "r--")
ax.set_xlabel(r"$V,\, \mathrm{m/s}$", fontsize=20)
ax.set_ylabel(r"$\theta, \mathrm{deg}$", fontsize=20)
ax.set_aspect(0.62)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$F_{rotors},\, \mathrm{N}$", fontsize=20, labelpad=15)
fig.tight_layout()

""" Figure 6 - Fp """
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
contour = ax.contourf(
    VT, theta, Fp.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
ax.set_xlabel(r"$V,\, \mathrm{m/s}$", fontsize=20)
ax.set_ylabel(r"$\theta, \mathrm{deg}$", fontsize=20)
ax.set_aspect(0.62)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$F_{pushers},\, \mathrm{N}$", fontsize=20, labelpad=15)
fig.tight_layout()

""" Figure 7 - non-corridor """
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
cmap = mcolors.ListedColormap(["lightcoral", "green"])
sc1 = ax.scatter(VT, theta, s=50, c=Fz.T, cmap=cmap, alpha=0.8, label="Fz")

cmap = mcolors.ListedColormap(["k"])
sc2 = ax.scatter(VT, theta, s=5, c=Fx.T, cmap=cmap, alpha=0.4, label="Fx")

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=r"$F_x^I < 0$",
        markerfacecolor="black",
        markersize=10,
        alpha=0.8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=r"$F_z^I > 0$",
        markerfacecolor="lightcoral",
        markersize=10,
        # alpha=0.8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=r"$F_z^I < 0$",
        markerfacecolor="green",
        markersize=10,
        # alpha=0.6,
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    fontsize=20,
)
ax.set_xlabel(r"$V,\, \mathrm{m/s}$", fontsize=20)
ax.set_ylabel(r"$\theta, \mathrm{deg}$", fontsize=20)
ax.set_xlim([0, 45])
ax.set_ylim([-30, 30])
ax.set_aspect(0.55)
fig.tight_layout()


""" Figure 8 - Optimal Trajectory """
N = 200
tspan = linspace(0, data["tf"], N + 1)
z = data["X"][0, :]
vel = data["X"][1:, :]
Fr = data["U"][0, :]
Fp = data["U"][1, :]
theta = data["U"][2, :]

Fx_I = np.zeros((N, 1))
Fz_I = np.zeros((N, 1))
for i in range(N):
    R = angle2dcm(0, theta[i], 0)
    Fx, Fz = plant.B_Fuselage(vel[:, i])

    FB = np.vstack((Fp[i] + Fx, 0, -Fr[i] + Fz))
    F = R.T @ FB + np.vstack((0, 0, plant.m * plant.g))
    Fx_I[i] = F[0]
    Fz_I[i] = F[2]


""" States trajectory """
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
ax = axs[0]
ax.plot(tspan, -data["X"][0, :], "k", linewidth=3)
ax.set_ylabel("$h$, m", fontsize=20)
ax.set_ylim([5, 15])
ax.set_xlabel("Time, s", fontsize=20)
ax.grid()
ax.set_xlim([0, data["tf"]])
# fig.tight_layout()

# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = axs[1]
ax.plot(tspan, data["X"][1, :], "k", linewidth=3)
ax.set_ylabel("$V_x^B$, m/s", fontsize=20)
ax.set_xlim([0, data["tf"]])
# ax.set_xlabel("Time, s", fontsize=20)
ax.grid()
# fig.tight_layout()

# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = axs[2]
ax.plot(tspan, data["X"][2, :], "k", linewidth=3)
ax.set_ylabel("$V_z^B$, m/s", fontsize=20)
ax.set_xlabel("Time, s", fontsize=20)
ax.set_ylim([-10, 5])
ax.set_xlim([0, data["tf"]])
ax.grid()
fig.tight_layout()

""" Input trajectories """
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
ax = axs[0]
ax.plot(tspan[:-1], data["U"][0, :], "k", linewidth=3)
ax.plot(tspan[:-1], Fr_max * np.ones((N, 1)), "r--")
ax.plot(tspan[:-1], np.zeros((N, 1)), "r--")
ax.set_ylabel("$F^{rotor}$, N", fontsize=20)
# ax.set_xlabel("Time, s", fontsize=20)
ax.set_xlim([0, data["tf"]])
ax.grid()
# fig.tight_layout()

# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = axs[1]
ax.plot(tspan[:-1], data["U"][1, :], "k", linewidth=3)
ax.plot(tspan[:-1], Fp_max * np.ones((N, 1)), "r--")
ax.plot(tspan[:-1], np.zeros((N, 1)), "r--")
ax.set_ylabel("$F^{pusher}$, N", fontsize=20)
ax.set_xlim([0, data["tf"]])
# ax.set_xlabel("Time, s", fontsize=20)
ax.grid()
# fig.tight_layout()

# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = axs[2]
ax.plot(tspan[:-1], np.rad2deg(data["U"][2, :]), "k", linewidth=3)
ax.plot(tspan[:-1], -30 * np.ones((N, 1)), "r--")
ax.plot(tspan[:-1], 30 * np.ones((N, 1)), "r--")
ax.set_ylabel(r"$\theta$, deg", fontsize=20)
ax.set_xlabel("Time, s", fontsize=20)
ax.set_ylim([-35, 35])
ax.set_xlim([0, data["tf"]])
ax.grid()
fig.tight_layout()

""" Figure 9 - VT, theta traj """
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
VT_traj = np.zeros((N, 1))
theta_traj = np.zeros((N, 1))
for i in range(N):
    VT_traj[i] = norm_2(data["X"][1:3, i])
    theta_traj[i] = np.rad2deg(data["U"][2, i])

ax.plot(VT_traj[1:], theta_traj[1:], "r-", linewidth=5)
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT, theta, s=success.T, c="b")
ax.set_xlabel("V, m/s", fontsize=20)
ax.set_ylabel(r"$\theta$, deg", fontsize=20)
# ax.set_title("Dynamic Transition Corridor", fontsize=20)
fig.tight_layout()

""" Fx, Fz traj """
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.plot(tspan[:-1], Fx_I[:], "k", linewidth=3)
ax.set_ylabel(r"$F_x^I,\, \mathrm{N}$", fontsize=20)
ax.set_xlim([0, data["tf"]])
ax.grid()

ax = axs[1]
ax.plot(tspan[:-1], Fz_I[:], "k", linewidth=3)
ax.set_ylabel(r"$F_z^I,\, \mathrm{N}$", fontsize=20)
ax.set_xlim([0, data["tf"]])
ax.grid()
fig.tight_layout()


plt.show()
