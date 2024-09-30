import matplotlib.pyplot as plt
import numpy as np

from ftc.models.LC62_opt import LC62
from ftc.trst_corr.poly_corr import boundary, poly, weighted_poly

plant = LC62()
Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max

# Using maximum power of rotors and pushers
# Trst_corr = np.load("ftc/trst_corr/corr.npz")

# Using 80% of maximum power of rotors and pushers
Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
cost = Trst_corr["cost"]
success = Trst_corr["success"]
Fr = Trst_corr["Fr"]
Fp = Trst_corr["Fp"]

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


""" Figure 1 """
fig, axs = plt.subplots(1, 2)
ax = axs[0]
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT, theta, s=success.T, c="b")
ax.set_xlabel("VT, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Dynamic Transition Corridor", fontsize=20)

ax = axs[1]
degree = 3
upper_bound, lower_bound = boundary(Trst_corr)
upper, lower, central = poly(degree, Trst_corr, upper_bound, lower_bound)

VT_target = VT_corr[-1]
weighted = weighted_poly(degree, Trst_corr, VT_target, upper, lower)

ax.plot(VT_corr, upper_bound, "o", label="Upper Bound Data", color="blue", alpha=0.3)
ax.plot(VT_corr, lower_bound, "o", label="Lower Bound Data", color="orange", alpha=0.3)
ax.plot(VT_corr, upper(VT_corr), "b--", label="Upper Bound Polynomial")
ax.plot(VT_corr, lower(VT_corr), "y--", label="Lower Bound Polynomial")
ax.plot(VT_corr, weighted(VT_corr), "k-", label="Weighted Line")
ax.set_xlabel("VT, m/s", fontsize=15)
ax.set_ylabel("θ, deg", fontsize=15)
ax.legend()
ax.grid()

""" Figure 2 """
fig, ax = plt.subplots(1, 1)
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT.T, acc_corr, s=3)
ax.set_xlabel("VT, m/s", fontsize=15)
ax.set_ylabel(r"$a_x, m/s^{2}$", fontsize=15)


""" Figure 3 """
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(VT, theta, acc_corr.T, cmap="plasma", edgecolor="none")
ax.contourf(VT, theta, acc_corr.T, zdir="z", offset=7, cmap="plasma")

""" Figure 4 - Trst 2D """
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(
    VT, theta, acc_corr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
ax.set_xlabel("V, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Forward Acceleration Corridor", fontsize=20)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$a_x, m/s^{2}$", fontsize=15)

""" Figure 5 - Fr, Fp """
fig, axs = plt.subplots(1, 2, figsize=(18, 5), squeeze=False, sharex=True)
ax = axs[0, 0]
contour = ax.contourf(
    VT, theta, Fr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
ax.plot(VT_corr, Fr_margin, "r--")
ax.set_xlabel("V, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Rotor Force Corridor", fontsize=20)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$Fr, N$", fontsize=15)

ax = axs[0, 1]
contour = ax.contourf(
    VT, theta, Fp.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
)
ax.plot(VT_corr, Fp_margin, "r--")
ax.set_xlabel("V, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Pusher Force Corridor", fontsize=20)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$Fp, N$", fontsize=15)

fig.tight_layout()
fig.subplots_adjust(wspace=0.2)

plt.show()
