import fym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fym.utils.rot import angle2quat, quat2angle, quat2dcm
from mpl_toolkits.mplot3d import axes3d
from numpy import cos, sin
from scipy.interpolate import interp1d

from Corridor.poly_corr import boundary, poly, weighted_poly

Trst_corr = np.load("Corridor/data_final/corr.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
cost = Trst_corr["cost"]
success = Trst_corr["success"]

# """ Figure 1 - Trim """
# theta_trim = theta_corr[:, 0]
# fig, ax = plt.subplots(1, 1)
# ax.plot(VT_corr, theta_trim)
# ax.set_xlabel("VT, m/s", fontsize=15)
# ax.set_ylabel(r"$\theta$, deg", fontsize=15)
# ax.set_title("Trim condition", fontsize=20)

# """ Figure 1 - cost """
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# VT, acc = np.meshgrid(VT_corr, acc_corr)
# ax.scatter(acc, VT, cost.T, cmap="plasma", edgecolor="none")
# # ax.contourf(acc, VT, cost.T, zdir='x', offset=7, cmap='plasma')
# ax.set_xlabel(r"$a, m/s^{2}$", fontsize=15)
# ax.set_ylabel("VT, m/s", fontsize=15)
# ax.set_zlabel(r"cost", fontsize=15)

# """ Figure 2 - Trst 3D """
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# VT, acc = np.meshgrid(VT_corr, acc_corr)
# ax.plot_surface(acc, VT, theta_corr.T, cmap="plasma", edgecolor="none")
# ax.contourf(acc, VT, theta_corr.T, zdir="x", offset=10, cmap="plasma")
# ax.set_xlabel(r"$a, m/s^{2}$", fontsize=15)
# ax.set_ylabel("VT, m/s", fontsize=15)
# ax.set_zlabel(r"$\theta$, deg", fontsize=15)
# ax.set_title("Transition Corridor", fontsize=20)


# """ Figure 3 - Trst 2D """
# fig = plt.figure()
# ax = fig.add_subplot(111)
# contour = ax.contourf(
#     VT, theta_corr.T, acc, levels=np.shape(theta_corr)[0], cmap="plasma"
# )
# ax.set_xlabel("VT, m/s", fontsize=15)
# ax.set_ylabel(r"$\theta$, deg", fontsize=15)
# ax.set_title("Dynamic Transition Corridor", fontsize=20)
# cbar = fig.colorbar(contour)
# cbar.ax.set_xlabel(r"$a, m/s^{2}$")

# """ Figure 4 """
# fig = plt.figure()
# ax = fig.add_subplot(111)
# contour = ax.contourf(
#     acc, theta_corr.T, VT, levels=np.shape(theta_corr)[1], cmap="plasma"
# )
# ax.set_xlabel(r"$a, m/s^{2}$", fontsize=15)
# ax.set_ylabel(r"$\theta$, deg", fontsize=15)
# ax.set_title("Dynamic Transition Corridor", fontsize=20)
# cbar = fig.colorbar(contour)
# cbar.ax.set_xlabel("VT, m/s")

""" Figure 1 """
fig, ax = plt.subplots(1, 1)
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT, theta, success.T)
ax.set_xlabel("VT, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Dynamic Transition Corridor", fontsize=20)

# ax = axs[1]
# degree = 3
# upper_bound, lower_bound = boundary(Trst_corr)
# upper, lower, central = poly(degree, Trst_corr, upper_bound, lower_bound)

# VT_target = VT_corr[-1]
# weighted = weighted_poly(degree, Trst_corr, VT_target, upper, lower)

# ax.plot(VT_corr, upper_bound, "o", label="Upper Bound Data", color="blue", alpha=0.3)
# ax.plot(VT_corr, lower_bound, "o", label="Lower Bound Data", color="orange", alpha=0.3)
# ax.plot(VT_corr, upper(VT_corr), "b--", label="Upper Bound Polynomial")
# ax.plot(VT_corr, lower(VT_corr), "y--", label="Lower Bound Polynomial")
# ax.plot(VT_corr, weighted(VT_corr), "k-", label="Weighted Line")
# ax.set_xlabel("VT, m/s", fontsize=15)
# ax.set_ylabel("θ, deg", fontsize=15)
# ax.legend()
# ax.grid()

""" Figure 2 """
fig, ax = plt.subplots(1, 1)
# ax = axs[1]
VT, theta = np.meshgrid(VT_corr, theta_corr)
ax.scatter(VT.T, acc_corr, s=3)
ax.set_xlabel("VT, m/s", fontsize=15)
ax.set_ylabel(r"$a_x, m/s^{2}$", fontsize=15)

plt.show()
