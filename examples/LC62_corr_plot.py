import fym
import numpy as np
import scipy
from fym.utils.rot import angle2quat, quat2dcm, quat2angle
from numpy import cos, sin
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

Trst_corr = np.load('corr.npz')
VT_corr = Trst_corr['VT_corr']
acc_corr = Trst_corr['acc_corr']
theta_corr = Trst_corr['theta_corr']

""" Figure 1 - Trim """
theta_trim = theta_corr[:, np.where(acc_corr == 0)].squeeze(-1)
fig, ax = plt.subplots(1, 1)
ax.plot(VT_corr, theta_trim)
ax.set_xlabel("V, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Trim condition", fontsize=20)

""" Figure 2 - Trst 3D """
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
VT, acc = np.meshgrid(VT_corr, acc_corr)
ax.plot_surface(acc, VT, theta_corr.T, cmap='plasma', edgecolor='none')
ax.contourf(acc, VT, theta_corr.T, zdir='x', offset=6, cmap='plasma')
ax.set_xlabel(r"$a_x, m/s^{2}$", fontsize=15)
ax.set_ylabel("V, m/s", fontsize=15)
ax.set_zlabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Transition Corridor", fontsize=20)

""" Figure 3 - Trst 2D """
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(VT, theta_corr.T, acc, levels=np.shape(theta_corr)[0], cmap='plasma')
ax.set_xlabel("V, m/s", fontsize=15)
ax.set_ylabel(r"$\theta$, deg", fontsize=15)
ax.set_title("Dynamic Transition Corridor", fontsize=20)
cbar = fig.colorbar(contour)
cbar.ax.set_xlabel(r"$a_x, m/s^{2}$")

plt.show()



