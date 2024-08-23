import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from scipy.optimize import curve_fit

import ftc
from ftc.models.LC62_optq import LC62
from ftc.trst_corr.poly_corr import boundary

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
W = diag([1, 1, 0])
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

    # tht = U[2, k]
    # VT = norm_2(X[1:, k])
    # opti.subject_to(
    #     opti.bounded(lower_func(VT), theta, upper_func(VT))
    # )  # close the gaps

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
opti.subject_to(opti.bounded(0, T, 20))

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
opti.set_initial(theta, X_target[2] / 2)
opti.set_initial(Fr, plant.m * plant.g / 2)
opti.set_initial(Fp, U_target[1] / 2)
opti.set_initial(q, U_target[2] / 2)
opti.set_initial(T, 20)

# ---- solve NLP              ------
opti.solver("ipopt")  # set numerical backend
sol = opti.solve()  # actual solve

# ---- post-processing        ------
from pylab import figure, grid, legend, plot, show, subplot, xlabel, ylabel

tf = sol.value(T)
tspan = linspace(0, tf, N + 1)
cost = sol.value(obj)

""" States trajectory """
fig, axs = plt.subplots(3, 2, squeeze=False, sharex=True)
ax = axs[0, 0]
ax.plot(tspan, -sol.value(z), "k")
ax.set_ylabel("$h$, m")
ax.grid()

ax = axs[1, 0]
ax.plot(tspan, sol.value(vx), "k")
ax.set_ylabel("$V_x$, m/s")
ax.grid()

ax = axs[2, 0]
ax.plot(tspan, sol.value(vz), "k")
ax.set_ylabel("$V_z$, m/s")
ax.set_xlabel("Time, s")
ax.grid()

ax = axs[0, 1]
ax.plot(tspan[:-1], sol.value(Fr), "k")
ax.set_ylabel("Rotor, N")
ax.grid()

ax = axs[1, 1]
ax.plot(tspan[:-1], sol.value(Fp), "k")
ax.set_ylabel("Pusher, N")
ax.grid()

ax = axs[2, 1]
ax.plot(tspan, np.rad2deg(sol.value(theta)), "k")
ax.set_ylabel(r"$\theta$, m/s")
ax.set_xlabel("Time, s")
ax.grid()


""" Cost trajectory """
fig, ax = plt.subplots(1, 1)
# ax.plot(tspan[:-1], cost, "k")
ylabel("Cost")
xlabel("Time, s")
grid()


""" Corridor trajectory """
# vx = sol.value(vx)
# vz = sol.value(vz)

# VT_traj = np.zeros((np.size(tspan), 1))
# theta_traj = np.rad2deg(sol.value(theta))
# for i in range(np.size(tspan)):
#     VT_traj[i] = np.linalg.norm(vx[i], vz[i])


# fig, ax = plt.subplots(1, 1)
# ax.plot(VT_traj, theta_traj, "k")
# ylabel("VT, m/s")
# xlabel(r"$\theta$, deg")
# grid()


# """ Corridor """
# Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
# VT_corr = Trst_corr["VT_corr"]
# acc_corr = Trst_corr["acc"]
# theta_corr = np.rad2deg(Trst_corr["theta_corr"])
# Fr = Trst_corr["Fr"]
# Fp = Trst_corr["Fp"]

# upper_bound, lower_bound = boundary(Trst_corr)
# VT, theta = np.meshgrid(VT_corr, theta_corr)

# upper = upper_func(VT_corr)
# lower = lower_func(VT_corr)
# fig, ax = plt.subplots(1, 1)
# ax.plot(VT_corr, upper, "r--", label="upper function")
# ax.plot(VT_corr, lower, "b--", label="lower function")


show()
