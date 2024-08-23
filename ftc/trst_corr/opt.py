import matplotlib.pyplot as plt
from casadi import *

from ftc.models.LC62_opt import LC62

# from dyn import LC62


plant = LC62()

""" Get trim """
x_trim, u_trim = plant.get_trim()

""" Optimization """
N = 100  # number of control intervals

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
W = diag([1, 1, 10000])
opti.minimize(dot(U, W @ U))

dt = T / N
for k in range(N):  # loop over control intervals
    # Runge-Kutta 4 integration
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

Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max
# ---- input constraints --------
opti.subject_to(opti.bounded(0, Fr, Fr_max))
opti.subject_to(opti.bounded(0, Fp, Fp_max))
opti.subject_to(opti.bounded(np.deg2rad(-50), theta, np.deg2rad(50)))

# ---- state constraints --------
z_eps = 2
opti.subject_to(opti.bounded(x_trim[1] - z_eps, z, x_trim[1] + z_eps))
opti.subject_to(opti.bounded(0, T, 20))

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
opti.subject_to(Fr[-1] == u_trim[0])
opti.subject_to(Fp[-1] == u_trim[1])
opti.subject_to(theta[-1] == u_trim[2])

# ---- initial values for solver ---
opti.set_initial(z, x_trim[1])
opti.set_initial(vx, x_trim[2] / 2)
opti.set_initial(vz, x_trim[3] / 2)
opti.set_initial(Fr, plant.m * plant.g / 2)
opti.set_initial(Fp, u_trim[1] / 2)
opti.set_initial(theta, u_trim[2] / 2)
opti.set_initial(T, 20)

# ---- solve NLP              ------
opti.solver("ipopt")  # set numerical backend
sol = opti.solve()  # actual solve

# ---- post-processing        ------
from pylab import figure, legend, plot, show

tf = sol.value(T)
tspan = linspace(0, tf, N + 1)

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
ax.plot(tspan[:-1], np.rad2deg(sol.value(theta)), "k")
ax.set_ylabel(r"$\theta$, m/s")
ax.set_xlabel("Time, s")
ax.grid()


plt.show()
