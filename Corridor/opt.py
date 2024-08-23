from casadi import *

from dyn import LC62

plant = LC62()

""" Get trim """
X_trim, U_trim = plant.get_trim()

""" Optimization """
N = 100  # number of control intervals

opti = Opti()  # Optimization problem

# ---- decision variables ---------
X = opti.variable(4, N + 1)  # state trajectory
x = X[0, :]
z = X[1, :]
vx = X[2, :]
vz = X[3, :]
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
    k1 = plant.deriv(X[:, k], U[:, k])
    k2 = plant.deriv(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = plant.deriv(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = plant.deriv(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

# ---- input constraints --------
opti.subject_to(opti.bounded(0, Fr, 800))
opti.subject_to(opti.bounded(0, Fp, 200))
opti.subject_to(opti.bounded(np.deg2rad(-50), theta, np.deg2rad(50)))

# ---- state constraints --------
# z_eps = 2
# opti.subject_to(opti.bounded(x_trim[0] - z_eps, z, x_trim[0] + z_eps))
opti.subject_to(-z >= 0)
opti.subject_to(opti.bounded(0, T, 20))

# ---- boundary conditions --------
opti.subject_to(x[0] == 0)
opti.subject_to(z[0] == 0)
opti.subject_to(vx[0] == 0)
opti.subject_to(vz[0] == 0)
opti.subject_to(Fr[0] == plant.m * plant.g)
opti.subject_to(Fp[0] == 0)
opti.subject_to(theta[0] == np.deg2rad(0))

opti.subject_to(z[-1] == X_trim[1])
opti.subject_to(vx[-1] == X_trim[2])
opti.subject_to(vz[-1] == X_trim[3])
opti.subject_to(Fr[-1] == U_trim[0])
opti.subject_to(Fp[-1] == U_trim[1])
opti.subject_to(theta[-1] == U_trim[2])

# ---- initial values for solver ---
opti.set_initial(z, X_trim[1])
opti.set_initial(vx, X_trim[2] / 2)
opti.set_initial(vz, X_trim[3] / 2)
opti.set_initial(Fr, plant.m * plant.g / 2)
opti.set_initial(Fp, U_trim[1] / 2)
opti.set_initial(theta, U_trim[2] / 2)
opti.set_initial(T, 20)

# ---- solve NLP              ------
opti.solver("ipopt")  # set numerical backend
sol = opti.solve()  # actual solve

# ---- post-processing        ------
from pylab import figure, grid, legend, plot, show, subplot, xlabel, ylabel

tf = sol.value(T)
tspan = linspace(0, tf, N + 1)

figure()
subplot(2, 1, 1)
plot(tspan, sol.value(x), "k")
ylabel("$x$, m")
grid()
subplot(2, 1, 2)
plot(tspan, -sol.value(z), "k")
ylabel("$h$, m")
xlabel("Time, s")
grid()

figure()
subplot(2, 1, 1)
plot(tspan, sol.value(vx), "k")
ylabel("$v_x$, m/s")
grid()
subplot(2, 1, 2)
plot(tspan, sol.value(vz), "k")
ylabel("$v_z$, m/s")
xlabel("Time, s")
grid()

figure()
subplot(3, 1, 1)
plot(tspan[:-1], sol.value(Fr), "k")
ylabel("Rotor, N")
grid()
subplot(3, 1, 2)
plot(tspan[:-1], sol.value(Fp), "k")
ylabel("Pusher, N")
grid()
subplot(3, 1, 3)
plot(tspan[:-1], np.rad2deg(sol.value(theta)), "k")
ylabel(r"$\theta$, deg")
xlabel("Time, s")
grid()

show()
