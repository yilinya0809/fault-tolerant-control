import os

import casadi as ca
import fym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fym.utils.rot import angle2dcm, angle2quat, quat2angle, quat2dcm
from mpl_toolkits.mplot3d import axes3d
from numpy import cos, sin
from scipy.interpolate import interp1d

import ftc
from Corridor.poly_corr import boundary, poly, weighted_poly
from ftc.models.LC62S import LC62
from ftc.utils import safeupdate

Trst_corr = np.load("Corridor/data/corr_final.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
cost = Trst_corr["cost"]
success = Trst_corr["success"]


max_t = 10
dt = 0.1
N = max_t / dt

plant = LC62()

h_ref = 10
VT_ref = 45

z_init = -h_ref
Vx_init = 0
Vz_init = 0
theta_init = 0

X_trim, U_trim = plant.get_trim(fixed={"h": h_ref, "VT": VT_ref})
X_target = X_trim[1:]
U_target = U_trim


z = ca.MX.sym("z")
Vx = ca.MX.sym("Vx")
Vz = ca.MX.sym("Vz")
states = ca.vertcat(z, Vx, Vz)
n_states = states.numel()

Fr = ca.MX.sym("Fr")
Fp = ca.MX.sym("Fp")
theta = ca.MX.sym("theta")
controls = ca.vertcat(Fr, Fp, theta)
n_controls = controls.numel()

X = ca.MX.sym("X", n_states, N + 1)
U = ca.MX.sym("U", n_controls, N)
P = ca.MX.sym("P", 2 * n_states + n_controls)

Xdot = plant.deriv(states, controls)
f = ca.Function("f", [states, controls], [Xdot])
cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation

# terminal condition
phi = vertcat(X[:, N] - X_target, U[:, N] - U_target)
cost_fn = phi.T @ Q @ phi
# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    st_error = st - P[n_states : 2 * n_states]
    con_error = con - P[2 * n_states :]
    cost_fn = cost_fn + st_error.T @ Q @ st_error + con_error.T @ R @ con_error
    st_next = X[:, k + 1]
    k1 = f(st, con)
    k2 = f(st + step_horizon / 2 * k1, con)
    k3 = f(st + step_horizon / 2 * k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
opt_params = ca.vertcat(P.reshape((-1, 1)))
nlp_prob = {"f": cost_fn, "x": opt_variables, "g": g, "p": opt_params}

opts = {
    "ipopt": {
        "max_iter": 1000,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-6,
    },
    "print_time": 0,
}

solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)


efficiency = 0.7
