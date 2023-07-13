import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.LC62S import LC62

import fym
from fym.utils.rot import quat2angle

def DM2Arr(dm):
    return np.array(dm.full())


plant = LC62()

Fr_max = 6 * plant.th_r_max
Fp_max = 2 * plant.th_p_max
theta_max = np.deg2rad(45)

step_horizon = 0.1  # time between steps in seconds
N = 10  # number of look ahead steps
sim_time = 10  # [sec]

z_init = -10
vx_init = 0
vz_init = 0

""" Get trim """
X_trim, U_trim = plant.get_trim(fixed={"h": 10, "VT": 45})
_, z_target, vx_target, vz_target = X_trim.ravel()
Fr_target, Fp_target, theta_target = U_trim.ravel()

z = ca.MX.sym("z")
vx = ca.MX.sym("vx")
vz = ca.MX.sym("vz")
states = ca.vertcat(z, vx, vz)
n_states = states.numel()

Fr = ca.MX.sym("Fr")
Fp = ca.MX.sym("Fp")
theta = ca.MX.sym("theta")
controls = ca.vertcat(Fr, Fp, theta)
n_controls = controls.numel()

X = ca.MX.sym("X", n_states, N + 1)
U = ca.MX.sym("U", n_controls, N)
P = ca.MX.sym("P", 2 * n_states + n_controls)

Q = 10 * ca.diagcat(1, 1, 1) # Gain matrix for X
R = 0.0 * ca.diagcat(0, 0, 1000) # Gain matrix for U

Xdot = plant.derivnox(states, controls, q=0.0)
f = ca.Function("f", [states, controls], [Xdot])

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation

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


lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

lbx[0 : n_states * (N + 1) : n_states] = z_target - 2  # z min
ubx[0 : n_states * (N + 1) : n_states] = z_target + 1  # z max
# ubx[0 : n_states * (N + 1) : n_states] = 0 # z max
lbx[1 : n_states * (N + 1) : n_states] = 0  # Vx min
ubx[1 : n_states * (N + 1) : n_states] = ca.inf  # Vx max
lbx[2 : n_states * (N + 1) : n_states] = -ca.inf  # Vz min
ubx[2 : n_states * (N + 1) : n_states] = ca.inf  # Vz max

Fr_max = 6 * plant.th_r_max  # 955.2534
Fp_max = 2 * plant.th_p_max  # 183.1982
lbx[n_states * (N + 1) :: n_controls] = -Fr_max  # Fr min
ubx[n_states * (N + 1) :: n_controls] = 0  # Fr max
lbx[n_states * (N + 1) + 1 :: n_controls] = 0  # Fp min
ubx[n_states * (N + 1) + 1 :: n_controls] = Fp_max  # Fp max
lbx[n_states * (N + 1) + 2 :: n_controls] = -theta_max  # theta min
ubx[n_states * (N + 1) + 2 :: n_controls] = theta_max  # theta min

args = {
    "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
    "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
    "lbx": lbx,
    "ubx": ubx,
}

t0 = 0
state_init = ca.DM([z_init, vx_init, vz_init])
state_target = ca.DM([z_target, vx_target, vz_target])
control_init = ca.DM([-plant.m * plant.g, 0, 0])
control_target = ca.DM([Fr_target, Fp_target, theta_target])
t = ca.DM(t0)

def solve_mpc():
    pass

def observation():
    pass

def plot():
    pass

if __name__ == "__main__":
 

