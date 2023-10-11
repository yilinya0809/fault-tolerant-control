from time import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from casadi import cos, pi, sin

from dyn_nox import LC62

plant = LC62()

step_horizon = 0.1  # time between steps in seconds
N = 10              # number of look ahead steps
sim_time = 20       # simulation time

x_init = 0
z_init = -10
Vx_init = 0
Vz_init = 0

X_trim, U_trim = plant.get_trim(fixed={"h": 10, "VT": 45})
z_target, Vx_target, Vz_target = np.ravel(X_trim)
Fr_target, Fp_target, theta_target = np.ravel(U_trim)

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


def plot(cat_states, cat_controls, t, step_horizon, N, save=False):
    """ Fig 1. States """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)
    
    ax = axes[0, 0]
    ax.plot(t, cat_states[0, 0, :])
    ax.set_ylabel("z [m]")
    ax.set_xlabel("Time, sec")

    ax = axes[1, 0]
    ax.plot(t, cat_states[1, 0, :])
    ax.set_ylabel("Vx [m/s]")
    ax.set_xlabel("Time, sec")


    ax = axes[2, 0]
    ax.plot(t, cat_states[2, 0, :])
    ax.set_ylabel("Vz [m/s]")
    ax.set_xlabel("Time, sec")


    """ Fig 2. Controls """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)
    
    ax = axes[0, 0]
    ax.plot(t, cat_controls[0, 0, :])
    ax.set_ylabel("Fr [N]")
    ax.set_xlabel("Time, sec")



    ax = axes[1, 0]
    ax.plot(t, cat_controls[1, 0, :])
    ax.set_ylabel("Fp [N]")
    ax.set_xlabel("Time, sec")


    ax = axes[2, 0]
    ax.plot(t, np.rad2deg(cat_controls[2, 0, :]))
    ax.set_ylabel(r"$\theta [deg]$")
    ax.set_xlabel("Time, sec")


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    plt.show()



# state symbolic variables
# x = ca.MX.sym('x')
z = ca.MX.sym('z')
Vx = ca.MX.sym('Vx')
Vz = ca.MX.sym('Vz')

states = ca.vertcat(
    z,
    Vx,
    Vz,
)
n_states = states.numel()

# control symbolic variables
Fr = ca.MX.sym('Fr')
Fp = ca.MX.sym('Fp')
theta = ca.MX.sym('theta')
controls = ca.vertcat(
    Fr,
    Fp,
    theta,
)
n_controls = controls.numel()

state_init = ca.DM([z_init, Vx_init, Vz_init])        
state_target = ca.DM([z_target, Vx_target, Vz_target]) 
control_init = ca.DM([plant.m * plant.g, 0, 0])
control_target = ca.DM([Fr_target, Fp_target, theta_target])

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.MX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.MX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.MX.sym('P', 2 * n_states + n_controls)

# state weights matrix (Q_x, Q_z, Q_Vx, Q_Vz)

Q = ca.diagcat(100, 100, 100)

# controls weights matrix (R_Fr, R_Fp, R_theta)
R = ca.diagcat(0.01, 0.1, 100)


Xdot = plant.deriv(states, controls)
f = ca.Function('f', [states, controls], [Xdot])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    st_err = st - P[n_states:2*n_states]
    con_err = con - P[2*n_states:]
    cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err
        # + con_err.T @ R @ con
        # + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

z_eps = 2
lbx[0: n_states*(N+1): n_states] = z_target - z_eps     # z min
ubx[0: n_states*(N+1): n_states] = z_target + z_eps     # z max
lbx[1: n_states*(N+1): n_states] = 0                    # Vx min
ubx[1: n_states*(N+1): n_states] = ca.inf               # Vx max
lbx[2: n_states*(N+1): n_states] = -ca.inf              # Vz min
ubx[2: n_states*(N+1): n_states] = ca.inf               # Vz max


Fr_max = 6 * plant.th_r_max # 955.2534
Fp_max = 2 * plant.th_p_max # 183.1982
lbx[n_states*(N+1)::n_controls] = 0                     # Fr min
ubx[n_states*(N+1)::n_controls] = Fr_max                # Fr max
lbx[n_states*(N+1)+1::n_controls] = 0                   # Fp min
ubx[n_states*(N+1)+1::n_controls] = Fp_max              # Fp max
lbx[n_states*(N+1)+2::n_controls] = -np.deg2rad(50)     # theta min
ubx[n_states*(N+1)+2::n_controls] = np.deg2rad(50)      # theta min

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
t = ca.DM(t0)

# u0 = ca.DM.zeros((n_controls, N))  # initial control
u0 = ca.repmat(control_init, 1, N)
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])



if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,     # current state
            state_target,   # target state
            control_target  # target control
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.dstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        t2 = time()
        # print(mpc_iter)
        # print(t2-t1)


        times = np.vstack((
            times,
            t0
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    st_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final state error: ', st_error)

    plot(cat_states, cat_controls, times, step_horizon, N, save=False)
