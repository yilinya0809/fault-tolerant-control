import argparse

import casadi as ca
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import ftc
from ftc.models.LC62_mpc import LC62
from ftc.trst_corr.poly_corr import boundary


class TrajectoryOptimizer:
    def __init__(self, max_t=10, dt=0.1, h_ref=10, VT_ref=45):
        self.max_t = max_t
        self.dt = dt
        self.N = int(max_t / dt)
        self.t = np.linspace(0, max_t, self.N + 1)

        self.h_ref = h_ref
        self.VT_ref = VT_ref

        self.plant = LC62()

        self.state_init = ca.DM([-self.h_ref, 0, 0, 0])
        X_trim, U_trim = self.plant.get_trim(fixed={"h": self.h_ref, "VT": self.VT_ref})
        self.X_target = X_trim[1:]
        self.U_target = U_trim

        # CasADi optimization variables
        self.states, self.controls, self.X, self.U, self.P = self.define_variables()

        Xdot = self.plant.deriv(self.states, self.controls)
        self.f = ca.Function("f", [self.states, self.controls], [Xdot])
        self.Q = 1000 * ca.diagcat(1, 1, 1, 1, 1, 1, 1)
        self.R = ca.diagcat(0, 0, 0)

        # Boundary conditions
        Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
        self.min_theta, self.upper, self.lower = self.corridor_boundaries(Trst_corr)

        # Prepare the solver
        self.solver, self.args = self.setup_solver()

    def define_variables(self):
        z = ca.MX.sym("z")
        Vx = ca.MX.sym("Vx")
        Vz = ca.MX.sym("Vz")
        theta = ca.MX.sym("theta")
        states = ca.vertcat(z, Vx, Vz, theta)
        n_states = states.numel()

        Fr = ca.MX.sym("Fr")
        Fp = ca.MX.sym("Fp")
        q = ca.MX.sym("q")
        controls = ca.vertcat(Fr, Fp, q)
        n_controls = controls.numel()

        X = ca.MX.sym("X", n_states, self.N + 1)
        U = ca.MX.sym("U", n_controls, self.N)
        P = ca.MX.sym("P", n_states + 2)

        return states, controls, X, U, P

    def corridor_boundaries(self, Trst_corr):
        VT_corr = Trst_corr["VT_corr"]
        upper_bound, lower_bound = boundary(Trst_corr)

        # Fit the lower boundary
        mask = lower_bound > np.min(lower_bound)
        VT_filtered = VT_corr[mask]
        lower_bound_filtered = lower_bound[mask]

        popt, _ = curve_fit(
            lambda x, a, b, c: -a * np.exp(-b * x) + c,
            VT_filtered,
            lower_bound_filtered,
        )
        lower = np.vstack((VT_filtered[0], popt[0], popt[1], popt[2]))

        # Fit the upper boundary
        deg = 3
        upper = np.polyfit(VT_corr, upper_bound, deg)

        return np.min(lower_bound), lower, upper

    def lower_func(self, VT):
        offset, a, b, c = self.lower
        return ca.if_else(
            VT <= offset, self.min_theta, -a * ca.exp(-b * (VT - offset)) + c
        )

    def upper_func(self, VT):
        value = 0
        for i in range(len(self.upper)):
            coeff = self.upper[i]
            value += coeff * VT ** (len(self.upper) - i - 1)
        return value

    def setup_solver(self):
        cost_fn = 0
        g_rk4 = self.X[:, 0] - self.P[: self.states.numel()]  # equality constraints
        g_cons = ca.vertcat(0, 0)  # inequality constraints

        # Terminal cost
        phi = ca.vertcat(
            self.X[:, self.N] - self.X_target, self.U[:, self.N - 1] - self.U_target
        )
        cost_fn = phi.T @ self.Q @ phi

        # Runge-Kutta integration and cost accumulation
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            cost_fn = cost_fn + con.T @ self.R @ con

            # dynamic equation constraint
            st_next = self.X[:, k + 1]
            k1 = self.f(st, con)
            k2 = self.f(st + self.dt / 2 * k1, con)
            k3 = self.f(st + self.dt / 2 * k2, con)
            k4 = self.f(st + self.dt * k3, con)
            st_next_RK4 = st + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g_rk4 = ca.vertcat(g_rk4, st_next - st_next_RK4)

            # corridor boundary constraint
            tht = st[-1]
            VT = ca.norm_2(st[1:3])
            g1 = self.lower_func(VT) - tht
            g2 = tht - self.upper_func(VT)
            g_cons = ca.vertcat(g_cons, g1, g2)

        opt_cons = ca.vertcat(g_rk4, g_cons)
        opt_variables = ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1)))
        opt_params = ca.vertcat(self.P.reshape((-1, 1)))
        nlp_prob = {"f": cost_fn, "x": opt_variables, "g": opt_cons, "p": opt_params}

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

        lbx = ca.DM.zeros(
            (self.states.numel() * (self.N + 1) + self.controls.numel() * self.N, 1)
        )
        ubx = ca.DM.zeros(
            (self.states.numel() * (self.N + 1) + self.controls.numel() * self.N, 1)
        )
        lbg = ca.DM.zeros(((self.states.numel() + 2) * (self.N + 1), 1))
        ubg = ca.DM.zeros(((self.states.numel() + 2) * (self.N + 1), 1))

        z_eps = 2
        lbx[0 : self.states.numel() * (self.N + 1) : self.states.numel()] = -ca.inf
        ubx[0 : self.states.numel() * (self.N + 1) : self.states.numel()] = ca.inf
        lbx[1 : self.states.numel() * (self.N + 1) : self.states.numel()] = 0
        ubx[1 : self.states.numel() * (self.N + 1) : self.states.numel()] = ca.inf
        lbx[2 : self.states.numel() * (self.N + 1) : self.states.numel()] = -ca.inf
        ubx[2 : self.states.numel() * (self.N + 1) : self.states.numel()] = ca.inf
        lbx[3 : self.states.numel() * (self.N + 1) : self.states.numel()] = -np.deg2rad(
            30
        )
        ubx[3 : self.states.numel() * (self.N + 1) : self.states.numel()] = np.deg2rad(
            30
        )

        Fr_max = 6 * self.plant.th_r_max
        Fp_max = 2 * self.plant.th_p_max
        lbx[self.states.numel() * (self.N + 1) :: self.controls.numel()] = -Fr_max
        ubx[self.states.numel() * (self.N + 1) :: self.controls.numel()] = 0
        lbx[self.states.numel() * (self.N + 1) + 1 :: self.controls.numel()] = 0
        ubx[self.states.numel() * (self.N + 1) + 1 :: self.controls.numel()] = Fp_max
        lbx[self.states.numel() * (self.N + 1) + 2 :: self.controls.numel()] = -ca.inf
        ubx[self.states.numel() * (self.N + 1) + 2 :: self.controls.numel()] = ca.inf

        lbg[: self.states.numel() * (self.N + 1)] = 0
        ubg[: self.states.numel() * (self.N + 1)] = 0
        lbg[self.states.numel() * (self.N + 1) :] = -ca.inf
        ubg[self.states.numel() * (self.N + 1) :] = 0

        args = {
            "lbg": lbg,
            "ubg": ubg,
            "lbx": lbx,
            "ubx": ubx,
        }

        return solver, args

    def run(self):
        control_init = ca.DM([-self.plant.m * self.plant.g, 0, 0])
        u0 = ca.repmat(control_init, 1, self.N)
        X0 = ca.repmat(self.state_init, 1, self.N + 1)

        self.args["p"] = ca.vertcat(self.state_init, ca.DM(0), ca.DM(0))
        self.args["x0"] = ca.vertcat(
            ca.reshape(X0, self.states.numel() * (self.N + 1), 1),
            ca.reshape(u0, self.controls.numel() * self.N, 1),
        )

        sol = self.solver(
            x0=self.args["x0"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            p=self.args["p"],
        )

        self.X_sol = ca.reshape(
            sol["x"][: self.states.numel() * (self.N + 1)],
            self.states.numel(),
            self.N + 1,
        )
        self.U_sol = ca.reshape(
            sol["x"][self.states.numel() * (self.N + 1) :],
            self.controls.numel(),
            self.N,
        )

        self.save_results()
        plot()

    def save_results(self):
        with h5py.File("trajectory_data.h5", "w") as hf:
            hf.create_dataset("cat_states", data=DM2Arr(self.X_sol))
            hf.create_dataset("cat_controls", data=DM2Arr(self.U_sol))
            hf.create_dataset("time", data=self.t)


def DM2Arr(dm):
    return np.array(dm.full())


def plot():
    with h5py.File("trajectory_data.h5", "r") as hf:
        cat_states = hf["cat_states"][:]
        cat_controls = hf["cat_controls"][:]
        t = hf["time"][:]

    """Fig 1. States"""
    fig, axes = plt.subplots(2, 2, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, cat_states[0, :], "k-")
    ax.set_ylabel(r"$z$ [m]")
    ax.set_xlabel("Time [s]")
    ax.set_ylim([-20, 0])

    ax = axes[0, 1]
    ax.plot(t, cat_states[1, :], "k-")
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.set_xlabel("Time [s]")

    ax = axes[1, 0]
    ax.plot(t, cat_states[2, :], "k-")
    ax.set_ylabel(r"$v_z$ [m/s]")
    ax.set_xlabel("Time [s]")

    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(cat_states[3, :]), "k-")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_xlabel("Time [s]")

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Fig 2. Controls """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t[:-1], -cat_controls[0, :], "k-")
    ax.set_ylabel(r"$F_r$ [N]")
    ax.set_xlabel("Time [s]")

    ax = axes[1, 0]
    ax.plot(t[:-1], cat_controls[1, :], "k-")
    ax.set_ylabel(r"$F_p$ [N]")
    ax.set_xlabel("Time [s]")

    ax = axes[2, 0]
    ax.plot(t[:-1], np.rad2deg(cat_controls[2, :]), "k-")
    ax.set_ylabel(r"$q$ [deg/s]")
    ax.set_xlabel("Time [s]")

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Fig 3. Transition Corridor """
    Trst_corr = np.load("ftc/trst_corr/corr_safe.npz")
    VT_corr = Trst_corr["VT_corr"]
    acc_corr = Trst_corr["acc"]
    theta_corr = np.rad2deg(Trst_corr["theta_corr"])
    cost = Trst_corr["cost"]
    success = Trst_corr["success"]
    Fr = Trst_corr["Fr"]
    Fp = Trst_corr["Fp"]

    upper_bound, lower_bound = boundary(Trst_corr)
    VT, theta = np.meshgrid(VT_corr, theta_corr)

    VT_traj = np.zeros((np.size(t), 1))
    theta_traj = np.zeros((np.size(t), 1))
    for i in range(np.size(t)):
        VT_traj[i] = np.linalg.norm(cat_states[1:3, i])
        theta_traj[i] = np.rad2deg(cat_states[3, i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # contour = ax.contourf(
    #     VT, theta, acc_corr.T, levels=np.shape(theta_corr)[0], cmap="viridis", alpha=1.0
    # )
    ax.plot(
        VT_corr,
        np.rad2deg(upper_bound),
        "o",
        label="Upper Bound Data",
        color="blue",
        alpha=0.3,
    )
    ax.plot(
        VT_corr,
        np.rad2deg(lower_bound),
        "o",
        label="Lower Bound Data",
        color="orange",
        alpha=0.3,
    )

    ax.plot(VT_traj, theta_traj, "k-")
    ax.set_xlabel("VT, m/s", fontsize=15)
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)
    ax.set_title("Forward Acceleration Corridor", fontsize=20)
    # cbar = fig.colorbar(contour)
    # cbar.ax.set_xlabel(r"$a_x, m/s^{2}$", fontsize=15)

    plt.show()


def main(args):
    if args.only_plot:
        plot()
        return
    else:
        optimizer = TrajectoryOptimizer()
        optimizer.run()
        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)
