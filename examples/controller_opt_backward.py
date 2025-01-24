import argparse

import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat, quat2angle
from scipy.integrate import cumtrapz

import ftc
from ftc.models.LC62R_lin import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


""" Results of Outer-loop optimal trajectory """
opt_traj = {}
f = h5py.File("data/opt_backward.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)

Vxd = opt_traj["X"][1, :]
Vzd = opt_traj["X"][2, :]
thetad = opt_traj["U"][2, :]
xdot = []
for i in range(N):
    xdot.append(Vxd[i + 1] * np.cos(thetad[i]) + Vzd[i + 1] * np.sin(thetad[i]))
Xd = cumtrapz(xdot, tspan[1:], initial=0)


class MyEnv(fym.BaseEnv):
    VT_cruise = 45
    h = 10
    ang0 = thetad[0]
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -h)),
                "vel": np.vstack((VT_cruise * np.cos(ang0), 0, VT_cruise * np.sin(ang0))),
                "quat": angle2quat(0, ang0, 0),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62R(env_config["plant"])
        self.ang_lim = np.deg2rad(30)

        # FW
        self.x_trims_FW, self.u_trims_fixed_FW = self.plant.get_trim_fixed(
            fixed={"h": self.h, "VT": self.VT_cruise}
        )
        self.u_trims_vtol_FW = np.zeros((6, 1))
        self.Q_FW = np.diag([0, 0, 200, 10, 10, 20, 100, 200, 100, 0, 0, 0])
        self.R_FW = np.diag([400, 400, 1, 1, 1])

        # HV
        self.x_trims_HV, self.u_trims_fixed_HV = self.plant.get_trim_fixed(
            fixed={"h": self.h, "VT": 0}
        )
        self.u_trims_vtol_HV = self.plant.get_trim_vtol(
            fixed={"x_trims": self.x_trims_HV, "u_trims_fixed": self.u_trims_fixed_HV}
        )
        self.Q_HV = np.diag([0, 0, 20, 10, 10, 20, 10, 50000, 10, 0, 0, 0])
        self.R_HV = 10000 * np.diag([1, 1, 1, 1, 1, 1])

        self.controller_trst = ftc.make("Trst-Corr", self)
        self.controller_lqr = ftc.make("FWHV", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t):
        zd = -self.h
        if t <= tspan[-1]:
            xd = np.interp(t, tspan[1:], Xd[:])
            Vxd = np.interp(t, tspan, opt_traj["X"][1, :])
            Vzd = np.interp(t, tspan, opt_traj["X"][2, :])
            veld = np.vstack((Vxd, 0, Vzd))
            thetad = np.interp(t, tspan[1:], opt_traj["U"][2, :])
            mode = "BTC"
        else:
            xd = Xd[-1]
            veld = np.zeros((3, 1))
            thetad = 0
            mode = "HV"

        return xd, zd, veld, thetad, mode

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        _, _, _, _, mode = self.get_ref(t)

        if mode == "BTC":
            ctrls0, controller_info = self.controller_trst.get_control(t, self)
        elif mode == "HV":
            ctrls0, controller_info = self.controller_lqr.get_control(t, self)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
            "Fr": -self.plant.B_VTOL(ctrls[:6], omega)[2],
            "Fp": self.plant.B_Pusher(ctrls[6:8])[0],
        }

        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data_opt_backward.h5")

    env.reset()
    try:
        while True:
            env.render()

            done, env_info = env.step()
            flogger.record(env=env_info)

            if done:
                break
    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data_opt_backward.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(2, 3, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.set_ylabel(r"$x$, m", fontsize=14)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(tspan, opt_traj["X"][1, :], "r--")
    ax.set_ylabel(r"$v_x$, m/s", fontsize=14)

    ax.set_xlabel("Time, sec", fontsize=14)

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(tspan, opt_traj["X"][0, :], "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$z$, m", fontsize=14)
    ax.set_ylim([-15, -5])

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(tspan, opt_traj["X"][2, :], "r--")
    ax.set_ylabel(r"$v_z$, m/s", fontsize=14)
    ax.set_ylim([-10, 10])

    ax.set_xlabel("Time, sec", fontsize=14)

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg", fontsize=14)

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s", fontsize=14)

    ax.set_xlabel("Time, sec", fontsize=14)
    fig.tight_layout()

    """ Figure 2 - Rotor inputs """
    fig, axes = plt.subplots(2, 4, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b-")
    ax.set_ylabel("Rotor 1", fontsize=14)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim(-0.1, 1.1)

    ax = axes[1, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b-")
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b-")
    ax.set_ylabel("Rotor 3", fontsize=14)
    ax.set_ylim(-0.1, 1.1)

    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b-")
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[0, 2]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b-")
    ax.set_ylabel("Rotor 5", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 2]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b-")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 3]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    ax.set_ylabel("Pusher 1", fontsize=14)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim(-0.1, 1.1)

    ax = axes[1, 3]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(data["t"][0], data["t"][-1])
    fig.tight_layout()

    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 5 - Thrust """
    fig, axes = plt.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], data["Frd"], "r--")
    ax.plot(data["t"], data["Fr"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{rotors}$, N")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    ax.plot(data["t"], data["Fpd"], "r--")
    ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{pushers}$, N")
    ax.set_xlabel("Time, sec")

    plt.tight_layout()
    fig.align_ylabels(axes)

    plt.show()


def main(args):
    if args.only_plot:
        plot()
        return
    else:
        run()

        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)
