import argparse

import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle, angle2quat

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


""" Results of Outer-loop optimal trajectory """
opt_traj = {}
f = h5py.File("data/back_traj.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)


class MyEnv(fym.BaseEnv):
    theta0 = opt_traj["U"][2, 0]
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": tspan[-1],
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, opt_traj["X"][0, 0])),
                "vel": np.vstack((opt_traj["X"][1, 0], 0, opt_traj["X"][2, 0])),
                "quat": np.vstack(angle2quat(0, theta0, 0)),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62R(env_config["plant"])
        self.ang_lim = np.deg2rad(30)
        self.controller = ftc.make("Trst", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t):
        zd = np.interp(t, tspan, opt_traj["X"][0, :])
        Vxd = np.interp(t, tspan, opt_traj["X"][1, :])
        Vzd = np.interp(t, tspan, opt_traj["X"][2, :])
        veld = np.vstack((Vxd, 0, Vzd))
        thetad = np.interp(t, tspan[1:], opt_traj["U"][2, :])
        return zd, veld, thetad

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self)
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
    flogger = fym.Logger("data.h5")

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
    data = fym.load("data.h5")["env"]

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
