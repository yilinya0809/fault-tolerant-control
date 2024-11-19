import argparse

import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle

import ftc
from ftc.models.LC62R_lin import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


""" Optimal transition trajectory """
opt_traj = {}
f = h5py.File("data/opt_corr.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)


class MyEnv(fym.BaseEnv):
    VT_cruise = 45
    h = 10
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -h)),
                "vel": np.zeros((3, 1)),
                "quat": np.vstack((1, 0, 0, 0)),
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
        self.Q = np.diag([0, 0, 100, 10, 10, 10, 100, 100, 100, 0, 0, 0])
        # self.R = np.diag([1, 1, 100, 100, 100])
        self.R = np.diag([400, 400, 1, 1, 1])

        self.controller_trst = ftc.make("Trst", self)
        self.controller_fw = ftc.make("FW", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    # Optimal Transition Trajectory Reference
    def get_ref(self, t):
        zd = np.interp(t, tspan, opt_traj["X"][0, :])
        Vxd = np.interp(t, tspan, opt_traj["X"][1, :])
        Vzd = np.interp(t, tspan, opt_traj["X"][2, :])
        veld = np.vstack((Vxd, 0, Vzd))
        thetad = np.interp(t, tspan[1:], opt_traj["U"][2, :])
        return zd, veld, thetad

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        VT = np.linalg.norm(vel)
        if VT < self.VT_cruise - 2:
            # if t < opt_traj["tf"]:
            ctrls0, controller_info = self.controller_trst.get_control(t, self)
        else:
            ctrls0, controller_info = self.controller_fw.get_control(t, self)

        ctrls = self.plant.saturate(ctrls0)
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "FM": FM,
            "ctrls": ctrls,
            "ctrls0": ctrls0,
            "Fr": -self.plant.B_VTOL(ctrls[:6], omega)[2],
            "Fp": self.plant.B_Pusher(ctrls[6:8])[0],
        }

        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data_opt_ndi.h5")

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
    data = fym.load("data/data_switch.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["posd"][:, 1], "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "b-")
    ax.set_ylabel(r"$y$, m")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    # ax.plot(tspan, opt_traj["X"][0, :], "r--")
    ax.plot(data["t"], data["posd"][:, 2], "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-15, -5])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data["t"], data["veld"][:, 0], "r--")
    ax.set_ylabel(r"$v_x$, m/s")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "b-")
    ax.plot(data["t"], data["veld"][:, 1], "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(data["t"], data["veld"][:, 2], "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_ylim([-10, 10])

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()

    """ Figure 2 - Rotor inputs """
    fig, axes = plt.subplots(3, 2, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b-")
    ax.set_ylabel("Rotor 1")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b-")
    ax.set_ylabel("Rotor 2")

    ax = axes[2, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b-")
    ax.set_ylabel("Rotor 3")
    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b-")
    ax.set_ylabel("Rotor 4")

    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b-")
    ax.set_ylabel("Rotor 5")

    ax = axes[2, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b-")
    ax.set_ylabel("Rotor 6")
    ax.set_xlabel("Time, sec")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 3 - Pusher input """
    fig, axes = plt.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    ax.set_ylabel("Pusher 2")
    ax.set_xlabel("Time, sec")

    plt.tight_layout()
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
