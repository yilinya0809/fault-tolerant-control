import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat, quat2angle

import ftc
from ftc.models.LinearLC62 import LinearLC62
# from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ang = np.deg2rad(np.random.uniform(-0, 0, size=(3, 1)))
    VT_ref = 45
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0, 0, -10)),
                "vel": np.vstack((40, 0, 0)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LinearLC62(env_config["plant"])

        # FW
        self.x_trims_FW, self.u_trims_fixed_FW = self.plant.get_trim_fixed(
            fixed={"h": 10, "VT": self.VT_ref}
        )
        self.u_trims_vtol_FW = 1e3 * np.ones((6, 1))
        breakpoint()

        # self.Q_FW = 10 * np.diag([100, 1, 2000, 2000, 1, 200, 100, 100, 100, 0, 0, 0])
        # self.R_FW = np.diag([1, 1, 10, 1000, 10])
        self.Q = 1e3 * np.diag([0, 0, 0, 10, 10, 10, 100, 100, 100, 0, 0, 0])
        self.R = np.diag([1, 1, 100, 100, 100])

        self.controller = ftc.make("LQR-LC62-FW", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = ctrls0

        pos, vel, quat, omega = self.plant.observe_list()
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "FM": FM,
            "ctrls": ctrls,
            "ctrls0": ctrls0,
            "Lambda": np.ones((11, 1)),
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
    fig, axes = plt.subplots(3, 4, squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlabel("Time, sec")
    # ax.legend(["Response", "Ref"], loc="upper right")
    ax.grid()

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.set_xlabel("Time, sec")
    ax.grid()

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_xlabel("Time, sec")
    ax.grid()

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_xlabel("Time, sec")
    ax.grid()

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)
    ax.grid()

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)
    ax.grid()

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.set_ylabel(r"$p$, deg/s")

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Rotor thrusts """
    fig, axs = plt.subplots(3, 2, sharex=True)
    ylabels = np.array(
        (["Rotor 1", "Rotor 2"], ["Rotor 3", "Rotor 4"], ["Rotor 5", "Rotor 6"])
    )
    for i, _ylabel in np.ndenumerate(ylabels):
        x, y = i
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2 * x + y], "k-")
        ax.set_xlim(data["t"][0], data["t"][-1])
        ax.grid()
        if i[0] == 2:
            ax.set_xlabel("Time, sec")

        plt.setp(ax, ylabel=_ylabel)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 3 - Pusher and Control surfaces """
    fig, axs = plt.subplots(5, 1, sharex=True)
    ylabels = np.array(
        ("Pusher 1", "Pusher 2", r"$\delta_a$", r"$\delta_e$", r"$\delta_r$")
    )
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, i + 6], "k-")
        ax.set_xlim(data["t"][0], data["t"][-1])
        ax.grid()
        plt.setp(ax, ylabel=_ylabel)

        if i == 4:
            ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.align_ylabels(axs)

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
