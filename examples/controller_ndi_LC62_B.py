import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle
from numpy import cos, sin

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")

class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -10.0)),
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
        self.controller = ftc.make("NDI-B", self)
        self.cruise_speed = 45

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        alp = ang[1]

        """ VTOL + Hovering + Level flight """ 
        # if 0 <= t <= 20::
        #     # VTOL + Hovering
        #     posd = np.vstack((0, 0, -10))
        #     posd_dot = np.vstack((0, 0, 0))
        # elif 20 < t <= 60:
        #     # Level Flight
        #     VT = self.cruise_speed
        #     # posd_dot = np.vstack((VT*cos(alp), 0, VT*sin(alp)))
        #     posd_dot = np.vstack((VT, 0, 0))
        #     posd = np.vstack((0, 0, -10)) + (t - 20) * posd_dot

        """ Level Flight only """
        VT = self.cruise_speed
        posd = np.vstack((t*VT*cos(alp), 0, -10))
        posd_dot = np.vstack((VT*cos(alp), 0, VT*sin(alp)))

        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
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
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim([0, 2000])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.set_ylim([-1, 1])
    ax.legend(["Response", "Command"], loc="upper right")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-12, -9])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_ylim([0, 50])

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(left = 0.05, right = 0.99, wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Generalized forces """
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Generalized forces: Forces """
    ax = axes[0, 0]
    ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_x$")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-1, 1]

    ax = axes[1, 0]
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_y$")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Generalized forces: Moments """
    ax = axes[0, 1]
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_x$")
    ax.set_ylim([-1, 1])

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_y$")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_z$")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Rotor thrusts """
    fig, axs = plt.subplots(3, 2, sharex=True)
    ylabels = np.array(
        (["Rotor 1", "Rotor 2"], ["Rotor 3", "Rotor 4"], ["Rotor 5", "Rotor 6"])
    )
    for i, _ylabel in np.ndenumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, sum(i)], "k-") 
        ax.grid()
        ax.set_ylim([0, 1])
        ax.set_xlim(data["t"][0], data["t"][-1])
        plt.setp(ax, ylabel=_ylabel)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor Thrusts")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 4 - Pusher and Control surfaces """
    fig, axs = plt.subplots(5, 1, sharex=True)
    ylabels = np.array(
        ("Pusher 1", "Pusher 2", r"$\delta_a$", r"$\delta_e$", r"$\delta_r$")
    )
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, i + 6], "k-")
        ax.grid()
        ax.set_ylim([-1, 1])
        ax.set_xlim(data["t"][0], data["t"][-1])
        plt.setp(ax, ylabel=_ylabel)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Pusher and Control Surfaces")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
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
