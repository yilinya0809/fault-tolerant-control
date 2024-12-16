import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np

from LC62R import LC62R
from linearctrl import LinearCtrl

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        pos0 = np.vstack((0, 0, -1))
        # vel0 = np.zeros((3, 1))
        vel0 = np.vstack((0, 0, -1))
        quat0 = np.vstack((1, 0, 0, 0))
        # omega0 = np.zeros((3, 1))
        omega0 = np.vstack((-0.7, 0.01, -0.01))
        self.plant = LC62R(pos0, vel0, quat0, omega0)
        self.controller = LinearCtrl(self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t):
        posd = np.vstack((0, 0, -10))
        return posd

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        # ctrls = ctrls0
        ctrls = self.plant.saturate(ctrls0)
        # print(ctrls0, ctrls)
        # breakpoint()

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "FM": FM,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
        }

        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data_lin.h5")

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
    data = fym.load("data_lin.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.legend(["Response", "Command"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")

    ax.set_xlabel("Time, s")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_x$, m/s")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_y$, m/s")

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, s")

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

    ax.set_xlabel("Time, s")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$p$, deg/s")
    ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, s")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    # """ Figure 2 - Generalized forces """
    # fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    # """ Column 1 - Generalized forces: Forces """
    # ax = axes[0, 0]
    # ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "r--")
    # ax.set_ylabel(r"$F_x$")
    # ax.legend(["Response", "Command"], loc="upper right")
    # ax.set_xlim(data["t"][0], data["t"][-1])

    # ax = axes[1, 0]
    # ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "r--")
    # ax.set_ylabel(r"$F_y$")

    # ax = axes[2, 0]
    # ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "r--")
    # ax.set_ylabel(r"$F_z$")

    # ax.set_xlabel("Time, s")

    # """ Column 2 - Generalized forces: Moments """
    # ax = axes[0, 1]
    # ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "r--")
    # ax.set_ylabel(r"$M_x$")
    # ax.legend(["Response", "Ref"], loc="upper right")

    # ax = axes[1, 1]
    # ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "r--")
    # ax.set_ylabel(r"$M_y$")

    # ax = axes[2, 1]
    # ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    # ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "r--")
    # ax.set_ylabel(r"$M_z$")

    # ax.set_xlabel("Time, s")

    # plt.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axes)

    """ Figure 3 - Rotor thrusts """
    fig, axes = plt.subplots(3, 2, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["ctrls"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 1")
    ax.legend(["Response", "Command"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["ctrls"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 2")

    ax = axes[2, 0]
    ax.plot(data["t"], data["ctrls"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 3")

    ax.set_xlabel("Time, s")

    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrls"][:, 3].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 3].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 4")

    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrls"][:, 4].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 4].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 5")

    ax = axes[2, 1]
    ax.plot(data["t"], data["ctrls"][:, 5].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctrls0"][:, 5].squeeze(-1), "r--")
    ax.set_ylabel("Rotor 6")

    ax.set_xlabel("Time, s")

    plt.gcf().supxlabel("Time, s")
    plt.gcf().supylabel("Rotor Thrusts")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    # """ Figure 4 - Pusher and Control surfaces """
    # fig, axes = plt.subplots(5, 1, sharex=True)

    # ax = axes[0]
    # ax.plot(data["t"], data["ctrls"][:, 6].squeeze(-1), "k-")
    # ax.plot(data["t"], data["ctrls0"][:, 6].squeeze(-1), "r--")
    # ax.set_ylabel("Pusher 1")
    # ax.legend(["Response", "Command"], loc="upper right")
    # ax.set_xlim(data["t"][0], data["t"][-1])

    # ax = axes[1]
    # ax.plot(data["t"], data["ctrls"][:, 7].squeeze(-1), "k-")
    # ax.plot(data["t"], data["ctrls0"][:, 7].squeeze(-1), "r--")
    # ax.set_ylabel("Pusher 2")

    # ax = axes[2]
    # ax.plot(data["t"], np.rad2deg(data["ctrls"])[:, 8].squeeze(-1), "k-")
    # ax.plot(data["t"], np.rad2deg(data["ctrls0"])[:, 8].squeeze(-1), "r--")
    # ax.set_ylabel(r"$\delta_a$")

    # ax = axes[3]
    # ax.plot(data["t"], np.rad2deg(data["ctrls"])[:, 9].squeeze(-1), "k-")
    # ax.plot(data["t"], np.rad2deg(data["ctrls0"])[:, 9].squeeze(-1), "r--")
    # ax.set_ylabel(r"$\delta_e$")

    # ax = axes[4]
    # ax.plot(data["t"], np.rad2deg(data["ctrls"])[:, 10].squeeze(-1), "k-")
    # ax.plot(data["t"], np.rad2deg(data["ctrls0"])[:, 10].squeeze(-1), "r--")
    # ax.set_ylabel(r"$\delta_r$")

    # ax.set_xlabel("Time, s")

    # plt.gcf().supxlabel("Time, s")
    # plt.gcf().supylabel("Pusher and Control Surfaces")

    # plt.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axes)

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
