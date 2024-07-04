import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle
from poly_corr import boundary, poly, weighted_poly

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")

Trst_corr = np.load("corr_conti.npz")
# Trst_corr = np.load("corr_back.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc_corr"]
theta_corr = Trst_corr["theta_corr"]


class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 10,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -50.0)),
                "vel": np.zeros((3, 1)),
                # "vel": np.vstack((40, 0, 0)),
                "quat": np.vstack((1, 0, 0, 0)),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62R(env_config["plant"])
        self.ang_lim = np.deg2rad(45)
        self.controller = ftc.make("NMPC-GESO", self)
        self.agent = ftc.make("NMPC-Corr", self)
        # self.controller = ftc.make("NMPC-DI-test", self)

    def step(self, action):
        env_info, done = self.update(action=action)
        obs = self.observation()

        return obs, done, env_info

    def observation(self):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        obs = (pos[2], vel[0], vel[2], ang[1], omega[1])  # Current state
        return obs

    def set_dot(self, t, action):
        tf = self.clock.max_t
        pos, vel, quat, omega = self.plant.observe_list()
        stated = self.agent.set_ref(t, tf, VT_corr[0], VT_corr[-1])
        # stated = self.agent.set_ref(t, tf, VT_corr[-1], VT_corr[0])
        ctrls0, controller_info = self.controller.get_control(t, self, action)
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
            "Fr": self.plant.B_VTOL(ctrls[:6], omega)[2],
            "Fp": self.plant.B_Pusher(ctrls[6:8])[0],
            "stated": stated,
        }

        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data_corr.h5")
    # flogger = fym.Logger("data_corr_back.h5")

    env.reset()
    try:
        while True:
            env.render()

            action, agent_info = env.agent.get_action()
            obs, done, env_info = env.step(action=action)
            env.agent.solve_mpc(obs)
            flogger.record(env=env_info, agent=agent_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data_corr_archive.h5")["env"]

    # data = fym.load("data_corr_back.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim([0, 500])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.legend(["Response", "Command"], loc="upper right")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["stated"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-60, -40])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["stated"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_ylim([0, 50])

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    # ax.plot(data["t"], data["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["stated"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_ylim([-3, 3])

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
    # ax.set_ylim([-1, 1])

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
    fig.subplots_adjust(left=0.05, right=0.99, wspace=0.3)
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

    """ Figure 3 - Thrusts """
    fig, axs = plt.subplots(2, 4, sharex=True)

    ax = axs[0, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "k-")
    ax.set_ylabel("Rotor 1")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "k-")
    ax.set_ylabel("Rotor 2")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "k-")
    ax.set_ylabel("Rotor 3")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "k-")
    ax.set_ylabel("Rotor 4")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 2]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "k-")
    ax.set_ylabel("Rotor 5")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 2]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "k-")
    ax.set_ylabel("Rotor 6")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 3]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "k-")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 3]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "k-")
    ax.set_ylabel("Pusher 2")
    ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.5, 1.5])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    fig.suptitle("Rotational Thrusts", y=0.85)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.8, wspace=0.25, hspace=0.2)
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
