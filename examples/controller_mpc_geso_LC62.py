import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 10,
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
        self.ang_lim = np.deg2rad(45)
        self.controller = ftc.make("geso", self)

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
        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self, action)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        dtrb, dtrb_info = self.get_dtrb(t, self)
        dtrb = np.zeros((4, 1))
        FM_dtrb = FM + np.vstack((0, 0, dtrb))
        self.plant.set_dot(t, FM_dtrb)
        self.controller.set_dot(t, self)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            **dtrb_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM_dtrb,
            "Fr": self.plant.B_VTOL(ctrls[:6], omega)[2],
            "Fp": self.plant.B_Pusher(ctrls[6:8])[0],
        }

        return env_info

    def get_dtrb(self, t, env):
        dtrb_wind = 0
        amplitude = frequency = phase_shift = []
        for i in range(5):
            a = np.random.randn(1)
            w = np.random.randint(1, 10, 1) * np.pi
            p = np.random.randint(1, 10, 1)
            dtrb_wind = dtrb_wind + a * np.sin(w * t + p)
            amplitude = np.append(amplitude, a)
            frequency = np.append(frequency, w)
            phase_shift = np.append(phase_shift, p)

        dtrb_w = dtrb_wind * np.ones((4, 1))
        pos, vel, quat, omega = env.plant.observe_list()
        del_J = 0.3 * env.plant.J
        dtrb_model = np.cross(omega, del_J @ omega, axis=0)
        dtrb_m = np.vstack((0, dtrb_model))

        dtrb = dtrb_w + dtrb_m
        dtrb_info = {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase_shift": phase_shift,
        }
        return dtrb, dtrb_info


def run():
    env = MyEnv()
    agent = ftc.make("NMPC", env)
    flogger = fym.Logger("data_geso.h5")

    env.reset()
    try:
        while True:
            env.render()

            action, agent_info = agent.get_action()

            obs, done, env_info = env.step(action=action)
            agent.solve_mpc(obs)

            flogger.record(env=env_info, agent=agent_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data_geso.h5")["env"]
    agent_data = fym.load("data_geso.h5")["agent"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "b-")
    ax.set_ylabel(r"$y$, m")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    a1 = ax.plot(data["t"], agent_data["Xd"][:, 0].squeeze(-1), "r--")
    a2 = ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-15, -5])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data["t"], agent_data["Xd"][:, 1].squeeze(-1), "--r")
    ax.set_ylabel(r"$v_x$, m/s")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "b-")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(data["t"], agent_data["Xd"][:, 2].squeeze(-1), "--r")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_ylim([-20, 20])

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

    # fig.tight_layout()
    fig.legend(
        [a1, a2],
        labels=["Commands from NMPC", "States"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=12,
        ncol=2,
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Generalized forces """
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Generalized forces: Forces """
    ax = axes[0, 0]
    ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_x$")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_y$")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Generalized forces: Moments """
    ax = axes[0, 1]
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "b-")
    ax.set_ylabel(r"$M_x$")
    ax.set_ylim([-1, 1])

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "b-")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "b-")
    ax.set_ylabel(r"$M_z$")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Rotor inputs """
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

    """ Figure 4 - Pusher input """
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
    ax.plot(data["t"], -data["Frd"], "r--")
    ax.plot(data["t"], -data["Fr"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{rotors}$, N")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    l1 = ax.plot(data["t"], data["Fpd"], "r--")
    l2 = ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{pushers}$, N")
    ax.set_xlabel("Time, sec")

    fig.legend(
        [l1, l2],
        labels=["Commands from NMPC", "Results"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=13,
        ncol=2,
    )

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
