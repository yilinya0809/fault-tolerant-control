"""
LC62-50B LQR control
considering Flight Mode Envelope
w.r.t. current velocity 

VTOL mode: use only rotors
FW   mode: use all actuators
"""

import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat, quat2angle
from matplotlib import animation

import ftc
from ftc.models.LC62R import LC62R
from ftc.plotframe import LC62Frame, update_plot
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ang = np.random.uniform(-np.deg2rad(0), np.deg2rad(0), size=(3, 1))
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0, 0, -10)),
                "vel": np.vstack((0, 0, 0)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62R(env_config["plant"])
        cruise_speed = 40

        # Hovering
        self.x_trims_HV, self.u_trims_fixed_HV = self.plant.get_trim_fixed(
            fixed={"h": 10, "VT": 0}
        )
        self.u_trims_vtol_HV = self.plant.get_trim_vtol(
            fixed={"x_trims": self.x_trims_HV, "u_trims_fixed": self.u_trims_fixed_HV}
        )

        self.Q_HV = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        self.R_HV = 1000 * np.diag([1, 1, 1, 1, 1, 1])

        # # FW
        # self.x_trims_FW, self.u_trims_fixed_FW = self.plant.get_trim_fixed(
        #     fixed={"h": 10, "VT": self.cruise_speed}
        # )
        # self.u_trims_vtol_FW = self.plant.get_trim_vtol(
        #     fixed={"x_trims": self.x_trims_FW, "u_trims_fixed": self.u_trims_fixed_FW}
        # )
        # self.Q_FW = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        # self.R_FW = 1000 * np.diag([1, 1, 1, 1, 1])

        # FW Flight envelope
        self.VT_max = cruise_speed
        self.VT = 5
        len = int(self.VT_max / self.VT)

        pos_trim = np.zeros((3, len))
        vel_trim = np.zeros((3, len))
        quat_trim = np.zeros((4, len))
        omega_trim = np.zeros((3, len))
        ang_trim = np.zeros((3, len))
        rcmds_trim = np.zeros((6, len))
        pcmds_trim = np.zeros((2, len))
        dels_trim = np.zeros((3, len))
        self.x_trims_FW = np.zeros((12, len))
        self.u_trims_FW = np.zeros((11, len))

        for i in range(len):
            x_trims_FW, u_trims_fixed_FW = self.plant.get_trim_fixed(
                fixed={"h": 10, "VT": self.VT * (i + 1)}
            )

            (
                pos_trim[:, i : i + 1],
                vel_trim[:, i : i + 1],
                quat_trim[:, i : i + 1],
                omega_trim[:, i : i + 1],
            ) = x_trims_FW

            ang_trim[:, i : i + 1] = np.vstack(
                quat2angle(quat_trim[:, i : i + 1])[::-1]
            )

            (pcmds_trim[:, i : i + 1], dels_trim[:, i : i + 1]) = u_trims_fixed_FW

            rcmds_trim[:, i : i + 1] = self.plant.get_trim_vtol(
                fixed={"x_trims": x_trims_FW, "u_trims_fixed": u_trims_fixed_FW}
            )

            self.x_trims_FW[:, i : i + 1] = np.vstack(
                (
                    pos_trim[:, i : i + 1],
                    vel_trim[:, i : i + 1],
                    ang_trim[:, i : i + 1],
                    omega_trim[:, i : i + 1],
                )
            )
            self.u_trims_FW[:, i : i + 1] = np.vstack(
                (
                    rcmds_trim[:, i : i + 1],
                    pcmds_trim[:, i : i + 1],
                    dels_trim[:, i : i + 1],
                )
            )

        self.Q_FW = np.zeros((12, 12, len))
        self.R_FW = np.zeros((5, 5, len))
        for i in range(len):
            self.Q_FW[:, :, i] = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
            self.R_FW[:, :, i] = 1000 * np.diag([1, 1, 1, 1, 1])

        self.controller = ftc.make("LQR-LC62-modeEnv", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        # if 0 <= t <= 20:
        #     # VTOL + Hovering
        #     posd = np.vstack((0, 0, -10))
        #     posd_dot = np.vstack((0, 0, 0))
        #     mode = "VTOL"
        # elif 20 < t <= 40:
        #     # Level Flight
        #     VT = self.cruise_speed
        #     posd_dot = np.vstack((VT, 0, 0))
        #     posd = np.vstack((0, 0, -10)) + (t - 20) * posd_dot
        #     mode = "FW"

        # Level Flight
        VT = self.VT_max
        posd_dot = np.vstack((VT, 0, 0))
        posd = np.vstack((0, 0, -10)) + t * posd_dot
        mode = "FW"

        refs = {"posd": posd, "posd_dot": posd_dot, "mode": mode}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls, controller_info = self.controller.get_control(t, self)

        pos, vel, quat, omega = self.plant.observe_list()
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "FM": FM,
            "ctrls": ctrls,
            # "Lambda": np.ones((11, 1)),
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
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    x = ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    x_ref = ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlabel("Time, sec")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.set_xlabel("Time, sec")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_xlabel("Time, sec")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)

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

    """ Figure 4 - Generalized forces """
    fig, axs = plt.subplots(3, 2)
    ylabels = np.array((["Fx", "Mx"], ["Fy", "My"], ["Fz", "Mz"]))
    for i, _ylabel in np.ndenumerate(ylabels):
        x, y = i
        ax = axs[i]
        ax.plot(data["t"], data["FM"].squeeze(-1)[:, 2 * x + y], "k-")
        ax.set_xlim(data["t"][0], data["t"][-1])
        if i[0] == 2:
            ax.set_xlabel("Time, sec")

    plt.gcf().supylabel("Generalized Forces")

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
