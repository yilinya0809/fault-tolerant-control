import argparse

import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle
from scipy.integrate import cumtrapz

import ftc
from ftc.models.LC62R_lin import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
)

""" Forward Transition Reference """
FTC = np.load("data/corr_forward.npz")
VT_ftc = FTC["VT_corr"]
theta_ftc = np.rad2deg(FTC["theta_corr"])
success_ftc = FTC["success"]

ftc_traj = {}
f = h5py.File("data/opt_forward.h5", "r")
ftc_traj["tf"] = f.get("tf")[()]
ftc_traj["X"] = f.get("X")[:]
ftc_traj["U"] = f.get("U")[:]

N = np.shape(ftc_traj["U"])[1]
t_ftc = np.linspace(0, ftc_traj["tf"], N + 1)
Vxd_ftc = ftc_traj["X"][1, :]
Vzd_ftc = ftc_traj["X"][2, :]
thetad_ftc = ftc_traj["U"][2, :]
xdot = []
for i in range(N):
    xdot.append(
        Vxd_ftc[i + 1] * np.cos(thetad_ftc[i]) + Vzd_ftc[i + 1] * np.sin(thetad_ftc[i])
    )

Xd_ftc = cumtrapz(xdot, t_ftc[1:], initial=0)

""" Backward Transition Reference """
BTC = np.load("data/corr_backward.npz")
VT_btc = BTC["VT_corr"]
theta_btc = np.rad2deg(BTC["theta_corr"])
success_btc = BTC["success"]

btc_traj = {}
f = h5py.File("data/opt_backward.h5", "r")
btc_traj["tf"] = f.get("tf")[()]
btc_traj["X"] = f.get("X")[:]
btc_traj["U"] = f.get("U")[:]

t_btc = np.linspace(0, btc_traj["tf"], N + 1)
Vxd_btc = btc_traj["X"][1, :]
Vzd_btc = btc_traj["X"][2, :]
thetad_btc = btc_traj["U"][2, :]
xdot = []
for i in range(N):
    xdot.append(
        Vxd_btc[i + 1] * np.cos(thetad_btc[i]) + Vzd_btc[i + 1] * np.sin(thetad_btc[i])
    )

Xd_btc = cumtrapz(xdot, t_btc[1:], initial=0)


class MyEnv(fym.BaseEnv):
    VT_cruise = 45
    h = 10
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 50,
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

    # Optimal Transition Trajectory Reference
    def get_ref(self, t):
        zd = -self.h

        if t <= t_ftc[-1]:  # FTC
            xd = np.interp(t, t_ftc[1:], Xd_ftc[:])
            Vxd_ftc = np.interp(t, t_ftc, ftc_traj["X"][1, :])
            Vzd_ftc = np.interp(t, t_ftc, ftc_traj["X"][2, :])
            veld = np.vstack((Vxd_ftc, 0, Vzd_ftc))
            thetad = np.interp(t, t_ftc[1:], ftc_traj["U"][2, :])
            mode = "FTC"

        elif t_ftc[-1] < t <= 20:  # FW
            xd = Xd_ftc[-1] + self.VT_cruise * (t - t_ftc[-1])
            veld = np.vstack((ftc_traj["X"][1, -1], 0, ftc_traj["X"][2, -1]))
            thetad = ftc_traj["U"][2, -1]
            mode = "FW"

        elif 20 < t <= 20 + t_btc[-1]:
            xd = (
                Xd_ftc[-1]
                + self.VT_cruise * (20 - t_ftc[-1])
                + np.interp(t - 20, t_btc[1:], Xd_btc[:])
            )
            Vxd_btc = np.interp(t - 20, t_btc, btc_traj["X"][1, :])
            Vzd_btc = np.interp(t - 20, t_btc, btc_traj["X"][2, :])
            veld = np.vstack((Vxd_btc, 0, Vzd_btc))
            thetad = np.interp(t - 20, t_btc[1:], btc_traj["U"][2, :])
            mode = "BTC"

        elif 20 + t_btc[-1] < t:
            xd = Xd_ftc[-1] + self.VT_cruise * (20 - t_ftc[-1]) + Xd_btc[-1]
            veld = np.zeros((3, 1))
            thetad = 0
            mode = "HV"

        return xd, zd, veld, thetad, mode

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        _, _, _, _, mode = self.get_ref(t)
        VT = np.linalg.norm(vel)

        if mode == "FW" or mode == "HV":
            ctrls0, controller_info = self.controller_lqr.get_control(t, self)
        else:
            ctrls0, controller_info = self.controller_trst.get_control(t, self)

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
    flogger = fym.Logger("data_opt_switch.h5")

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
    data = fym.load("data_opt_switch.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 2, figsize=(7, 8.5), squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-", linewidth=3)
    ax.plot(data["t"], data["posd"][:, 0], "r--")
    ax.set_ylabel(r"$x$, m", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.grid()

    ax = axes[1, 0]
    # ax.plot(t_ftc, ftc_traj["X"][0, :], "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-", linewidth=3)
    ax.plot(data["t"], data["posd"][:, 2], "r--")
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.set_ylim([-12, -8])
    ax.grid()

    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-", linewidth=3)
    ax.plot(data["t"], data["veld"][:, 0], "r--")
    ax.set_ylabel(r"$V_x^B$, m/s", fontsize=20)
    ax.grid()

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-", linewidth=3)
    ax.plot(data["t"], data["veld"][:, 2], "r--")
    ax.set_ylabel(r"$V_z^B$, m/s", fontsize=20, labelpad=-2)
    ax.set_ylim([-10, 5])
    ax.grid()

    ax = axes[2, 0]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-", linewidth=3)
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.grid()
    ax.set_ylabel(r"$\theta$, deg", fontsize=20, labelpad=-5)
    ax.set_xlabel("Time, s", fontsize=20)

    ax = axes[2, 1]
    ax.plot(
        data["t"],
        np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)),
        "b-",
        linewidth=3,
    )
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s", fontsize=20, labelpad=-5)
    ax.set_xlabel("Time, s", fontsize=20)
    ax.grid()

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.legend(
        labels=["Responses", "Optimal commands"],
        loc="upper center",  # Position the legend above the figure
        bbox_to_anchor=(0.5, 1.005),  # Center the legend horizontally
        ncol=2,  # Number of columns
        fontsize=14,  # Font size for legend
    )

    """ Figure 2 - Rotor inputs """
    fig, axes = plt.subplots(4, 2, figsize=(7, 8.5), sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[0, 1]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--", linewidth=2)
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=20)

    ax = axes[1, 0]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=20)

    ax = axes[1, 1]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=20)

    ax = axes[2, 0]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=20)

    ax = axes[2, 1]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=20)

    ax = axes[3, 0]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=20)
    ax.set_xlabel("Time, s", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[3, 1]
    ax.plot(data["t"], np.ones((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], np.zeros((len(data["t"]), 1)), "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-", linewidth=2)
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=20)
    ax.set_xlabel("Time, s", fontsize=20)

    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.2)
    # fig.align_ylabels(axes)

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
    ax.set_xlabel("Time, s")

    #     """ Figure 4 - Transition Corridor """
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #     VT, theta = np.meshgrid(VT_ftc, theta_ftc)
    #     ax.scatter(VT, theta, s=success_ftc.T, c="b")

    #     ax.plot(np.linalg.norm(data["plant"]["vel"].squeeze(-1), axis=1), np.rad2deg(data["ang"][:, 1]), "r-", linewidth=5)
    #     ax.set_xlabel("V, m/s", fontsize=20)
    #     ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    #     ax.legend(fontsize=20)
    #     fig.tight_layout()

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
