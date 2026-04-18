import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
)

""" Transition Corridor """
FTC = np.load("data/corr_forward_wide.npz")
VT_ftc = FTC["VT_corr"]
theta_ftc = np.rad2deg(FTC["theta_corr"])
success_ftc = FTC["success"]

BTC = np.load("data/corr_backward.npz")
VT_btc = BTC["VT_corr"]
theta_btc = np.rad2deg(BTC["theta_corr"])
success_btc = BTC["success"]

""" Control results """


def refine(data, agent=None):
    result = {
        "time": data["t"],
        "x": data["plant"]["pos"][:, 0],
        "z": data["plant"]["pos"][:, 2],
        "posd": data["posd"],
        "angd": data["angd"],
        "V": data["plant"]["vel"].squeeze(-1),
        "VT": np.linalg.norm(data["plant"]["vel"], axis=1),
        "Veld": np.linalg.norm(data["veld"], axis=1),
        "theta": data["ang"][:, 1],
        "q": data["plant"]["omega"][:, 1],
        "Fr": data["Fr"],
        "Fp": data["Fp"],
        "rotors": data["ctrls"][:, 0:6],
        "pushers": data["ctrls"][:, 6:8],
    }

    if agent is not None:
        result.update(
            {
                "zd": agent["Xd"][:, 0],
                "Vd": agent["Xd"][:, 1:],
                "VTd": np.linalg.norm(agent["Xd"][:, 1:], axis=1),
                "qd": agent["qd"],
                "Fr_trim": agent["Ud"][:, 0],
                "Fp_trim": agent["Ud"][:, 1],
                "theta_trim": agent["Ud"][:, 2],
            }
        )

    return result


def plot():
    """ Figure 7 - Full mode switching """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))

    ax = axes[0]
    ax.plot(t_full, data_head10["z"], "b-", linewidth=3, label="headwind 10%")
    ax.plot(t_full, data_head20["z"], "g-.", linewidth=3, label="headwind 20%")
    ax.plot(t_full, data_tail10["z"], color="coral", linestyle="--", linewidth=3, label="tailwind 10%")
    ax.plot(t_full, data_head10["posd"][:, 2], "r--", linewidth=1, label="Command")
    ax.axvspan(0, 5, color="gray", alpha=0.1)
    ax.axvspan(5, 14.5, color="g", alpha=0.25)
    ax.axvspan(14.5, 25, color="gray", alpha=0.1)
    ax.axvspan(25, 47, color="g", alpha=0.25)
    ax.axvspan(47, 60, color="gray", alpha=0.1)
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.grid()
    # ax.set_ylim([-12, -8])

    ax = axes[1]
    ax.plot(t_full, data_head10["VT"], "b-", linewidth=3, label="headwind 10%")
    ax.plot(t_full, data_head20["VT"], "g-.", linewidth=3, label="headwind 20%")
    ax.plot(t_full, data_tail10["VT"], color="coral", linestyle="--", linewidth=3, label="tailwind 10%")
    ax.plot(t_full, data_head10["Veld"], "r--", linewidth=1, label="Command")
    ax.axvspan(0, 5, color="gray", alpha=0.1)
    ax.axvspan(5, 14.5, color="g", alpha=0.25)
    ax.axvspan(14.5, 25, color="gray", alpha=0.1)
    ax.axvspan(25, 47, color="g", alpha=0.25)
    ax.axvspan(47, 60, color="gray", alpha=0.1)

    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    ax.grid()

    ax = axes[2]
    ax.plot(
        t_full, np.rad2deg(data_head10["theta"]), "b-", linewidth=3, label="headwind 10%"
    )
    ax.plot(
        t_full, np.rad2deg(data_head20["theta"]), "g-.", linewidth=3, label="headwind 20%"
    )
    ax.plot(
        t_full, np.rad2deg(data_tail10["theta"]), color="coral", linestyle="--", linewidth=3, label="tailwind 10%"
    )
    ax.plot(
        t_full, np.rad2deg(data_head10["angd"][:, 1]), "r--", linewidth=1, label="Command"
    )
    ax.axvspan(0, 5, color="gray", alpha=0.1)
    ax.axvspan(5, 14.5, color="g", alpha=0.25)
    ax.axvspan(14.5, 25, color="gray", alpha=0.1)
    ax.axvspan(25, 47, color="g", alpha=0.25)
    ax.axvspan(47, 60, color="gray", alpha=0.1)

    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_xlabel("Time, s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.grid()

    ax.legend(loc="lower right", fontsize=15)
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 8 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    ax = axes[0, 0]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    ax.grid()
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 1]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 2], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)
    ax.grid()

    ax = axes[1, 1]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 3], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 2]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 4], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)
    ax.grid()

    ax = axes[1, 2]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["rotors"][:, 5], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 3]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["pushers"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)
    ax.grid()

    ax = axes[1, 3]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_head10["pushers"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    opt_switch = fym.load("data/data_opt_switch.h5")["env"]
    data_head10 = refine(opt_switch)
    t_full = data_head10["time"]

    opt_head20 = fym.load("data_opt_switch_head20.h5")["env"]
    data_head20 = refine(opt_head20)

    opt_tail10 = fym.load("data_opt_switch_tail10.h5")["env"]
    data_tail10 = refine(opt_tail10)

    plot()
