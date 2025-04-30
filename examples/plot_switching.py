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
    """Figure 1 - FTC states"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(t_ftc, data_ndi_fw["z"], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["z"], "r--", linewidth=3, label="MPC")
    ax.plot(t_ftc, data_opt_fw["z"], "b-", linewidth=3, label="Corr-Opt")
    ax.plot(t_ftc, data_mpc_fw["zd"], "k:", linewidth=3, label="Trim")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.set_ylim([-12, -8])
    fig.tight_layout()

    ax = axes[1]
    ax.plot(t_ftc, data_ndi_fw["VT"], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["VT"], "r--", linewidth=3, label="MPC")
    ax.plot(t_ftc, data_opt_fw["VT"], "b-", linewidth=3, label="Corr-Opt")
    ax.plot(t_ftc, data_mpc_fw["VTd"], "k:", linewidth=3, label="Trim")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    fig.tight_layout()

    ax = axes[2]
    ax.plot(t_ftc, np.rad2deg(data_ndi_fw["theta"]), "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, np.rad2deg(data_mpc_fw["theta"]), "r--", linewidth=3, label="MPC")
    ax.plot(
        t_ftc, np.rad2deg(data_opt_fw["theta"]), "b-", linewidth=3, label="Corr-Opt"
    )
    ax.plot(
        t_ftc, np.rad2deg(data_mpc_fw["theta_trim"]), "k:", linewidth=3, label="Trim"
    )
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_xlabel("Time, s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)

    ax.legend(loc="lower right", fontsize=15)
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 2 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    ax = axes[0, 0]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 0], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 1], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 1]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 2], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 2], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 2], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)

    ax = axes[1, 1]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 3], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 3], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 3], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 2]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 4], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 4], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 4], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)

    ax = axes[1, 2]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 5], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 5], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 5], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 3]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["pushers"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["pushers"][:, 0], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["pushers"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)

    ax = axes[1, 3]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["pushers"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["pushers"][:, 1], "r:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["pushers"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    fig.legend(
        handles=handles,
        labels=["NDI", "MPC", "Corr-Opt"],
        loc="upper center",  # Position the legend above the figure
        bbox_to_anchor=(0.5, 1.005),  # Center the legend horizontally
        ncol=3,  # Number of columns
        fontsize=14,  # Font size for legend
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    """ Figure 3 - Transition Corridor """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    VT, theta = np.meshgrid(VT_ftc, theta_ftc)
    ax.scatter(VT, theta, s=success_ftc.T, c="gray")

    ax.plot(
        data_ndi_fw["VT"],
        np.rad2deg(data_ndi_fw["theta"]),
        "g-.",
        linewidth=5,
        label="NDI",
    )
    ax.plot(
        data_mpc_fw["VT"],
        np.rad2deg(data_mpc_fw["theta"]),
        "r--",
        linewidth=5,
        label="MPC",
    )
    ax.plot(
        data_opt_fw["VT"],
        np.rad2deg(data_opt_fw["theta"]),
        "b-",
        linewidth=5,
        label="Corr-Opt",
    )
    ax.set_xlabel("V, m/s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.legend(fontsize=15)
    fig.tight_layout()

    """Figure 4 - BTC states """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(t_btc, data_ndi_bw["z"], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["z"], "r--", linewidth=3, label="MPC")
    ax.plot(t_btc, data_opt_bw["z"], "b-", linewidth=3, label="Corr-Opt")
    ax.plot(t_btc, data_mpc_bw["zd"], "k:", linewidth=3, label="Trim")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylabel(r"$z$, m", fontsize=20)
    # ax.set_ylim([-15, -5])
    ax.legend(loc="lower right", fontsize=15)
    fig.tight_layout()

    ax = axes[1]
    ax.plot(t_btc, data_ndi_bw["VT"], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["VT"], "r--", linewidth=3, label="MPC")
    ax.plot(t_btc, data_opt_bw["VT"], "b-", linewidth=3, label="Corr-Opt")
    ax.plot(t_btc, data_mpc_bw["VTd"], "k:", linewidth=3, label="Trim")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    fig.tight_layout()

    ax = axes[2]
    ax.plot(t_btc, np.rad2deg(data_ndi_bw["theta"]), "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, np.rad2deg(data_mpc_bw["theta"]), "r--", linewidth=3, label="MPC")
    ax.plot(
        t_btc, np.rad2deg(data_opt_bw["theta"]), "b-", linewidth=3, label="Corr-Opt"
    )
    ax.plot(
        t_btc, np.rad2deg(data_mpc_bw["theta_trim"]), "k:", linewidth=3, label="Trim"
    )
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_xlabel("Time, s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)

    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 5 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    ax = axes[0, 0]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 0], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 1], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 1]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 2], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 2], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 2], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)

    ax = axes[1, 1]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 3], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 3], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 3], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 2]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 4], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 4], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 4], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)

    ax = axes[1, 2]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["rotors"][:, 5], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["rotors"][:, 5], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["rotors"][:, 5], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    ax = axes[0, 3]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["pushers"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["pushers"][:, 0], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["pushers"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)

    ax = axes[1, 3]
    ax.plot(t_btc, np.ones((len(t_btc), 1)), "k:")
    ax.plot(t_btc, np.zeros((len(t_btc), 1)), "k:")
    ax.plot(t_btc, data_ndi_bw["pushers"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_btc, data_mpc_bw["pushers"][:, 1], "r:", linewidth=2, label="MPC")
    ax.plot(t_btc, data_opt_bw["pushers"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_btc[0], t_btc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    fig.legend(
        handles=handles,
        labels=["NDI", "MPC", "Corr-Opt"],
        loc="upper center",  # Position the legend above the figure
        bbox_to_anchor=(0.5, 1.005),  # Center the legend horizontally
        ncol=3,  # Number of columns
        fontsize=14,  # Font size for legend
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    """ Figure 6 - Transition Corridor """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    VT, theta = np.meshgrid(VT_btc, theta_btc)
    ax.scatter(VT, theta, s=success_btc.T, c="grey")

    ax.plot(
        data_ndi_bw["VT"],
        np.rad2deg(data_ndi_bw["theta"]),
        "g-.",
        linewidth=5,
        label="NDI",
    )
    ax.plot(
        data_mpc_bw["VT"],
        np.rad2deg(data_mpc_bw["theta"]),
        "r--",
        linewidth=5,
        label="MPC",
    )
    ax.plot(
        data_opt_bw["VT"],
        np.rad2deg(data_opt_bw["theta"]),
        "b-",
        linewidth=5,
        label="Corr-Opt",
    )
    ax.set_xlabel("V, m/s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.legend(fontsize=15)
    fig.tight_layout()


    """ Figure 7 - Full mode switching """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))

    ax = axes[0]
    ax.plot(t_full, data_opt_full["z"], "b-", linewidth=3, label="Response")
    ax.plot(t_full, data_opt_full["posd"][:, 2], "r--", linewidth=2, label="Command")
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
    ax.plot(t_full, data_opt_full["VT"], "b-", linewidth=3, label="Response")
    ax.plot(t_full, data_opt_full["Veld"], "r--", linewidth=2, label="Command")
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
        t_full, np.rad2deg(data_opt_full["theta"]), "b-", linewidth=3, label="Response"
    )
    ax.plot(
        t_full, np.rad2deg(data_opt_full["angd"][:, 1]), "r--", linewidth=2, label="Command"
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
    ax.plot(t_full, data_opt_full["rotors"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    ax.grid()
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["rotors"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 1]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["rotors"][:, 2], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)
    ax.grid()

    ax = axes[1, 1]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["rotors"][:, 3], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 2]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["rotors"][:, 4], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)
    ax.grid()

    ax = axes[1, 2]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["rotors"][:, 5], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()

    ax = axes[0, 3]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["pushers"][:, 0], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)
    ax.grid()

    ax = axes[1, 3]
    ax.plot(t_full, np.ones((len(t_full), 1)), "k:")
    ax.plot(t_full, np.zeros((len(t_full), 1)), "k:")
    ax.plot(t_full, data_opt_full["pushers"][:, 1], "b-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_full[0], t_full[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, s", fontsize=14)
    ax.grid()
    fig.tight_layout()

    plt.show()


def cost(data, z_trim, V_trim):
    cost = 0
    Q = 10 * np.diag((1, 1, 1))
    for k in range(np.size(data["time"])):
        err = np.vstack((data["z"][k], data["V"][k, 0], data["V"][k, 1])) - np.vstack(
            (z_trim[k], V_trim[k, 0], V_trim[k, 1])
        )
        cost += err.T @ Q @ err
    return cost


def rotor_cost(data):
    cost = 0
    A = np.diag((1, 1, 1, 1, 1, 1, 1, 1))
    for k in range(np.size(data["time"])):
        ctrls = np.vstack((data["rotors"][k, :], data["pushers"][k, :]))
        cost += ctrls.T @ A @ ctrls

    return cost


if __name__ == "__main__":
    opt_switch = fym.load("data_opt_switch.h5")["env"]
    fw_opt = fym.load("data/data_opt_forward.h5")["env"]
    fw_mpc = fym.load("data/data_mpc_forward.h5")["env"]
    fw_mpc_agent = fym.load("data/data_mpc_forward.h5")["agent"]
    fw_ndi = fym.load("data/data_ndi_forward.h5")["env"]
    bw_opt = fym.load("data/data_opt_backward.h5")["env"]
    bw_mpc = fym.load("data/data_mpc_backward.h5")["env"]
    bw_mpc_agent = fym.load("data/data_mpc_backward.h5")["agent"]
    bw_ndi = fym.load("data/data_ndi_backward.h5")["env"]

    data_opt_full = refine(opt_switch)
    data_opt_fw = refine(fw_opt)
    data_opt_bw = refine(bw_opt)
    data_mpc_fw = refine(fw_mpc, fw_mpc_agent)
    data_mpc_bw = refine(bw_mpc, bw_mpc_agent)
    data_ndi_fw = refine(fw_ndi)
    data_ndi_bw = refine(bw_ndi)

    t_ftc = data_opt_fw["time"]
    t_btc = data_opt_bw["time"]
    t_full = data_opt_full["time"]

    fw_z_trim = data_mpc_fw["zd"][:, 0]
    fw_V_trim = data_mpc_fw["Vd"][:, :, 0]

    hv_z_trim = data_mpc_bw["zd"][:, 0]
    hv_V_trim = data_mpc_bw["Vd"][:, :, 0]

    cost_opt_fw = cost(data_opt_fw, fw_z_trim, fw_V_trim)
    cost_opt_bw = cost(data_opt_fw, hv_z_trim, hv_V_trim)
    cost_mpc_fw = cost(data_mpc_fw, fw_z_trim, fw_V_trim)
    cost_mpc_bw = cost(data_mpc_fw, hv_z_trim, hv_V_trim)
    cost_ndi_fw = cost(data_ndi_fw, fw_z_trim, fw_V_trim)
    cost_ndi_bw = cost(data_ndi_fw, hv_z_trim, hv_V_trim)

    rcost_opt_fw = rotor_cost(data_opt_fw)
    rcost_opt_bw = rotor_cost(data_opt_bw)
    rcost_mpc_fw = rotor_cost(data_mpc_fw)
    rcost_mpc_bw = rotor_cost(data_mpc_bw)
    rcost_ndi_fw = rotor_cost(data_ndi_fw)
    rcost_ndi_bw = rotor_cost(data_ndi_bw)

    print(rcost_opt_fw, rcost_opt_bw)
    print(rcost_mpc_fw, rcost_mpc_bw)
    print(rcost_ndi_fw, rcost_ndi_bw)
    plot()
