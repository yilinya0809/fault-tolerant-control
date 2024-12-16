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
# Trst_corr = np.load("data/corr.npz")
Trst_corr = np.load("corr_test.npz")
VT_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc"]
theta_corr = np.rad2deg(Trst_corr["theta_corr"])
Fr = Trst_corr["Fr"]
Fp = Trst_corr["Fp"]
success = Trst_corr["success"]


""" Optimal transition trajectory """
opt_traj = {}
f = h5py.File("data/opt_corr.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)

data_opt = fym.load("data_opt_switch.h5")["env"]
data_mpc = fym.load("data/data_mpc_switch.h5")["env"]
agent = fym.load("data/data_mpc_switch.h5")["agent"]
data_ndi = fym.load("data/data_ndi_switch.h5")["env"]


""" Pre-processing """
time = data_ndi["t"]
zd = agent["Xd"][:, 0]
Vd = agent["Xd"][:, 1:]
VTd = np.linalg.norm(Vd, axis=1)
qd = agent["qd"]

Fr_trim = agent["Ud"][:, 0]
Fp_trim = agent["Ud"][:, 1]
theta_trim = agent["Ud"][:, 2]

x_opt = data_opt["plant"]["pos"][:, 0]
z_opt = data_opt["plant"]["pos"][:, 2]
V_opt = data_opt["plant"]["vel"].squeeze(-1)
VT_opt = np.linalg.norm(V_opt, axis=1)
theta_opt = data_opt["ang"][:, 1]
q_opt = data_opt["plant"]["omega"][:, 1]

x_ndi = data_ndi["plant"]["pos"][:, 0]
z_ndi = data_ndi["plant"]["pos"][:, 2]
V_ndi = data_ndi["plant"]["vel"].squeeze(-1)
VT_ndi = np.linalg.norm(V_ndi, axis=1)
theta_ndi = data_ndi["ang"][:, 1]
q_ndi = data_ndi["plant"]["omega"][:, 1]

x_mpc = data_mpc["plant"]["pos"][:, 0]
z_mpc = data_mpc["plant"]["pos"][:, 2]
V_mpc = data_mpc["plant"]["vel"].squeeze(-1)
VT_mpc = np.linalg.norm(V_mpc, axis=1)
theta_mpc = data_mpc["ang"][:, 1]
q_mpc = data_mpc["plant"]["omega"][:, 1]

Fr_opt = data_opt["Fr"]
Fp_opt = data_opt["Fp"]
rotors_opt = data_opt["ctrls"][:, 0:6]
pushers_opt = data_opt["ctrls"][:, 6:8]
Fpd_opt = data_opt["Fpd"]
thetad_opt = data_opt["angd"][:, 1]

Fr_ndi = data_ndi["Fr"]
Fp_ndi = data_ndi["Fp"]
rotors_ndi = data_ndi["ctrls"][:, 0:6]
pushers_ndi = data_ndi["ctrls"][:, 6:8]
Fpd_ndi = data_ndi["Fpd"]
thetad_ndi = data_ndi["angd"][:, 1]

Fr_mpc = data_mpc["Fr"]
Fp_mpc = data_mpc["Fp"]
rotors_mpc = data_mpc["ctrls"][:, 0:6]
pushers_mpc = data_mpc["ctrls"][:, 6:8]
Frd_mpc = data_mpc["Frd"]
Fpd_mpc = data_mpc["Fpd"]
thetad_mpc = data_mpc["angd"][:, 1]

# """ Figure 1 - States """
# fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

# """ Column 1 - States: Position """
# ax = axes[0, 0]
# # ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 0].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 0].squeeze(-1), "b-",linewidth=3)
# ax.set_ylabel(r"$x$, m", fontsize=20)
# ax.set_xlim(data_ndi["t"][0], data_ndi["t"][-1])

# ax = axes[1, 0]
# # ax.plot(data_opt["t"], data_opt["posd"][:, 1], "r--",linewidth=3)
# ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 1].squeeze(-1), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 1].squeeze(-1), "r-",linewidth=3)
# ax.set_ylabel(r"$y$, m", fontsize=20)
# ax.set_ylim([-1, 1])

# ax = axes[2, 0]
# # ax.plot(tspan, opt_traj["X"][0, :], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["posd"][:, 2], "r--",linewidth=3)
# ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 2].squeeze(-1), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 2].squeeze(-1), "r-",linewidth=3)
# ax.set_ylabel(r"$z$, m", fontsize=20)
# ax.set_ylim([-15, -5])

# ax.set_xlabel("Time, sec", fontsize=20)

# """ Column 2 - States: Velocity """
# ax = axes[0, 1]
# ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 0].squeeze(-1), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 0].squeeze(-1), "r-",linewidth=3)
# # ax.plot(data_opt["t"], data_opt["veld"][:, 0], "r--",linewidth=3)
# ax.set_ylabel(r"$v_x$, m/s", fontsize=20)

# ax = axes[1, 1]
# ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 1].squeeze(-1), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 1].squeeze(-1), "r-",linewidth=3)
# # ax.plot(data_opt["t"], data_opt["veld"][:, 1], "r--",linewidth=3)
# ax.set_ylabel(r"$v_y$, m/s", fontsize=20)
# ax.set_ylim([-1, 1])

# ax = axes[2, 1]
# ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 2].squeeze(-1), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 2].squeeze(-1), "r-",linewidth=3)
# # ax.plot(data_opt["t"], data_opt["veld"][:, 2], "r--",linewidth=3)
# ax.set_ylabel(r"$v_z$, m/s", fontsize=20)
# ax.set_ylim([-10, 10])

# ax.set_xlabel("Time, sec", fontsize=20)

# """ Column 3 - States: Euler angles """
# ax = axes[0, 2]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["ang"][:, 0].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 0].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 0].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\phi$, deg", fontsize=20)
# ax.set_ylim([-1, 1])

# ax = axes[1, 2]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["ang"][:, 1].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 1].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 1].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\theta$, deg", fontsize=20)

# ax = axes[2, 2]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["ang"][:, 2].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 2].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 2].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\psi$, deg", fontsize=20)
# ax.set_ylim([-1, 1])

# ax.set_xlabel("Time, sec", fontsize=20)

# """ Column 4 - States: Angular rates """
# ax = axes[0, 3]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 0].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 0].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 0].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$p$, deg/s", fontsize=20)
# ax.set_ylim([-1, 1])

# ax = axes[1, 3]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 1].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 1].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 1].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$q$, deg/s", fontsize=20)

# ax = axes[2, 3]
# ax.plot(data_ndi["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 2].squeeze(-1)), "b-",linewidth=3)
# # ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 2].squeeze(-1)), "r-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 2].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$r$, deg/s", fontsize=20)
# ax.set_ylim([-1, 1])

# ax.set_xlabel("Time, sec", fontsize=20)

# fig.tight_layout()

# """ Figure 2 - Rotor inputs """
# fig, axes = plt.subplots(3, 2, sharex=True)

# ax = axes[0, 0]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 0], "b-",linewidth=3)
# ax.set_ylabel("Rotor 1", fontsize=20)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1, 0]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 1], "b-",linewidth=3)
# ax.set_ylabel("Rotor 2", fontsize=20)

# ax = axes[2, 0]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 2], "b-",linewidth=3)
# ax.set_ylabel("Rotor 3", fontsize=20)
# ax.set_xlabel("Time, sec", fontsize=20)

# ax = axes[0, 1]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 3], "b-",linewidth=3)
# ax.set_ylabel("Rotor 4", fontsize=20)

# ax = axes[1, 1]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 4], "b-",linewidth=3)
# ax.set_ylabel("Rotor 5", fontsize=20)

# ax = axes[2, 1]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 5], "b-",linewidth=3)
# ax.set_ylabel("Rotor 6", fontsize=20)
# ax.set_xlabel("Time, sec", fontsize=20)

# plt.tight_layout()
# fig.subplots_adjust(wspace=0.3)
# fig.align_ylabels(axes)

# """ Figure 3 - Pusher input """
# fig, axes = plt.subplots(2, 1, sharex=True)

# ax = axes[0]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 6], "b-",linewidth=3)
# ax.set_ylabel("Pusher 1", fontsize=20)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1]
# ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 7], "b-",linewidth=3)
# ax.set_ylabel("Pusher 2", fontsize=20)
# ax.set_xlabel("Time, sec", fontsize=20)

# plt.tight_layout()
# fig.align_ylabels(axes)

# # """ Figure 5 - Thrust """
# fig, axes = plt.subplots(2, 1, sharex=True)

# ax = axes[0]
# ax.plot(data_opt["t"], data_opt["Frd"], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["Fr"].squeeze(-1), "b-",linewidth=3)
# ax.set_ylabel(r"$F_{rotors}$, N",linewidth=3)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1]
# ax.plot(data_opt["t"], data_opt["Fpd"], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["Fp"].squeeze(-1), "b-",linewidth=3)
# ax.set_ylabel(r"$F_{pushers}$, N",linewidth=3)
# ax.set_xlabel("Time, sec",linewidth=3)

# plt.tight_layout()
# fig.align_ylabels(axes)

# plt.show()


def plot():
    """Figure 1 - States"""
    fig, axes = plt.subplots(3, 1, figsize=(10,8))

    ax = axes[0]
    ax.plot(time, z_ndi, "g-.", linewidth=3, label='NDI')
    ax.plot(time, z_mpc, "b--", linewidth=3, label='MPC-NDI')
    ax.plot(time, z_opt, "r-", linewidth=3, label='Opt-NDI')
    ax.plot(time, zd, "k:", linewidth=3, label='Trim')
    ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.set_ylim([-12, -8])
    # ax.legend(fontsize=15)
    fig.tight_layout()

    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = axes[1]
    ax.plot(time, VT_ndi, "g-.", linewidth=3, label='NDI')
    ax.plot(time, VT_mpc, "b--", linewidth=3, label='MPC-NDI')
    ax.plot(time, VT_opt, "r-", linewidth=3, label='Opt-NDI')
    ax.plot(time, VTd, "k:", linewidth=3, label='Trim')
    ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    # ax.legend(fontsize=20)
    fig.tight_layout()

    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = axes[2]
    ax.plot(time, np.rad2deg(theta_ndi), "g-.", linewidth=3, label="NDI")
    ax.plot(time, np.rad2deg(theta_mpc), "b--", linewidth=3, label="MPC-NDI")
    ax.plot(time, np.rad2deg(theta_opt), "r-", linewidth=3, label="Opt-NDI")
    ax.plot(time, np.rad2deg(theta_trim), "k:", linewidth=3, label="Trim")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)

    ax.legend(loc='lower right', fontsize=15)
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 2 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(time, rotors_mpc[:, 0], "b:", linewidth=2, label="MPC-NDI")
    ax.plot(time, rotors_opt[:, 0], "r-", linewidth=3, label="Opt-NDI")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 1], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 1], "b:", linewidth=2)
    ax.plot(time, rotors_opt[:, 1], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 1]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 2], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 2], "b:", linewidth=2)
    ax.plot(time, rotors_opt[:, 2], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)

    ax = axes[1, 1]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 3], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 3], "b:", linewidth=2)
    ax.plot(time, rotors_opt[:, 3], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 2]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 4], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 4], "b:", linewidth=2)
    ax.plot(time, rotors_opt[:, 4], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)

    ax = axes[1, 2]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, rotors_ndi[:, 5], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 5], "b:", linewidth=2)
    ax.plot(time, rotors_opt[:, 5], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 3]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, pushers_ndi[:, 0], "g-.", linewidth=3)
    ax.plot(time, pushers_mpc[:, 0], "b--", linewidth=2)
    ax.plot(time, pushers_opt[:, 0], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)

    ax = axes[1, 3]
    ax.plot(time, np.ones((len(time), 1)), "k:")
    ax.plot(time, np.zeros((len(time), 1)), "k:")
    ax.plot(time, pushers_ndi[:, 1], "g-.", linewidth=3)
    ax.plot(time, pushers_mpc[:, 1], "b--", linewidth=2)
    ax.plot(time, pushers_opt[:, 1], "r-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Time, sec", fontsize=14)
    ax.set_ylabel("Pusher 2", fontsize=14)

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    fig.legend(
        handles=handles,
    labels=["NDI", "MPC-NDI", "Opt-NDI"],
    loc="upper center",  # Position the legend above the figure
    bbox_to_anchor=(0.5, 1.005),  # Center the legend horizontally
    ncol=3,  # Number of columns
    fontsize=14,  # Font size for legend
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    #     """ Figure 3 - Forces of NMPC """
    #     fig, axes = plt.subplots(3, 1)

    #     """ Row 1 - Rotor forces """
    #     ax = axes[0]
    #     ax.plot(time, -Fr_mpc, "b-",linewidth=3)
    #     ax.plot(time, -Frd_mpc, "--r",linewidth=3)
    #     ax.set_xlim(time[0], time[-1])
    #     ax.set_ylabel(r"$F_{rotors}$, N", fontsize=13)

    #     """ Row 2 - Pusher forces """
    #     ax = axes[1]
    #     ax.plot(time, Fp_mpc, "b-",linewidth=3)
    #     ax.plot(time, Fpd_mpc, "--r",linewidth=3)
    #     ax.set_xlim(time[0], time[-1])
    #     ax.set_ylabel(r"$F_{pushers}$, N", fontsize=13)

    #     """ Row 3 - Pitch angle """
    #     ax = axes[2]
    #     l1 = ax.plot(time, np.rad2deg(theta_mpc), "b-",linewidth=3)
    #     l2 = ax.plot(time, np.rad2deg(thetad_mpc), "--r",linewidth=3)
    #     ax.set_xlim(time[0], time[-1])
    #     ax.set_ylabel(r"$\theta$, deg", fontsize=13)
    #     ax.set_xlabel("Time, sec",linewidth=3)

    #     fig.legend(
    #         [l1, l2],
    #         labels=["NMPC-DI Results", "NMPC Solutions"],
    #         loc="lower center",
    #         bbox_to_anchor=(0.55, 0),
    #         ncol=2,
    #         fontsize=20,
    #     )

    #     fig.tight_layout(h_pad=0.2)

    # """ Figure 4 - Forces of NDI """
    # fig, axes = plt.subplots(2, 1)

    # """ Row 1 - Pusher forces """
    # ax = axes[0]
    # ax.plot(time, Fp_ndi, "r-",linewidth=3)
    # ax.plot(time, Fpd_ndi, "--r",linewidth=3)
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec",linewidth=3)
    # ax.set_ylabel(r"$F_p$, N",linewidth=3)

    # """ Row 2 - Pitch angle """
    # ax = axes[1]
    # l1 = ax.plot(time, theta_ndi, "r-",linewidth=3)
    # l2 = ax.plot(time, thetad_ndi, "--r",linewidth=3)
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec",linewidth=3)
    # ax.set_ylabel(r"$\theta$, N",linewidth=3)

    # fig.legend([l1, l2],
    #            labels=["NDI", "Commands"],
    #            loc="lower center",
    #            bbox_to_anchor=(0.5, 0),
    #            ncol=2,
    #            )

    # fig.tight_layout()

    """ Figure 3 - Transition Corridor """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    VT, theta = np.meshgrid(VT_corr, theta_corr)
    ax.scatter(VT, theta, s=success.T, c="b")

    ax.plot(VT_ndi, np.rad2deg(theta_ndi), "g-.", linewidth=5, label="NDI")
    ax.plot(VT_mpc, np.rad2deg(theta_mpc), "b--", linewidth=5, label="MPC-NDI")
    ax.plot(VT_opt, np.rad2deg(theta_opt), "r-", linewidth=5, label="Opt-NDI")
    ax.set_xlabel("V, m/s", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.legend(fontsize=20)
    fig.tight_layout()

    plt.show()


def total_cost():
    Vxd = Vd[:, 0]
    Vzd = Vd[:, 1]
    
    Vx_opt = V_opt[:, 0]
    Vz_opt = V_opt[:, 2]
    Vx_ndi = V_ndi[:, 0]
    Vz_ndi = V_ndi[:, 2]
    Vx_mpc = V_mpc[:, 0]
    Vz_mpc = V_mpc[:, 2]

    cost_opt = 0
    cost_ndi = 0
    cost_mpc = 0
    Q = 10 * np.diag((1, 1, 1))

    for k in range(np.size(time)):
        Vx_opt = V_opt[k, 0]
        Vz_opt = V_opt[k, 2]
        Vx_ndi = V_ndi[k, 0]
        Vz_ndi = V_ndi[k, 2]
        Vx_mpc = V_mpc[k, 0]
        Vz_mpc = V_mpc[k, 2]

        err_ndi = np.vstack(
            (
                z_ndi[k] - zd[k],
                Vx_ndi - Vxd[k],
                Vz_ndi - Vzd[k],
            )
        )

        err_mpc = np.vstack(
            (
                z_mpc[k] - zd[k],
                Vx_mpc - Vxd[k],
                Vz_mpc - Vzd[k],
            )
        )

        cost_ndi = cost_ndi + err_ndi.T @ Q @ err_ndi
        cost_mpc = cost_mpc + err_mpc.T @ Q @ err_mpc

    return cost_ndi, cost_mpc


def rotor_cost():
    cost_opt = 0
    cost_ndi = 0
    cost_mpc = 0
    rcost_opt = 0
    rcost_ndi = 0
    rcost_mpc = 0

    A = np.diag((1, 1, 1, 1, 1, 1, 1, 1))
    B = np.diag((1, 1, 1, 1, 1, 1))

    for k in range(np.size(time)):
        r_opt = rotors_opt[k]
        r_ndi = rotors_ndi[k]
        r_mpc = rotors_mpc[k]
        rcost_opt = rcost_opt + r_opt.T @ B @ r_opt
        rcost_ndi = rcost_ndi + r_ndi.T @ B @ r_ndi
        rcost_mpc = rcost_mpc + r_mpc.T @ B @ r_mpc
        rcost = np.vstack((rcost_ndi, rcost_mpc, rcost_opt))

        ctrls_opt = np.vstack((rotors_opt[k], pushers_opt[k]))
        ctrls_ndi = np.vstack((rotors_ndi[k], pushers_ndi[k]))
        ctrls_mpc = np.vstack((rotors_mpc[k], pushers_mpc[k]))
        cost_opt = cost_opt + ctrls_opt.T @ A @ ctrls_opt
        cost_ndi = cost_ndi + ctrls_ndi.T @ A @ ctrls_ndi
        cost_mpc = cost_mpc + ctrls_mpc.T @ A @ ctrls_mpc
        cost = np.vstack((cost_ndi, cost_mpc, cost_opt))

    return rcost, cost


if __name__ == "__main__":
    rcost, cost = rotor_cost()
    print(rcost)
    print(cost)
    plot()
