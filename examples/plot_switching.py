import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np

""" Optimal transition trajectory """
opt_traj = {}
f = h5py.File("data/opt_corr.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)

data_opt = fym.load("data/data_opt_switch.h5")["env"]
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
# ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 0].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 0].squeeze(-1), "k-",linewidth=3)
# ax.set_ylabel(r"$x$, m",linewidth=3)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1, 0]
# ax.plot(data_opt["t"], data_opt["posd"][:, 1], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 1].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 1].squeeze(-1), "k-",linewidth=3)
# ax.set_ylabel(r"$y$, m",linewidth=3)
# ax.set_ylim([-1, 1])

# ax = axes[2, 0]
# # ax.plot(tspan, opt_traj["X"][0, :], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["posd"][:, 2], "r--",linewidth=3)
# ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 2].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 2].squeeze(-1), "k-",linewidth=3)
# ax.set_ylabel(r"$z$, m",linewidth=3)
# ax.set_ylim([-15, -5])

# ax.set_xlabel("Time, sec",linewidth=3)

# """ Column 2 - States: Velocity """
# ax = axes[0, 1]
# ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 0].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 0].squeeze(-1), "k-",linewidth=3)
# ax.plot(data_opt["t"], data_opt["veld"][:, 0], "r--",linewidth=3)
# ax.set_ylabel(r"$v_x$, m/s",linewidth=3)

# ax = axes[1, 1]
# ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 1].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 1].squeeze(-1), "k-",linewidth=3)
# ax.plot(data_opt["t"], data_opt["veld"][:, 1], "r--",linewidth=3)
# ax.set_ylabel(r"$v_y$, m/s",linewidth=3)
# ax.set_ylim([-1, 1])

# ax = axes[2, 1]
# ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 2].squeeze(-1), "b-",linewidth=3)
# ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 2].squeeze(-1), "k-",linewidth=3)
# ax.plot(data_opt["t"], data_opt["veld"][:, 2], "r--",linewidth=3)
# ax.set_ylabel(r"$v_z$, m/s",linewidth=3)
# ax.set_ylim([-10, 10])

# ax.set_xlabel("Time, sec",linewidth=3)

# """ Column 3 - States: Euler angles """
# ax = axes[0, 2]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 0].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 0].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 0].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\phi$, deg",linewidth=3)
# ax.set_ylim([-1, 1])

# ax = axes[1, 2]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 1].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 1].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 1].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\theta$, deg",linewidth=3)

# ax = axes[2, 2]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 2].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 2].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 2].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$\psi$, deg",linewidth=3)
# ax.set_ylim([-1, 1])

# ax.set_xlabel("Time, sec",linewidth=3)

# """ Column 4 - States: Angular rates """
# ax = axes[0, 3]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 0].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 0].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 0].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$p$, deg/s",linewidth=3)
# ax.set_ylim([-1, 1])

# ax = axes[1, 3]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 1].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 1].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 1].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$q$, deg/s",linewidth=3)

# ax = axes[2, 3]
# ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 2].squeeze(-1)), "b-",linewidth=3)
# ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 2].squeeze(-1)), "k-",linewidth=3)
# # ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 2].squeeze(-1)), "r--",linewidth=3)
# ax.set_ylabel(r"$r$, deg/s",linewidth=3)
# ax.set_ylim([-1, 1])

# ax.set_xlabel("Time, sec",linewidth=3)

# fig.tight_layout()

# """ Figure 2 - Rotor inputs """
# fig, axes = plt.subplots(3, 2, sharex=True)

# ax = axes[0, 0]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 0], "b-",linewidth=3)
# ax.set_ylabel("Rotor 1",linewidth=3)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1, 0]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 1], "b-",linewidth=3)
# ax.set_ylabel("Rotor 2",linewidth=3)

# ax = axes[2, 0]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 2], "b-",linewidth=3)
# ax.set_ylabel("Rotor 3",linewidth=3)
# ax.set_xlabel("Time, sec",linewidth=3)

# ax = axes[0, 1]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 3], "b-",linewidth=3)
# ax.set_ylabel("Rotor 4",linewidth=3)

# ax = axes[1, 1]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 4], "b-",linewidth=3)
# ax.set_ylabel("Rotor 5",linewidth=3)

# ax = axes[2, 1]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 5], "b-",linewidth=3)
# ax.set_ylabel("Rotor 6",linewidth=3)
# ax.set_xlabel("Time, sec",linewidth=3)

# plt.tight_layout()
# fig.subplots_adjust(wspace=0.3)
# fig.align_ylabels(axes)

# """ Figure 3 - Pusher input """
# fig, axes = plt.subplots(2, 1, sharex=True)

# ax = axes[0]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 6], "b-",linewidth=3)
# ax.set_ylabel("Pusher 1",linewidth=3)
# ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

# ax = axes[1]
# ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 7], "b-",linewidth=3)
# ax.set_ylabel("Pusher 2",linewidth=3)
# ax.set_xlabel("Time, sec",linewidth=3)

# plt.tight_layout()
# fig.align_ylabels(axes)

# """ Figure 5 - Thrust """
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
    # fig, axes = plt.subplots(2, 2)
    fig, axes = plt.subplots(3, 1)
    # fig.suptitle("State trajectories",linewidth=3)

    """ Row 1 - z, VT """
    ax = axes[0]
    ax.plot(time, z_ndi, "g-.", linewidth=3)
    ax.plot(time, z_mpc, "b--", linewidth=3)
    ax.plot(time, z_opt, "k-", linewidth=3)
    ax.plot(time, zd, "r:", linewidth=2)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$z$, m", fontsize=15)
    ax.set_ylim([-12, -8])

    ax = axes[1]
    ax.plot(time, VT_ndi, "g-.", linewidth=3)
    ax.plot(time, VT_mpc, "b--", linewidth=3)
    ax.plot(time, VT_opt, "k-", linewidth=3)
    ax.plot(time, VTd, "r:", linewidth=2)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=15)

    ax = axes[2]
    ax.plot(time, np.rad2deg(theta_ndi), "g-.", linewidth=3, label="NDI")
    ax.plot(time, np.rad2deg(theta_mpc), "b--", linewidth=3, label="MPC-NDI")
    ax.plot(time, np.rad2deg(theta_opt), "k-", linewidth=3, label="Opt-NDI")
    ax.plot(time, np.rad2deg(theta_trim), "r:", linewidth=2, label="Trim")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)

    #     ax = axes[1, 1]
    #     l1 = ax.plot(time, np.rad2deg(q_ndi), "k--",linewidth=3)
    #     l2 = ax.plot(time, np.rad2deg(q_mpc), "b-",linewidth=3)
    #     l3 = ax.plot(time, np.rad2deg(qd), "r:",linewidth=3)
    #     ax.set_xlim(time[0], time[-1])
    #     ax.set_xlabel("Time, sec",linewidth=3)
    #     ax.set_ylabel(r"$q$, deg/s", fontsize=13)

    ax.legend()
    # fig.legend(
    #     [l3, l1, l2, l4],
    #     labels=["Opt-NDI", "MPC-NDI", "NDI", "Trim"],
    #     loc="lower center",
    #     bbox_to_anchor=(0.55, 0),
    #     fontsize=15,
    #     ncol=4,
    # )
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 2 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    # fig.suptitle("Control input trajectories",linewidth=3)

    ax = axes[0, 0]
    ax.plot(time, rotors_ndi[:, 0], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 0], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 0], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 1", fontsize=13)

    ax = axes[1, 0]
    ax.plot(time, rotors_ndi[:, 1], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 1], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 1], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 2", fontsize=13)
    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(time, rotors_ndi[:, 2], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 2], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 2], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 3", fontsize=13)

    ax = axes[1, 1]
    ax.plot(time, rotors_ndi[:, 3], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 3], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 3], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 4", fontsize=13)
    ax.set_xlabel("Time, sec")

    ax = axes[0, 2]
    ax.plot(time, rotors_ndi[:, 4], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 4], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 4], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 5", fontsize=13)

    ax = axes[1, 2]
    ax.plot(time, rotors_ndi[:, 5], "g-.", linewidth=3)
    ax.plot(time, rotors_mpc[:, 5], "b--", linewidth=2)
    ax.plot(time, rotors_opt[:, 5], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 6", fontsize=13)
    ax.set_xlabel("Time, sec")

    ax = axes[0, 3]
    ax.plot(time, pushers_ndi[:, 0], "g-.", linewidth=3)
    ax.plot(time, pushers_mpc[:, 0], "b--", linewidth=2)
    ax.plot(time, pushers_opt[:, 0], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Pusher 1", fontsize=13)

    ax = axes[1, 3]
    g2 = ax.plot(time, pushers_ndi[:, 1], "g-.", linewidth=3)
    g3 = ax.plot(time, pushers_mpc[:, 1], "b--", linewidth=2)
    g1 = ax.plot(time, pushers_opt[:, 1], "k-", linewidth=3)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Pusher 2", fontsize=13)

    fig.subplots_adjust(left=0.05, right=0.99, wspace=0.3)

    fig.legend(
        [g1, g3, g2],
        labels=["Opt-NDI", "MPC-NDI", "NDI"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=13,
        ncol=3,
    )

    fig.tight_layout()

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
    #         fontsize=15,
    #     )

    #     fig.tight_layout(h_pad=0.2)

    # """ Figure 4 - Forces of NDI """
    # fig, axes = plt.subplots(2, 1)

    # """ Row 1 - Pusher forces """
    # ax = axes[0]
    # ax.plot(time, Fp_ndi, "k-",linewidth=3)
    # ax.plot(time, Fpd_ndi, "--r",linewidth=3)
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec",linewidth=3)
    # ax.set_ylabel(r"$F_p$, N",linewidth=3)

    # """ Row 2 - Pitch angle """
    # ax = axes[1]
    # l1 = ax.plot(time, theta_ndi, "k-",linewidth=3)
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

    plt.show()


def total_cost():
    Vxd = Vd[:, 0]
    Vzd = Vd[:, 1]

    Vx_ndi = V_ndi[:, 0]
    Vz_ndi = V_ndi[:, 2]
    Vx_mpc = V_mpc[:, 0]
    Vz_mpc = V_mpc[:, 2]

    cost_ndi = 0
    cost_mpc = 0
    Q = 10 * np.diag((1, 1, 1))

    for k in range(np.size(time)):
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
    cost_ndi = 0
    cost_mpc = 0
    rcost_ndi = 0
    rcost_mpc = 0

    A = np.diag((1, 1, 1, 1, 1, 1, 1, 1))
    B = np.diag((1, 1, 1, 1, 1, 1))
    # A = np.diag((1,1,1,1,1,1,1,1))

    for k in range(np.size(time)):
        r_ndi = rotors_ndi[k]
        r_mpc = rotors_mpc[k]
        rcost_ndi = rcost_ndi + r_ndi.T @ B @ r_ndi
        rcost_mpc = rcost_mpc + r_mpc.T @ B @ r_mpc

    #     for k in range(np.size(time)):
    #         ctrls_ndi = np.vstack((rotors_ndi[k], pushers_ndi[k]))
    #         ctrls_mpc = np.vstack((rotors_mpc[k], pushers_mpc[k]))
    #         cost_ndi = cost_ndi + ctrls_ndi.T @ A @ ctrls_ndi
    #         cost_mpc = cost_mpc + ctrls_mpc.T @ A @ ctrls_mpc

    # return cost_ndi, cost_mpc
    return rcost_ndi, rcost_mpc


if __name__ == "__main__":
    # cost_ndi, cost_mpc = total_cost()
    # ctrlcost_ndi, ctrlcost_mpc = rotor_cost()
    # print(cost_ndi)
    # print(cost_mpc)
    # print(ctrlcost_ndi)
    # print(ctrlcost_mpc)
    plot()
