import fym
import matplotlib.pyplot as plt
import numpy as np

from Corridor.poly_corr import boundary, poly, weighted_poly

Trst_corr = np.load("corr.npz")
Vel_corr = Trst_corr["VT_corr"]
acc_corr = Trst_corr["acc_corr"]
theta_corr = Trst_corr["theta_corr"]


def plot():
    corr = fym.load("data_corr_archive.h5")["env"]
    data = fym.load("data_geso.h5")["env"]
    agent = fym.load("data_geso.h5")["agent"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(corr["t"], corr["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 500])

    ax = axes[1, 0]
    ax.plot(corr["t"], corr["plant"]["pos"][:, 1].squeeze(-1), "k-")
    # ax.plot(corr["t"], corr["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.legend(["Response", "Command"], loc="upper right")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(corr["t"], corr["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(corr["t"], corr["stated"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-60, -40])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(corr["t"], corr["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(corr["t"], corr["stated"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_ylim([0, 50])

    ax = axes[1, 1]
    ax.plot(corr["t"], corr["plant"]["vel"][:, 1].squeeze(-1), "k-")
    # ax.plot(corr["t"], corr["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(corr["t"], corr["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(corr["t"], corr["stated"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(corr["t"], np.rad2deg(corr["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(corr["t"], np.rad2deg(corr["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(corr["t"], np.rad2deg(corr["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(corr["t"], np.rad2deg(corr["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 2]
    ax.plot(
        corr["t"], np.rad2deg(corr["ang"][:, 2].squeeze(-1)), "k-", label="response"
    )
    ax.plot(
        corr["t"], np.rad2deg(corr["angd"][:, 2].squeeze(-1)), "r--", label="command"
    )
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(corr["t"], np.rad2deg(corr["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(corr["t"], np.rad2deg(corr["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 3]
    ax.plot(corr["t"], np.rad2deg(corr["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=15,
        ncol=2,
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.99, wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Generalized forces """
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Generalized forces: Forces """
    ax = axes[0, 0]
    ax.plot(corr["t"], corr["FM"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_x$")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([-1, 1]

    ax = axes[1, 0]
    ax.plot(corr["t"], corr["FM"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_y$")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(corr["t"], corr["FM"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Generalized forces: Moments """
    ax = axes[0, 1]
    ax.plot(corr["t"], corr["FM"][:, 3].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_x$")
    ax.set_ylim([-1, 1])

    ax = axes[1, 1]
    ax.plot(corr["t"], corr["FM"][:, 4].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_y$")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(corr["t"], corr["FM"][:, 5].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_z$")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Thrusts """
    fig, axs = plt.subplots(2, 4, sharex=True)

    ax = axs[0, 0]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 0], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b--")
    ax.set_ylabel("Rotor 1")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 0]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 1], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b--")
    ax.set_ylabel("Rotor 2")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 1]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 2], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b--")
    ax.set_ylabel("Rotor 3")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 1]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 3], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b--")
    ax.set_ylabel("Rotor 4")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 2]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 4], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b--")
    ax.set_ylabel("Rotor 5")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 2]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 5], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b--")
    ax.set_ylabel("Rotor 6")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    # ax.set_ylim([0, 1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 3]
    ax.plot(corr["t"], corr["ctrls"].squeeze(-1)[:, 6], "k-", linewidth=2)
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b--")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[1, 3]
    ax.plot(
        corr["t"], corr["ctrls"].squeeze(-1)[:, 7], "k-", linewidth=2, label="NMPC-Corr"
    )
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b--", label="NMPC-GESO")
    ax.set_ylabel("Pusher 2")
    ax.set_xlim(corr["t"][0], corr["t"][-1])
    ax.set_ylim([-0.2, 1.2])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=15,
        ncol=2,
    )

    fig.suptitle("Rotational Thrusts", y=0.85)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.8, wspace=0.25, hspace=0.2)
    fig.align_ylabels(axs)

    """ Figure 4 - Longitudinal states """
    time = data["t"]
    zd = agent["Xd"][:, 0]
    Vd = agent["Xd"][:, 1:]
    VTd = np.linalg.norm(Vd, axis=1)
    qd = agent["qd"]

    z_geso = data["plant"]["pos"][:, 2]
    V_geso = data["plant"]["vel"].squeeze(-1)
    VT_geso = np.linalg.norm(V_geso, axis=1)
    theta_geso = data["ang"][:, 1]
    q_geso = data["plant"]["omega"][:, 1]

    z_corr = corr["plant"]["pos"][:, 2]
    V_corr = corr["plant"]["vel"].squeeze(-1)
    VT_corr = np.linalg.norm(V_corr, axis=1)
    theta_corr = corr["ang"][:, 1]
    q_corr = corr["plant"]["omega"][:, 1]

    fig, axes = plt.subplots(2, 2)
    # fig.suptitle("State trajectories")

    """ Row 1 - z, VT """
    ax = axes[0, 0]
    ax.plot(time, z_corr, "k-")
    ax.plot(time, z_geso, "b--")
    ax.plot(time, zd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$z$, m", fontsize=16)
    ax.set_ylim([-55, -45])
    # ax.set_xticks(np.arange(0, 21, 5))

    ax = axes[0, 1]
    ax.plot(time, VT_corr, "k-")
    ax.plot(time, VT_geso, "b--")
    ax.plot(time, VTd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=16)
    # ax.set_xticks(np.arange(0, 21, 5))

    """ Row 2 - Pitch angle, rate """
    ax = axes[1, 0]
    ax.plot(time, np.rad2deg(theta_corr), "k-")
    ax.plot(time, np.rad2deg(theta_geso), "b--")

    ax.plot(data["t"], np.rad2deg(agent["Ud"][:, 2].squeeze(-1)), "r:")
    # ax.plot(time, np.rad2deg(theta_trim), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec", fontsize=16)
    ax.set_ylabel(r"$\theta$, deg", fontsize=16)
    # ax.set_xticks(np.arange(0, 21, 5))

    ax = axes[1, 1]
    l1 = ax.plot(time, np.rad2deg(q_corr), "k-")
    l2 = ax.plot(time, np.rad2deg(q_geso), "b--")
    l3 = ax.plot(time, np.rad2deg(qd), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec", fontsize=16)
    ax.set_ylabel(r"$q$, deg/s", fontsize=16)
    # ax.set_xticks(np.arange(0, 21, 5))

    fig.legend(
        [l1, l2, l3],
        labels=["NMPC-Corr", "NMPC-GESO", "Cruise condition"],
        loc="lower center",
        bbox_to_anchor=(0.55, 0),
        fontsize=16,
        ncol=3,
    )
    fig.tight_layout()
    fig.align_ylabels(axes)

    """ Figure 6 - Transition Corridor """
    degree = 3
    upper_bound, lower_bound = boundary(Vel_corr)
    upper, lower, central = poly(degree, Vel_corr, upper_bound, lower_bound)

    Vel_target = Vel_corr[-1]
    weighted = weighted_poly(degree, Vel_corr, Vel_target, upper, lower)

    plt.figure(figsize=(10, 6))
    plt.plot(
        Vel_corr, upper_bound, "o", label="Upper Bound Data", color="green", alpha=0.3
    )
    plt.plot(
        Vel_corr, lower_bound, "o", label="Lower Bound Data", color="orange", alpha=0.3
    )
    plt.plot(Vel_corr, upper(Vel_corr), "g--", label="Upper Bound Polynomial")
    plt.plot(Vel_corr, lower(Vel_corr), "y--", label="Lower Bound Polynomial")
    # plt.plot(VT_corr, central(VT_corr), '-', label='Central Line')
    plt.plot(Vel_corr, weighted(Vel_corr), "r-.", label="Weighted Line")
    plt.scatter(VT_corr, np.rad2deg(theta_corr), s=4, color="k", label="NMPC-Corr")
    plt.scatter(
        VT_geso, np.rad2deg(theta_geso), s=4, marker="D", color="b", label="NMPC-GESO"
    )
    plt.xlabel("VT, m/s", fontsize=15)
    plt.ylabel("θ, deg", fontsize=15)
    plt.legend()
    plt.grid()
    plt.show()

    plt.show()


if __name__ == "__main__":
    plot()
