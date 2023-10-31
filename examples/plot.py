import argparse

import fym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle
from numpy import cos, sin

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")

data_ndi = fym.load("data_NDI.h5")["env"]
data_mpc = fym.load("data_MPC.h5")["env"]
agent = fym.load("data_MPC.h5")["agent"]

time = data_ndi["t"]
zd = agent["Xd"][:, 0]
Vd = agent["Xd"][:, 1:]
VTd = np.linalg.norm(Vd, axis=1)
qd = agent["qd"]

Fr_trim = agent["Ud"][:, 0]
Fp_trim = agent["Ud"][:, 1]
theta_trim = agent["Ud"][:, 2]

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


def plot():
    """ Figure 1 - States """
    fig, axes = plt.subplots(2, 2)
    # fig.suptitle("State trajectories")
    
    """ Row 1 - z, VT """
    ax = axes[0, 0]
    ax.plot(time, z_ndi, "k--")
    ax.plot(time, z_mpc, "b-")
    ax.plot(time, zd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$z$, m", fontsize=13)
    ax.set_ylim([-15, -5])

    ax = axes[0, 1]
    ax.plot(time, VT_ndi, "k--")
    ax.plot(time, VT_mpc, "b-")
    ax.plot(time, VTd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=13)

    """ Row 2 - Pitch angle, rate """
    ax = axes[1, 0]
    ax.plot(time, np.rad2deg(theta_ndi), "k--")
    ax.plot(time, np.rad2deg(theta_mpc), "b-")
    ax.plot(time, np.rad2deg(theta_trim), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$\theta$, deg", fontsize=13)

    ax = axes[1, 1]
    l1 = ax.plot(time, np.rad2deg(q_ndi), "k--")
    l2 = ax.plot(time, np.rad2deg(q_mpc), "b-")
    l3 = ax.plot(time, np.rad2deg(qd), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$q$, deg/s", fontsize=13)
    
    fig.legend([l1, l2, l3], 
               labels=["NDI", "NMPC-DI", "Trim"],
               loc="lower center",
               bbox_to_anchor=(0.55, 0),
               fontsize=15,
               ncol=3,
               )
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)


    """ Figure 2 - Control Inputs """
    fig, axes = plt.subplots(4, 2)
    # fig.suptitle("Control input trajectories")

    ax = axes[0, 0]
    ax.plot(time, rotors_ndi[:, 0], "k--")
    ax.plot(time, rotors_mpc[:, 0], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_1$", fontsize=13)

    ax = axes[0, 1]
    ax.plot(time, rotors_ndi[:, 1], "k--")
    ax.plot(time, rotors_mpc[:, 1], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_2$", fontsize=13)
    
    ax = axes[1, 0]
    ax.plot(time, rotors_ndi[:, 2], "k--")
    ax.plot(time, rotors_mpc[:, 2], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_3$", fontsize=13)

    ax = axes[1, 1]
    ax.plot(time, rotors_ndi[:, 3], "k--")
    ax.plot(time, rotors_mpc[:, 3], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_4$", fontsize=13)

    ax = axes[2, 0]
    ax.plot(time, rotors_ndi[:, 4], "k--")
    ax.plot(time, rotors_mpc[:, 4], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_5$", fontsize=13)

    ax = axes[2, 1]
    ax.plot(time, rotors_ndi[:, 5], "k--")
    ax.plot(time, rotors_mpc[:, 5], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$r_6$", fontsize=13)

    ax = axes[3, 0]
    ax.plot(time, pushers_ndi[:, 0], "k--")
    ax.plot(time, pushers_mpc[:, 0], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$p_1$", fontsize=13)

    ax = axes[3, 1]
    g1 = ax.plot(time, pushers_ndi[:, 1], "k--")
    g2 = ax.plot(time, pushers_mpc[:, 1], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$p_2$", fontsize=13)

    fig.legend([g1, g2], 
               labels=["NDI", "NMPC-DI"],
               loc="lower center",
               bbox_to_anchor=(0.5, 0),
               fontsize=13,
               ncol=2,
               )

    fig.tight_layout()

    """ Figure 3 - Forces of NMPC """
    fig, axes = plt.subplots(3, 1)
     
    """ Row 1 - Rotor forces """
    ax = axes[0]
    ax.plot(time, -Fr_mpc, "b-")
    ax.plot(time, -Frd_mpc, "--r")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$F_r$, N", fontsize=13)


    """ Row 2 - Pusher forces """
    ax = axes[1]
    ax.plot(time, Fp_mpc, "b-")
    ax.plot(time, Fpd_mpc, "--r")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$F_p$, N", fontsize=13)


    """ Row 3 - Pitch angle """
    ax = axes[2]
    l1 = ax.plot(time, np.rad2deg(theta_mpc), "b-")
    l2 = ax.plot(time, np.rad2deg(thetad_mpc), "--r")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$\theta$, deg", fontsize=13)
    ax.set_xlabel("Time, sec")

    fig.legend([l1, l2], 
               labels=["NMPC-DI", "Commands"],
               loc="lower center",
               bbox_to_anchor=(0.55, 0),
               ncol=2,
               fontsize=15,
               )

    fig.tight_layout(h_pad=0.2)

    
    # """ Figure 4 - Forces of NDI """
    # fig, axes = plt.subplots(2, 1)

    # """ Row 1 - Pusher forces """
    # ax = axes[0]
    # ax.plot(time, Fp_ndi, "k-")
    # ax.plot(time, Fpd_ndi, "--r")
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec")
    # ax.set_ylabel(r"$F_p$, N")


    # """ Row 2 - Pitch angle """
    # ax = axes[1]
    # l1 = ax.plot(time, theta_ndi, "k-")
    # l2 = ax.plot(time, thetad_ndi, "--r")
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xlabel("Time, sec")
    # ax.set_ylabel(r"$\theta$, N")

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

        err_ndi = np.vstack((
            z_ndi[k] - zd[k], 
            Vx_ndi - Vxd[k],
            Vz_ndi - Vzd[k],
        ))

        err_mpc = np.vstack((
            z_mpc[k] - zd[k], 
            Vx_mpc - Vxd[k],
            Vz_mpc - Vzd[k],
        ))

        cost_ndi = cost_ndi + err_ndi.T @ Q @ err_ndi
        cost_mpc = cost_mpc + err_mpc.T @ Q @ err_mpc

    return cost_ndi, cost_mpc


if __name__ == "__main__":
    cost_ndi, cost_mpc = total_cost()
    print(cost_ndi)
    print(cost_mpc)
    plot()

