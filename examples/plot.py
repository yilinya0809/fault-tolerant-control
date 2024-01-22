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

def plot1():

    data = data_mpc
    agent_data = agent

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
    ax.plot(data["t"], agent_data["Xd"][:, 0].squeeze(-1), "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-15, -5])

    ax.set_xlabel("Time, sec")
    ax.legend(["Commands from NMPC", "States"], 
              # loc="upper right",
              loc="lower right",
              bbox_to_anchor=(3.2, -0.5),
              fontsize=12,
              ncol=2,
              )
    # fig.legend([p1, p2], 
    #            labels=["Commands from NMPC", "States"],
    #            loc="lower center",
    #            bbox_to_anchor=(0.5, 0),
    #            fontsize=12,
    #            ncol=2,
    #            )
    

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
    p1 = ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "b-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    p2 = ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "b-")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    # fig.tight_layout()
    # fig.legend([p1, p2], 
    #            labels=["Commands from NMPC", "States"],
    #            loc="lower center",
    #            bbox_to_anchor=(0.5, 0),
    #            fontsize=12,
    #            ncol=2,
    #            )
    
    fig.tight_layout()
    fig.align_ylabels(axes)
    fig.subplots_adjust(left = 0.05, right = 0.99, wspace=0.3)


    """ Figure 5 - Thrust """
    fig, axes = plt.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], -data["Frd"], "r--")
    ax.plot(data["t"], -data["Fr"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{rotors}$, N", fontsize=13)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    l1 = ax.plot(data["t"], data["Fpd"], "r--")
    l2 = ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{pushers}$, N", fontsize=13)
    ax.set_xlabel("Time, sec")

    fig.legend([l1, l2], 
               labels=["Commands from NMPC", "Results"],
               loc="lower center",
               bbox_to_anchor=(0.55, 0),
               fontsize=12,
               ncol=2,
               )

    plt.tight_layout()
    fig.align_ylabels(axes)


    plt.show()


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
               labels=["ndi", "nmpc-di", "trim"],
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
    ax.set_ylabel("Rotor 1", fontsize=13)

    ax = axes[0, 1]
    ax.plot(time, rotors_ndi[:, 1], "k--")
    ax.plot(time, rotors_mpc[:, 1], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 2", fontsize=13)
    
    ax = axes[1, 0]
    ax.plot(time, rotors_ndi[:, 2], "k--")
    ax.plot(time, rotors_mpc[:, 2], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 3", fontsize=13)

    ax = axes[1, 1]
    ax.plot(time, rotors_ndi[:, 3], "k--")
    ax.plot(time, rotors_mpc[:, 3], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 4" , fontsize=13)

    ax = axes[2, 0]
    ax.plot(time, rotors_ndi[:, 4], "k--")
    ax.plot(time, rotors_mpc[:, 4], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 5", fontsize=13)

    ax = axes[2, 1]
    ax.plot(time, rotors_ndi[:, 5], "k--")
    ax.plot(time, rotors_mpc[:, 5], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Rotor 6", fontsize=13)

    ax = axes[3, 0]
    ax.plot(time, pushers_ndi[:, 0], "k--")
    ax.plot(time, pushers_mpc[:, 0], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Pusher 1", fontsize=13)

    ax = axes[3, 1]
    g1 = ax.plot(time, pushers_ndi[:, 1], "k--")
    g2 = ax.plot(time, pushers_mpc[:, 1], "b-")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Pusher 2", fontsize=13)

    fig.subplots_adjust(left = 0.05, right = 0.99, wspace=0.3)

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
    ax.set_ylabel(r"$F_{rotors}$, N", fontsize=13)


    """ Row 2 - Pusher forces """
    ax = axes[1]
    ax.plot(time, Fp_mpc, "b-")
    ax.plot(time, Fpd_mpc, "--r")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$F_{pushers}$, N", fontsize=13)


    """ Row 3 - Pitch angle """
    ax = axes[2]
    l1 = ax.plot(time, np.rad2deg(theta_mpc), "b-")
    l2 = ax.plot(time, np.rad2deg(thetad_mpc), "--r")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$\theta$, deg", fontsize=13)
    ax.set_xlabel("Time, sec")

    fig.legend([l1, l2], 
               labels=["NMPC-DI Results", "NMPC Solutions"],
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

def rotor_cost():
    cost_ndi = 0
    cost_mpc = 0
    rcost_ndi = 0
    rcost_mpc = 0

    A = np.diag((1,1,1,1,1,1,1,1))
    B = np.diag((1,1,1,1,1,1))
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
    cost_ndi, cost_mpc = total_cost()
    ctrlcost_ndi, ctrlcost_mpc = rotor_cost()
    print(cost_ndi)
    print(cost_mpc)
    print(ctrlcost_ndi)
    print(ctrlcost_mpc)
    plot()

