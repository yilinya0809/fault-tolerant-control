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
FTC = np.load("data/corr_forward.npz")
VT_ftc = FTC["VT_corr"]
theta_ftc = np.rad2deg(FTC["theta_corr"])
success_ftc = FTC["success"]

BTC = np.load("data/corr_backward.npz")
VT_btc = BTC["VT_corr"]
theta_btc = np.rad2deg(BTC["theta_corr"])
success_btc = BTC["success"]

""" Control results """
opt_switch = fym.load("data/data_opt_switch.h5")["env"]
fw_opt = fym.load("data/data_opt_forward.h5")["env"]
fw_mpc = fym.load("data/data_mpc_forward.h5")["env"]
fw_mpc_agent = fym.load("data/data_mpc_forward.h5")["agent"]
fw_ndi = fym.load("data/data_ndi_forward.h5")["env"]
bw_opt = fym.load("data/data_opt_backward.h5")["env"]
bw_mpc = fym.load("data/data_mpc_backward.h5")["env"]
bw_mpc_agent = fym.load("data/data_mpc_backward.h5")["agent"]
bw_ndi = fym.load("data/data_ndi_backward.h5")["env"]

def refine(data, agent=None):
    result = {
        "time": data["t"],
        "x": data["plant"]["pos"][:, 0],
        "z": data["plant"]["pos"][:, 2],
        "V": data["plant"]["vel"].squeeze(-1),
        "VT": np.linalg.norm(data["plant"]["vel"], axis=1),
        "theta": data["ang"][:, 1],
        "q": data["plant"]["omega"][:, 1],
        "Fr": data["Fr"],
        "Fp": data["Fp"],
        "rotors": data["ctrls"][:, 0:6],
        "pushers": data["ctrls"][:, 6:8],
    }

    if agent is not None:
        result.update({
            "zd": agent["Xd"][:, 0],
            "Vd": agent["Xd"][:, 1],
            "VTd": np.linalg.norm(agent["Xd"][:, 1], axis=1),
            "qd": agent["qd"],
            "Fr_trim": agent["Ud"][:, 0],
            "Fp_trim": agent["Ud"][:, 1],
            "theta_trim": agent["Ud"][:, 2],
        })

    return result

data_opt_full = refine(opt_switch)
data_opt_fw = refine(fw_opt)
data_opt_bw = refine(bw_opt)
data_mpc_fw = refine(fw_mpc, fw_mpc_agent)
data_mpc_bw = refine(bw_mpc, bw_mpc_agent)
data_ndi_fw = refine(fw_ndi)
data_ndi_bw = refine(bw_ndi)

t_ftc = data_opt_fw["time"]
t_btc = data_opt_bw["time"]

def plot():
    """Figure 1 - FTC states """
    fig, axes = plt.subplots(3, 1, figsize=(10,8))

    ax = axes[0]
    ax.plot(t_ftc, data_ndi_fw["z"], "g-.", linewidth=3, label='NDI')
    ax.plot(t_ftc, data_mpc_fw["z"], "b--", linewidth=3, label='MPC')
    ax.plot(t_ftc, data_opt_fw["z"], "r-", linewidth=3, label='Corr-Opt')
    ax.plot(t_ftc, data_mpc_fw["zd"], "k:", linewidth=3, label='Trim')
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.set_ylim([-12, -8])
    fig.tight_layout()

    ax = axes[1]
    ax.plot(t_ftc, data_ndi_fw["VT"], "g-.", linewidth=3, label='NDI')
    ax.plot(t_ftc, data_mpc_fw["VT"], "b--", linewidth=3, label='MPC')
    ax.plot(t_ftc, data_opt_fw["VT"], "r-", linewidth=3, label='Corr-Opt')
    ax.plot(t_ftc, data_mpc_fw["VTd"], "k:", linewidth=3, label='Trim')
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    fig.tight_layout()

    ax = axes[2]
    ax.plot(t_ftc, np.rad2deg(data_ndi_fw["theta"]), "g-.", linewidth=3, label='NDI')
    ax.plot(t_ftc, np.rad2deg(data_mpc_fw["theta"]), "b--", linewidth=3, label='MPC')
    ax.plot(t_ftc, np.rad2deg(data_opt_fw["theta"]), "r-", linewidth=3, label='Corr-Opt')
    ax.plot(t_ftc, np.rad2deg(data_mpc_fw["theta_trim"]), "k:", linewidth=3, label='Trim')
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)

    ax.legend(loc='lower right', fontsize=15)
    fig.tight_layout()
    # fig.subplot_adjust(right=0.85)

    """ Figure 2 - Control Inputs """
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 0], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 0], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 1", fontsize=14)
    # ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), fontsize=12)

    ax = axes[1, 0]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 1], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 1], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 2", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 1]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 2], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 2], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 2], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 3", fontsize=14)

    ax = axes[1, 1]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 3], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 3], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 3], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 4", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 2]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 4], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 4], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 4], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 5", fontsize=14)

    ax = axes[1, 2]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["rotors"][:, 5], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["rotors"][:, 5], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["rotors"][:, 5], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Rotor 6", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

    ax = axes[0, 3]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["pushers"][:, 0], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["pushers"][:, 0], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["pushers"][:, 0], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 1", fontsize=14)

    ax = axes[1, 3]
    ax.plot(t_ftc, np.ones((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, np.zeros((len(t_ftc), 1)), "k:")
    ax.plot(t_ftc, data_ndi_fw["pushers"][:, 1], "g-.", linewidth=3, label="NDI")
    ax.plot(t_ftc, data_mpc_fw["pushers"][:, 1], "b:", linewidth=2, label="MPC")
    ax.plot(t_ftc, data_opt_fw["pushers"][:, 1], "r-", linewidth=3, label="Corr-Opt")
    ax.set_xlim(t_ftc[0], t_ftc[-1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Pusher 2", fontsize=14)
    ax.set_xlabel("Time, sec", fontsize=14)

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
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    VT, theta = np.meshgrid(VT_ftc, theta_ftc)
    ax.scatter(VT, theta, s=success_ftc.T, c="b")

    ax.plot(data_ndi_fw["VT"], np.rad2deg(data_ndi_fw["theta"]), "g-.", linewidth=5, label="NDI")
    ax.plot(data_mpc_fw["VT"], np.rad2deg(data_mpc_fw["theta"]), "b--", linewidth=5, label="MPC")
    ax.plot(data_opt_fw["VT"], np.rad2deg(data_opt_fw["theta"]), "r-", linewidth=5, label="Corr-Opt")
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
    # rcost, cost = rotor_cost()
    # print(rcost)
    # print(cost)
    plot()
