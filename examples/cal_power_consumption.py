from typing import ValuesView
import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
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
        "th_r": 130 * data["ctrls"][:, 0:6],
        "th_p": 70 * data["ctrls"][:, 6:8],
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

def induced_velocity(Th, V, alpha, rho, A):
    if Th <= 0 :
        return 0.0

    v_hover = np.sqrt(Th / (2.0 * rho * A))

    def f(v):
        term1 = V * np.cos(alpha)
        term2 = V * np.sin(alpha) + v
        return 2.0 * rho * A * v * np.sqrt(term1**2 + term2**2) - Th

    v_min = 1e-6
    v_max = max(5 * v_hover, 1)
    # v_max = max(10 * v_hover, 5)
    v_induced = brentq(f, v_min, v_max)
    
    return v_induced


def rotor_power_consumption(data):
    rho = 1.2241 # air density at altitude 10m
    A_rotor = np.pi * (0.762/2)**2
    A_pusher = np.pi * (0.525/2)**2

    P_rotors = np.zeros((6, 1))
    for i in range(6):
        for k in range(np.size(data["time"])):
            th = data["th_r"][k, i]
            VT = data["VT"][k]
            alp = data["theta"][k]
            vi = induced_velocity(th, VT, alp, rho, A_rotor)
            P_rotors[i] += th * vi

    P = np.sum(P_rotors)
    return P


if __name__ == "__main__":
    opt_switch = fym.load("data/data_opt_switch.h5")["env"]
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

    """ Cost calculation """
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

    """ Rotor power consumption """
    rho = 1.2241 # air density at altitude 10m
    A_rotor = np.pi * (0.762/2)**2
 
    P_opt_fw = rotor_power_consumption(data_opt_fw)
    P_mpc_fw = rotor_power_consumption(data_mpc_fw)
    P_ndi_fw = rotor_power_consumption(data_ndi_fw)

    P_opt_bw = rotor_power_consumption(data_opt_bw)
    P_mpc_bw = rotor_power_consumption(data_mpc_bw)
    P_ndi_bw = rotor_power_consumption(data_ndi_bw)

    print(P_opt_fw, P_opt_bw)
    print(P_mpc_fw, P_mpc_bw)
    print(P_ndi_fw, P_ndi_bw)



