import argparse

import casadi as ca
import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle

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


class MyEnv(fym.BaseEnv):
    VT_cruise = 45
    h = 10
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -h)),
                "vel": np.vstack((44.97696983, 0, 1.43950854)),
                "quat": np.vstack((0.99987205, 0, 0.01599659, 0)),
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
        self.Q_HV = np.diag([0, 0, 200, 10, 10, 20, 100, 200, 100, 0, 0, 0])
        self.R_HV = 100 * np.diag([1, 1, 1, 1, 1, 1])

        self.controller_trst = ftc.make("NMPC-DI", self)
        self.controller_lqr = ftc.make("FWHV", self)

    def step(self, action):
        env_info, done = self.update(action=action)
        obs = self.observation()

        return obs, done, env_info

    def step2(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        obs = (pos[2], vel[0], vel[2], ang[1], omega[1])  # Current state
        return obs

    def get_ref(self, t):
        _, veld, angd, _ = self.x_trims_HV

        thetad = angd[1]
        xd = 0
        zd = -self.h
        mode = "HV"

        return xd, zd, veld, thetad, mode

    def set_dot(self, t, action):
        pos, vel, quat, omega = self.plant.observe_list()
        # VT = np.linalg.norm(vel)
        # if VT < self.VT_cruise - 2:
        #     ctrls0, controller_info = self.controller_trst.get_control(t, self, action)

        # else:
        #     ctrls0, controller_info = self.controller_fw.get_control(t, self)

        # if action == ca.DM.zeros((3, 1)):
        if np.array_equal(action.full(), np.zeros((3, 1))):
            ctrls0, controller_info = self.controller_lqr.get_control(t, self)
        else:
            ctrls0, controller_info = self.controller_trst.get_control(t, self, action)
        ctrls = self.plant.saturate(ctrls0)
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
            "Fr": self.plant.B_VTOL(ctrls[:6], omega)[2],
            "Fp": self.plant.B_Pusher(ctrls[6:8])[0],
        }

        return env_info


def run():
    env = MyEnv()
    agent = ftc.make("MPC-back", env)
    flogger = fym.Logger("data_mpc_backward.h5")

    env.reset()
    try:
        while True:
            env.render()

            t = env.clock.get()
            action, agent_info = agent.get_action()
            _, vel, _, _ = env.plant.observe_list()
            VT = np.linalg.norm(vel)
            # if t < 30:
            if VT > 2:
                obs, done, env_info = env.step(action=action)
                agent.solve_mpc(obs)
            else:
                _, done, env_info = env.step(action=ca.DM.zeros((3, 1)))

            flogger.record(env=env_info, agent=agent_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data_mpc_backward.h5")["env"]
    agent_data = fym.load("data_mpc_backward.h5")["agent"]
    t_mpc = len(agent_data["Xd"][:, 0])
    # t_mpc = 1000

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.set_ylabel(r"$x$, m", fontsize=15)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "b-")
    ax.set_ylabel(r"$y$, m", fontsize=15)
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    # ax.plot(tspan, opt_traj["X"][0, :], "r--")
    ax.plot(data["t"][:t_mpc], agent_data["Xd"][:t_mpc, 0].squeeze(-1), "r--")
    ax.plot(data["t"][t_mpc:], data["posd"][t_mpc:, 0].squeeze(-1), "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.set_ylabel(r"$z$, m", fontsize=15)
    ax.set_ylim([-12, -8])

    ax.set_xlabel("Time, sec", fontsize=15)

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data["t"][:t_mpc], agent_data["Xd"][:t_mpc, 1].squeeze(-1), "r--")
    ax.plot(data["t"][t_mpc:], data["veld"][t_mpc:, 0], "r--")
    ax.set_ylabel(r"$v_x$, m/s", fontsize=15)

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "b-")
    ax.plot(data["t"], data["veld"][:, 1], "r--")
    ax.set_ylabel(r"$v_y$, m/s", fontsize=15)
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(data["t"][:t_mpc], agent_data["Xd"][:t_mpc, 2].squeeze(-1), "r--")
    ax.plot(data["t"][t_mpc:], data["veld"][t_mpc:, 2], "r--")
    ax.set_ylabel(r"$v_z$, m/s", fontsize=15)
    ax.set_ylim([-10, 10])

    ax.set_xlabel("Time, sec", fontsize=15)

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg", fontsize=15)
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg", fontsize=15)

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg", fontsize=15)
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec", fontsize=15)

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(
        data["t"],
        np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)),
        "b-",
        label="Response",
    )
    ax.plot(
        data["t"],
        np.rad2deg(data["omegad"][:, 0].squeeze(-1)),
        "r--",
        label="Command",
    )
    ax.set_ylabel(r"$p$, deg/s", fontsize=15)
    ax.set_ylim([-1, 1])
    ax.legend()

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s", fontsize=15)

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$r$, deg/s", fontsize=15)
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec", fontsize=15)

    fig.tight_layout()

    # """ Figure 2 - Control inputs """
    # fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    # ax = axes[0, 0]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b-")
    # ax.set_ylabel("Rotor 1", fontsize=15)
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])

    # ax = axes[1, 0]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b-")
    # ax.set_ylabel("Rotor 2", fontsize=15)
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_xlabel("Time, sec", fontsize=15)

    # ax = axes[0, 1]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b-")
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_ylabel("Rotor 3", fontsize=15)

    # ax = axes[1, 1]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b-")
    # ax.set_ylabel("Rotor 4", fontsize=15)
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_xlabel("Time, sec", fontsize=15)

    # ax = axes[0, 2]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b-")
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_ylabel("Rotor 5", fontsize=15)

    # ax = axes[1, 2]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b-")
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_ylabel("Rotor 6", fontsize=15)
    # ax.set_xlabel("Time, sec", fontsize=15)

    # ax = axes[0, 3]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    # ax.set_ylabel("Pusher 1", fontsize=15)
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_xlim(data["t"][0], data["t"][-1])

    # ax = axes[1, 3]
    # ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    # ax.set_xlim(data["t"][0], data["t"][-1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_ylabel("Pusher 2", fontsize=15)
    # ax.set_xlabel("Time, sec", fontsize=15)

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.3)
    # fig.align_ylabels(axes)

    # #     """ Figure 3 - Pusher input """
    #     fig, axes = plt.subplots(2, 1, sharex=True)

    #     ax = axes[0]
    #     ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    #     ax.set_ylabel("Pusher 1")
    #     ax.set_xlim(data["t"][0], data["t"][-1])

    #     ax = axes[1]
    #     ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    #     ax.set_ylabel("Pusher 2")
    #     ax.set_xlabel("Time, sec")

    #     plt.tight_layout()
    #     fig.align_ylabels(axes)

    # """ Figure 5 - Thrust """
    # fig, axes = plt.subplots(2, 1, sharex=True)

    # ax = axes[0]
    # ax.plot(data["t"], data["Frd"], "r--")
    # ax.plot(data["t"], data["Fr"].squeeze(-1), "b-")
    # ax.set_ylabel(r"$F_{rotors}$, N")
    # ax.set_xlim(data["t"][0], data["t"][-1])

    # ax = axes[1]
    # ax.plot(data["t"], data["Fpd"], "r--")
    # ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    # ax.set_ylabel(r"$F_{pushers}$, N")
    # ax.set_xlabel("Time, sec")

    # plt.tight_layout()
    # fig.align_ylabels(axes)

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
