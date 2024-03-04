import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv1(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 1,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -50.0)),
                "vel": np.zeros((3, 1)),
                "quat": np.vstack((1, 0, 0, 0)),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62R(env_config["plant"])
        self.ang_lim = np.deg2rad(50)
        self.controller = ftc.make("NMPC-GESO", self)


    def step(self, action):
        env_info, done = self.update(action=action)
        obs = self.observation()

        return obs, done, env_info

    def observation(self):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        obs = (pos[2], vel[0], vel[2], ang[1], omega[1])  # Current state
        return obs

    def set_dot(self, t, action):
        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self, action)
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

    
class MyEnv2(MyEnv1):
    def set_dot(self, t, action):
        self.controller = ftc.make("NMPC-DI", self)

        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self, action)
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
    env1 = MyEnv1()
    agent1 = ftc.make("NMPC", env1)
    flogger1 = fym.Logger("data_geso.h5")

    env2 = MyEnv2()
    agent2 = ftc.make("NMPC", env2)
    flogger2 = fym.Logger("data_ndi.h5")

    env1.reset()
    env2.reset()
    try:
        while True:
            env1.render()
            env2.render()

            action1, agent_info1 = agent1.get_action()
            action2, agent_info2 = agent2.get_action()

            obs1, done1, env_info1 = env1.step(action=action1)
            obs2, done2, env_info2 = env2.step(action=action2)
            agent1.solve_mpc(obs1)
            agent2.solve_mpc(obs2)

            flogger1.record(env=env_info1, agent=agent_info1)
            flogger2.record(env=env_info2, agent=agent_info2)


            if done1 & done2:
                break


    finally:
        flogger1.close()
        flogger2.close()
        plot()


def plot():
    data = fym.load("data_geso.h5")["env"]
    data_ndi = fym.load("data_ndi.h5")["env"]
    agent = fym.load("data_geso.h5")["agent"]


    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "b-")
    ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$y$, m")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data["t"], agent["Xd"][:, 0].squeeze(-1), "r--")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$z$, m")
    # ax.set_ylim([-15, -5])

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data["t"], agent["Xd"][:, 1].squeeze(-1), "--r")
    ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_x$, m/s")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "b-")
    ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], agent["Xd"][:, 2].squeeze(-1), "--r")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_ylim([-20, 20])

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    # ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(agent["Ud"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")
    # ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$p$, deg/s")
    # ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    a1 = ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "b-")
    a2 = ax.plot(data["t"], np.rad2deg(data_ndi["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    a3 = ax.plot(data["t"], np.rad2deg(data["omegad"][:, 2].squeeze(-1)), "--r")
    ax.set_ylabel(r"$r$, deg/s")
    # ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    # fig.tight_layout()
    fig.legend([a1, a2], 
               labels=["NMPC-GESO", "NMPC-DI"],
               loc="lower center",
               bbox_to_anchor=(0.5, 0),
               fontsize=12,
               ncol=3,
               )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.align_ylabels(axes)


    """ Figure 2 - X, U """
    fig, axes = plt.subplots(3, 2, sharex=True)

    # z
    ax = axes[0, 0]
    ax.plot(data["t"], agent["Xd"][:, 0].squeeze(-1), "r:")
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "b-")
    # ax.plot(data_ndi["t"], data_ndi["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$z$, m", fontsize=20)

    # Vx
    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data["t"], agent["Xd"][:, 1].squeeze(-1), "r:")
    # ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_x$, m/s", fontsize=20)

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "b-")
    # ax.plot(data_ndi["t"], data_ndi["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], agent["Xd"][:, 2].squeeze(-1), "r:")
    ax.set_ylabel(r"$v_z$, m/s", fontsize=20)
    ax.set_ylim([-20, 20])
    ax.set_xlabel("Time, sec", fontsize=20)

    ax = axes[0, 1]
    ax.plot(data["t"], -data["Frd"], "r:")
    ax.plot(data["t"], -data["Fr"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{rotors}$, N", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 1]
    l1 = ax.plot(data["t"], data["Fpd"], "r:")
    l2 = ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{pushers}$, N", fontsize=20)

    ax = axes[2, 1]
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r:")
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    # ax.plot(data_ndi["t"], np.rad2deg(data_ndi["angd"][:, 1].squeeze(-1)), "k--")
    # ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_xticks(np.arange(0, 21, 5))

    fig.legend([l1, l2],
               labels=["Optimal trajectories from NMPC", "NMPC-GESO"],
               loc = "upper center",
               fontsize=20, 
               ncol = 2,
               )


    """ Figure 3 - Control inputs """
    fig, axes = plt.subplots(4, 2, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 0], "k-.")
    ax.set_ylabel("Rotor 1", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))

    ax = axes[1, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 1], "k-.")
    ax.set_ylabel("Rotor 2", fontsize=20)
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))


    ax = axes[2, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 2], "k-.")
    ax.set_ylabel("Rotor 3", fontsize=20)
    # ax.set_xlabel("Time,sec")
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    # ax.set_xticks(np.arange(0, 21, 5))


    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 3], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 3], "k-.")
    ax.set_ylabel("Rotor 4", fontsize=20)
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))


    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 4], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 4], "k-.")
    ax.set_ylabel("Rotor 5", fontsize=20)
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))



    ax = axes[2, 1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 5], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 5], "k-.")
    ax.set_ylabel("Rotor 6", fontsize=20)
    # ax.set_xlabel("Time, sec")
    ax.set_ylim([-0.1, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))


    ax = axes[3, 0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 6], "k-.")
    ax.set_ylabel("Pusher 1", fontsize=20)
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylim([-0, 1.1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_xticks(np.arange(0, 21, 5))

    ax = axes[3, 1]
    d1 = ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    d2 = ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 7], "k-.")
    ax.set_ylabel("Pusher 2", fontsize=20)
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylim([-0, 1.1])
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_xticks(np.arange(0, 21, 5))

    fig.legend([d1, d2],
               labels=["NMPC-GESO", "NMPC-DI"],
               loc = "lower center",
               bbox_to_anchor=(0.55, 0),
               fontsize=20, 
               ncol = 2,
               )



    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 3 - Pusher input """
    fig, axes = plt.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 6], "k-")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "b-")
    ax.plot(data_ndi["t"], data_ndi["ctrls"].squeeze(-1)[:, 7], "k-")
    ax.set_ylabel("Pusher 2")
    ax.set_xlabel("Time, sec")

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Figure 4 - Thrust """
    fig, axes = plt.subplots(3, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], -data["Frd"], "r--")
    ax.plot(data["t"], -data["Fr"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{rotors}$, N")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1]
    l1 = ax.plot(data["t"], data["Fpd"], "r--")
    l2 = ax.plot(data["t"], data["Fp"].squeeze(-1), "b-")
    ax.set_ylabel(r"$F_{pushers}$, N")
    ax.set_xlabel("Time, sec")

    ax = axes[2]
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data_ndi["t"], np.rad2deg(data_ndi["angd"][:, 1].squeeze(-1)), "k--")
    ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$\theta$, deg")
    ax.set_xlabel("Time, sec")


    """ Figure 5 - attitude error """
    fig, axes = plt.subplots(3, 1, sharex=True)

    ax = axes[0]
    l1 = ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 0].squeeze(-1)), "k-.")
    l2 = ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r:")
    ax.set_ylabel(r"$\phi_{e}$, deg", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_ylim([-1, 1])
    
    ax = axes[1]
    error_ndi = np.rad2deg(data_ndi["ang"][:, 1].squeeze(-1)) - np.rad2deg(data_ndi["angd"][:, 1].squeeze(-1))

    error_geso = np.rad2deg(data["ang"][:, 1].squeeze(-1)) -np.rad2deg(data["angd"][:, 1].squeeze(-1)) 
    ax.plot(data["t"], error_ndi, "k-.")
    ax.plot(data["t"], error_geso, "b-")
    ax.plot(data["t"], np.rad2deg(agent["Ud"][:, 2].squeeze(-1)), "r:")
    ax.set_ylabel(r"$\theta_e$, deg", fontsize=20)
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[2]
    l1 = ax.plot(data["t"], np.rad2deg(data_ndi["ang"][:, 2].squeeze(-1)), "k-.")
    l2 = ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "b-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r:")
    ax.set_ylabel(r"$\psi_e$, deg", fontsize=20)
    ax.set_xlabel("Time, sec", fontsize=20)
    ax.set_ylim([-1, 1])
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.set_xticks(np.arange(0, 21, 2))
    
    fig.legend([l1, l2], 
              labels=["NMPC-DI", "NMPC-GESO"],
              loc="upper center",
              # bbox_to_anchor=(0.5, 0),
              fontsize=20,
              ncol=3,
             )


    """ Figure 6 Longitudinal States """
    time = data_ndi["t"]
    zd = agent["Xd"][:, 0]
    Vd = agent["Xd"][:, 1:]
    VTd = np.linalg.norm(Vd, axis=1)
    qd = agent["qd"]

    z_ndi = data_ndi["plant"]["pos"][:, 2]
    V_ndi = data_ndi["plant"]["vel"].squeeze(-1)
    VT_ndi = np.linalg.norm(V_ndi, axis=1)
    theta_ndi = data_ndi["ang"][:, 1]
    q_ndi = data_ndi["plant"]["omega"][:, 1]
    z_geso = data["plant"]["pos"][:, 2]
    V_geso = data["plant"]["vel"].squeeze(-1) 
    VT_geso = np.linalg.norm(V_geso, axis=1)
    theta_geso = data["ang"][:, 1]
    q_geso = data["plant"]["omega"][:, 1]



    fig, axes = plt.subplots(2, 2)
    # fig.suptitle("State trajectories")
    
    """ Row 1 - z, VT """
    ax = axes[0, 0]
    ax.plot(time, z_ndi, "k--")
    ax.plot(time, z_geso, "b-")
    ax.plot(time, zd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$z$, m", fontsize=20)
    ax.set_ylim([-55, -45])
    ax.set_xticks(np.arange(0, 21, 5))

    ax = axes[0, 1]
    ax.plot(time, VT_ndi, "k--")
    ax.plot(time, VT_geso, "b-")
    ax.plot(time, VTd, "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(r"$V$, m/s", fontsize=20)
    ax.set_xticks(np.arange(0, 21, 5))

    """ Row 2 - Pitch angle, rate """
    ax = axes[1, 0]
    ax.plot(time, np.rad2deg(theta_ndi), "k--")
    ax.plot(time, np.rad2deg(theta_geso), "b-")

    ax.plot(data["t"], np.rad2deg(agent["Ud"][:, 2].squeeze(-1)), "r:")
    # ax.plot(time, np.rad2deg(theta_trim), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec", fontsize=18)
    ax.set_ylabel(r"$\theta$, deg", fontsize=20)
    ax.set_xticks(np.arange(0, 21, 5))

    ax = axes[1, 1]
    l1 = ax.plot(time, np.rad2deg(q_ndi), "k--")
    l2 = ax.plot(time, np.rad2deg(q_geso), "b-")
    l3 = ax.plot(time, np.rad2deg(qd), "r:")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Time, sec", fontsize=18)
    ax.set_ylabel(r"$q$, deg/s", fontsize=20)
    ax.set_xticks(np.arange(0, 21, 5))
    
    fig.legend([l1, l2, l3], 
               labels=["NMPC-DI", "NMPC-GESO", "Trim"],
               loc="lower center",
               bbox_to_anchor=(0.55, 0),
               fontsize=18,
               ncol=3,
               )
    fig.tight_layout()
    fig.align_ylabels(axes)

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
