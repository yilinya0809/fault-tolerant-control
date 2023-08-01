import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2angle
from numpy import cos, sin

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")

class MyEnv_A(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -10.0)),
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
        self.controller = ftc.make("NDI-A", self)
        self.cruise_speed = 45
        self.ang_lim = np.deg2rad(30)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        alp = ang[1]

        VT = self.cruise_speed
        posd = np.vstack((t*VT*cos(alp), 0, -10))
        posd_dot = np.vstack((VT*cos(alp), 0, VT*sin(alp)))

        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
        }

        return env_info


class MyEnv_B(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -10.0)),
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
        self.controller = ftc.make("NDI-B", self)
        self.cruise_speed = 45
        self.ang_lim = np.deg2rad(30)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        # ang_min, ang_max = -self.ang_lim, self.ang_lim
        # ang = np.clip(ang0, ang_min, ang_max)
        alp = ang[1]

        VT = self.cruise_speed
        posd = np.vstack((t*VT*cos(alp), 0, -10))
        posd_dot = np.vstack((VT*cos(alp), 0, VT*sin(alp)))

        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
        }

        return env_info

class MyEnv_C(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, -10.0)),
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
        self.controller = ftc.make("NDI-C", self)
        self.cruise_speed = 45
        self.ang_lim = np.deg2rad(30)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        pos, vel, quat, omega = self.plant.observe_list()
        ang0 = np.vstack(quat2angle(quat)[::-1])
        ang_min, ang_max = -self.ang_lim, self.ang_lim
        ang = np.clip(ang0, ang_min, ang_max)
        alp = ang[1]

        VT = self.cruise_speed
        posd = np.vstack((t*VT*cos(alp), 0, -10))
        posd_dot = np.vstack((VT*cos(alp), 0, VT*sin(alp)))

        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        pos, vel, quat, omega = self.plant.observe_list()
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = self.plant.saturate(ctrls0)

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
        }

        return env_info


def run():
    env_A = MyEnv_A()
    env_B = MyEnv_B()
    env_C = MyEnv_C()
    flogger = fym.Logger("data.h5")

    env_A.reset()
    env_B.reset()
    env_C.reset()
    try:
        while True:
            env_A.render()
            env_B.render()
            env_C.render()

            done, env_info_A = env_A.step()
            done, env_info_B = env_B.step()
            done, env_info_C = env_C.step()
            flogger.record(env_A=env_info_A)
            flogger.record(env_B=env_info_B)
            flogger.record(env_C=env_info_C)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data_A = fym.load("data.h5")["env_A"]
    data_B = fym.load("data.h5")["env_B"]
    data_C = fym.load("data.h5")["env_C"]


    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)
    ax = axes[0, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 0].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 0].squeeze(-1), "g-")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 2000])

    ax = axes[1, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 1].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 1].squeeze(-1), "g-")
 
    ax.plot(data_A["t"], data_A["posd"][:, 1].squeeze(-1), "r--")
    ax.plot(data_B["t"], data_B["posd"][:, 1].squeeze(-1), "r--")
    ax.plot(data_C["t"], data_C["posd"][:, 1].squeeze(-1), "r--")

    ax.set_ylabel(r"$y$, m")
    ax.legend(["A response", "B response", "C response", "A command", "B command", "C command"], loc="upper right")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 2].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 2].squeeze(-1), "g-")
 
    ax.plot(data_A["t"], data_A["posd"][:, 2].squeeze(-1), "r--")
    ax.plot(data_B["t"], data_B["posd"][:, 2].squeeze(-1), "r--")
    ax.plot(data_C["t"], data_C["posd"][:, 2].squeeze(-1), "r--")

    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-12, -9])


    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 0].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 0].squeeze(-1), "g-")
    ax.plot(data_A["t"], data_A["veld"][:, 0].squeeze(-1), "r--")
    ax.plot(data_B["t"], data_B["veld"][:, 0].squeeze(-1), "r--")
    ax.plot(data_C["t"], data_C["veld"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_ylim([0, 50])

    ax = axes[1, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 1].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 1].squeeze(-1), "g-")
    ax.plot(data_A["t"], data_A["veld"][:, 1].squeeze(-1), "r--")
    ax.plot(data_B["t"], data_B["veld"][:, 1].squeeze(-1), "r--")
    ax.plot(data_C["t"], data_C["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 2].squeeze(-1), "b-")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 2].squeeze(-1), "g-")
    ax.plot(data_A["t"], data_A["veld"][:, 2].squeeze(-1), "r--")
    ax.plot(data_B["t"], data_B["veld"][:, 2].squeeze(-1), "r--")
    ax.plot(data_C["t"], data_C["veld"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    # ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 0].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 0].squeeze(-1)), "g-")

    ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 0].squeeze(-1)), "r--")
    ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 0].squeeze(-1)), "r--")
    ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 1].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 1].squeeze(-1)), "g-")

    ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 1].squeeze(-1)), "r--")
    ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 1].squeeze(-1)), "r--")
    ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 2].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 2].squeeze(-1)), "g-")

    ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 2].squeeze(-1)), "r--")
    ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 2].squeeze(-1)), "r--")
    ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])
    ax.set_xlabel("Time, sec")


    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 0].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 0].squeeze(-1)), "g-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 1].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 1].squeeze(-1)), "g-")
    ax.set_ylabel(r"$q$, deg/s")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 2].squeeze(-1)), "b-")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 2].squeeze(-1)), "g-")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(left = 0.05, right = 0.99, wspace=0.3)
    fig.align_ylabels(axes)


    """ Figure 2 - Thrusts """
    fig, axs = plt.subplots(2, 4, sharex=True)

    ax = axs[0, 0]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 0], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 0], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 0], "g-")
    ax.set_ylabel("Rotor 1")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 0]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 1], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 1], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 1], "g-")
    ax.set_ylabel("Rotor 2")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")
 
    ax = axs[0, 1]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 2], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 2], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 2], "g-")
    ax.set_ylabel("Rotor 3")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 1]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 3], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 3], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 3], "g-")
    ax.set_ylabel("Rotor 4")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 2]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 4], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 4], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 4], "g-")
    ax.set_ylabel("Rotor 5")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 2]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 5], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 5], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 5], "g-")
    ax.set_ylabel("Rotor 6")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 3]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 6], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 6], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 6], "g-")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([-0.5, 1.5])
    ax.set_box_aspect(1)

    ax = axs[1, 3]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 7], "k-")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 7], "b-")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 7], "g-")
    ax.set_ylabel("Pusher 2")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([-0.5, 1.5])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    fig.suptitle("Rotational Thrusts", y=0.85)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.8, wspace=0.25, hspace=0.2)
    fig.align_ylabels(axs)


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
