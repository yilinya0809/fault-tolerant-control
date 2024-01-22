import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from dyn import Quadrotor
from fym.utils.rot import angle2quat, quat2angle
from numpy import cos, sin

import ftc
# from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate
from GESO import GESOController

np.seterr(all="raise")

class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 10,
        },
        "plant": {
            "init": {
                "quat": np.vstack((1, 0, 0, 0)),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = Quadrotor(env_config["plant"])
        self.controller = GESOController(self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        phid = np.deg2rad(30)
        thetad = np.deg2rad(15)
        psid = np.deg2rad(25)
        quatd = np.vstack(angle2quat(psid, thetad, phid))
        omegad = np.vstack((0, 0, 0))

        refs = {"quatd": quatd, "omegad": omegad}
        return [refs[key] for key in args]

    def disturbance(self, t):
        num = 7
        w = (np.random.rand(num, 1) * 2.45 + 0.05) * np.pi
        p = np.random.rand(num, 1) * 10
        a = np.random.rand(num, 1) * 2
        d0 = 1
        
        dtrb_wind = d0
        for i in range(num):
            dtrb_wind = dtrb_wind + a[i] * np.sin(w[i] * t + p[i])

#         del_J = 0.3 * self.plant.J
#         dtrb_model = np.cross(omega, del_J.dot(omega), axis=0) - del_J.dot(omega_dot)
        dtrb_model = np.zeros((3, 1))

        dtrb = dtrb_wind + dtrb_model

        dtrb_info = {
            "freq": w,
            "phase shift": p,
            "amplitude of sinusoid": a,
        }
        return dtrb, dtrb_info




    def set_dot(self, t):
        quat, omega = self.plant.observe_list()
        dtrb, _ = self.disturbance(t)
        u_nominal, controller_info = self.controller.get_control(t, self)
        u_dtrb, observer_info = self.controller.geso(t, self)

        u = u_nominal + u_dtrb
        self.plant.set_dot(t, u, dtrb)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            **observer_info,
            "nominal_ctrl": u_nominal,
            "dtrb_ctrl": u_dtrb,
        }
        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data.h5")
    
    env.reset()
    try:
        while True:
            env.render()

            done, env_info = env.step()
            flogger.record(env=env_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data.h5")["env"]

    fig, axes = plt.subplots(1, 3, squeeze=False, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "r-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "k--")
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[1]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "r-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "k--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "r-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "k--")
    ax.set_ylabel(r"$\psi$, deg")
    
    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
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
