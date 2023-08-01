import argparse

import fym
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fym.utils.rot import quat2angle
from numpy import cos, sin

import ftc
from ftc.models.LC62R import LC62R
from ftc.utils import safeupdate

np.seterr(all="raise")


def plot():
    data_A = fym.load("data_A.h5")["env"]
    data_B = fym.load("data_B.h5")["env"]
    data_C = fym.load("data_C.h5")["env"]

    """ Figure 1 - States """
    mpl.style.use('default')
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)
    ax = axes[0, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 0].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 0].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 0].squeeze(-1), "C1")
    ax.set_ylabel(r"$x$, m")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 2000])

    ax = axes[1, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 2].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 1].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 1].squeeze(-1), "C1")
 
    ax.plot(data_A["t"], data_A["posd"][:, 1].squeeze(-1), "C0--") 
    ax.plot(data_B["t"], data_B["posd"][:, 1].squeeze(-1), "C2--")
    ax.plot(data_C["t"], data_C["posd"][:, 1].squeeze(-1), "C1--")

    ax.set_ylabel(r"$y$, m")
    ax.set_ylim([-1, 1])

    ax = axes[2, 0]
    ax.plot(data_A["t"], data_A["plant"]["pos"][:, 2].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["pos"][:, 2].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["pos"][:, 2].squeeze(-1), "C1")

    ax.set_xlabel("Time, sec")
    ax.set_ylabel(r"$z$, m")
    ax.set_ylim([-12, -9])


    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 0].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 0].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 0].squeeze(-1), "C1")
    # ax.plot(data_A["t"], data_A["veld"][:, 0].squeeze(-1), "C0--")
    # ax.plot(data_B["t"], data_B["veld"][:, 0].squeeze(-1), "C2--")
    # ax.plot(data_C["t"], data_C["veld"][:, 0].squeeze(-1), "C1--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_ylim([0, 50])

    ax = axes[1, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 1].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 1].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 1].squeeze(-1), "C1")
    ax.plot(data_A["t"], data_A["veld"][:, 1].squeeze(-1), "C0--")
    ax.plot(data_B["t"], data_B["veld"][:, 1].squeeze(-1), "C2--")
    ax.plot(data_C["t"], data_C["veld"][:, 1].squeeze(-1), "C1--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_ylim([-1, 1])

    ax = axes[2, 1]
    ax.plot(data_A["t"], data_A["plant"]["vel"][:, 2].squeeze(-1), "C0")
    ax.plot(data_B["t"], data_B["plant"]["vel"][:, 2].squeeze(-1), "C2")
    ax.plot(data_C["t"], data_C["plant"]["vel"][:, 2].squeeze(-1), "C1")
    ax.plot(data_A["t"], data_A["veld"][:, 2].squeeze(-1), "C0--")
    ax.plot(data_B["t"], data_B["veld"][:, 2].squeeze(-1), "C2--")
    ax.plot(data_C["t"], data_C["veld"][:, 2].squeeze(-1), "C1--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 0].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 0].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 0].squeeze(-1)), "C1")

    ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 0].squeeze(-1)), "C0--")
    ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 0].squeeze(-1)), "C2--")
    ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 0].squeeze(-1)), "C1--")
    ax.set_ylabel(r"$\phi$, deg")
    ax.set_ylim([-1, 1])

    ax = axes[1, 2]
    ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 1].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 1].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 1].squeeze(-1)), "C1")

    # ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 1].squeeze(-1)), "C0--")
    # ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 1].squeeze(-1)), "C2--")
    # ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 1].squeeze(-1)), "C1--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data_A["t"], np.rad2deg(data_A["ang"][:, 2].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["ang"][:, 2].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["ang"][:, 2].squeeze(-1)), "C1")

    # ax.plot(data_A["t"], np.rad2deg(data_A["angd"][:, 2].squeeze(-1)), "C0--")
    # ax.plot(data_B["t"], np.rad2deg(data_B["angd"][:, 2].squeeze(-1)), "C2--")
    # ax.plot(data_C["t"], np.rad2deg(data_C["angd"][:, 2].squeeze(-1)), "C1--")
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_ylim([-1, 1])
    ax.set_xlabel("Time, sec")

    ax.legend(["A", "B", "C"], ncol=3, bbox_to_anchor=(0.2, -0.3)) 


    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 0].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 0].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 0].squeeze(-1)), "C1")
    ax.set_ylabel(r"$p$, deg/s")
    ax.set_ylim([-1, 1])

    ax = axes[1, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 1].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 1].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 1].squeeze(-1)), "C1")
    ax.set_ylabel(r"$q$, deg/s")
    # ax.set_ylim([-1, 1])

    ax = axes[2, 3]
    ax.plot(data_A["t"], np.rad2deg(data_A["plant"]["omega"][:, 2].squeeze(-1)), "C0")
    ax.plot(data_B["t"], np.rad2deg(data_B["plant"]["omega"][:, 2].squeeze(-1)), "C2")
    ax.plot(data_C["t"], np.rad2deg(data_C["plant"]["omega"][:, 2].squeeze(-1)), "C1")
    ax.set_ylabel(r"$r$, deg/s")
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(left = 0.05, right = 0.99, wspace=0.3)
    fig.align_ylabels(axes)


    """ Figure 2 - Thrusts """
    fig, axs = plt.subplots(2, 4, sharex=True)

    ax = axs[0, 0]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 0], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 0], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 0], "C1")
    ax.set_ylabel("Rotor 1")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 0]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 1], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 1], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 1], "C1")
    ax.set_ylabel("Rotor 2")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")
 
    ax = axs[0, 1]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 2], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 2], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 2], "C1")
    ax.set_ylabel("Rotor 3")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 1]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 3], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 3], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 3], "C1")
    ax.set_ylabel("Rotor 4")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax = axs[0, 2]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 4], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 4], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 4], "C1")
    ax.set_ylabel("Rotor 5")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)

    ax = axs[1, 2]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 5], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 5], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 5], "C1")
    ax.set_ylabel("Rotor 6")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel("Time, sec")

    ax.legend(["A", "B", "C"], ncol=3, bbox_to_anchor=(0.2, -0.3)) 

    ax = axs[0, 3]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 6], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 6], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 6], "C1")
    ax.set_ylabel("Pusher 1")
    ax.set_xlim(data_A["t"][0], data_A["t"][-1])
    ax.set_ylim([-0.5, 1.5])
    ax.set_box_aspect(1)

    ax = axs[1, 3]
    ax.plot(data_A["t"], data_A["ctrls"].squeeze(-1)[:, 7], "C0")
    ax.plot(data_B["t"], data_B["ctrls"].squeeze(-1)[:, 7], "C2")
    ax.plot(data_C["t"], data_C["ctrls"].squeeze(-1)[:, 7], "C1")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)

