from celluloid import Camera
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

t = np.load("time.npy").squeeze(-1)
cat_states = np.load("cat_states.npy")
cat_controls = np.load("cat_controls.npy")
N = 10

time_horizon = np.zeros((np.size(t), N))
for i in range(np.size(t)):
    for n in range(N):
        time_horizon[i, n] = t[i + n]
    

def plot():
    """Fig 1. States"""
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)
    camera = Camera(fig)
    breakpoint()

    ax = axes[0, 0]
    ax.plot(t, cat_states[0, 0, :])
    # for i in range(np.length(t)):
        # horizon = t[i]:(t[i]+1):0.1


    ax.set_ylabel(r"$z$ [m]")
    ax.set_xlabel("Time [s]")
    ax.set_ylim([-11, 0])
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_states[1, 0, :])
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, cat_states[2, 0, :])
    ax.set_ylabel(r"$v_z$ [m/s]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    """ Fig 2. Controls """
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(t, -cat_controls[0, 0, :])
    ax.set_ylabel(r"$F_r$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[1, 0]
    ax.plot(t, cat_controls[1, 0, :])
    ax.set_ylabel(r"$F_p$ [N]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    ax = axes[2, 0]
    ax.plot(t, np.rad2deg(cat_controls[2, 0, :]))
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_xlabel("Time [s]")
    # ax.grid()

    plt.tight_layout()
    fig.align_ylabels(axes)

    plt.show()
    

if __name__ == "__main__":
    breakpoint()
    plot()

