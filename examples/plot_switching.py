import fym
import h5py
import matplotlib.pyplot as plt
import numpy as np


""" Optimal transition trajectory """
opt_traj = {}
f = h5py.File("data/opt_corr.h5", "r")
opt_traj["tf"] = f.get("tf")[()]
opt_traj["X"] = f.get("X")[:]
opt_traj["U"] = f.get("U")[:]

N = np.shape(opt_traj["U"])[1]
tspan = np.linspace(0, opt_traj["tf"], N + 1)

data_opt = fym.load("data/data_opt_switch.h5")["env"]
data_mpc = fym.load("data/data_mpc_switch.h5")["env"]

""" Figure 1 - States """
fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

""" Column 1 - States: Position """
ax = axes[0, 0]
ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 0].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 0].squeeze(-1), "k-")
ax.set_ylabel(r"$x$, m")
ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

ax = axes[1, 0]
ax.plot(data_opt["t"], data_opt["posd"][:, 1], "r--")
ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 1].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 1].squeeze(-1), "k-")
ax.set_ylabel(r"$y$, m")
ax.set_ylim([-1, 1])

ax = axes[2, 0]
# ax.plot(tspan, opt_traj["X"][0, :], "r--")
ax.plot(data_opt["t"], data_opt["posd"][:, 2], "r--")
ax.plot(data_opt["t"], data_opt["plant"]["pos"][:, 2].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["pos"][:, 2].squeeze(-1), "k-")
ax.set_ylabel(r"$z$, m")
ax.set_ylim([-15, -5])

ax.set_xlabel("Time, sec")

""" Column 2 - States: Velocity """
ax = axes[0, 1]
ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 0].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 0].squeeze(-1), "k-")
ax.plot(data_opt["t"], data_opt["veld"][:, 0], "r--")
ax.set_ylabel(r"$v_x$, m/s")

ax = axes[1, 1]
ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 1].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 1].squeeze(-1), "k-")
ax.plot(data_opt["t"], data_opt["veld"][:, 1], "r--")
ax.set_ylabel(r"$v_y$, m/s")
ax.set_ylim([-1, 1])

ax = axes[2, 1]
ax.plot(data_opt["t"], data_opt["plant"]["vel"][:, 2].squeeze(-1), "b-")
ax.plot(data_mpc["t"], data_mpc["plant"]["vel"][:, 2].squeeze(-1), "k-")
ax.plot(data_opt["t"], data_opt["veld"][:, 2], "r--")
ax.set_ylabel(r"$v_z$, m/s")
ax.set_ylim([-10, 10])

ax.set_xlabel("Time, sec")

""" Column 3 - States: Euler angles """
ax = axes[0, 2]
ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 0].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 0].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 0].squeeze(-1)), "r--")
ax.set_ylabel(r"$\phi$, deg")
ax.set_ylim([-1, 1])

ax = axes[1, 2]
ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 1].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 1].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 1].squeeze(-1)), "r--")
ax.set_ylabel(r"$\theta$, deg")

ax = axes[2, 2]
ax.plot(data_opt["t"], np.rad2deg(data_opt["ang"][:, 2].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["ang"][:, 2].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["angd"][:, 2].squeeze(-1)), "r--")
ax.set_ylabel(r"$\psi$, deg")
ax.set_ylim([-1, 1])

ax.set_xlabel("Time, sec")

""" Column 4 - States: Angular rates """
ax = axes[0, 3]
ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 0].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 0].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 0].squeeze(-1)), "r--")
ax.set_ylabel(r"$p$, deg/s")
ax.set_ylim([-1, 1])

ax = axes[1, 3]
ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 1].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 1].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 1].squeeze(-1)), "r--")
ax.set_ylabel(r"$q$, deg/s")

ax = axes[2, 3]
ax.plot(data_opt["t"], np.rad2deg(data_opt["plant"]["omega"][:, 2].squeeze(-1)), "b-")
ax.plot(data_mpc["t"], np.rad2deg(data_mpc["plant"]["omega"][:, 2].squeeze(-1)), "k-")
# ax.plot(data_opt["t"], np.rad2deg(data_opt["omegad"][:, 2].squeeze(-1)), "r--")
ax.set_ylabel(r"$r$, deg/s")
ax.set_ylim([-1, 1])

ax.set_xlabel("Time, sec")

fig.tight_layout()

""" Figure 2 - Rotor inputs """
fig, axes = plt.subplots(3, 2, sharex=True)

ax = axes[0, 0]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 0], "b-")
ax.set_ylabel("Rotor 1")
ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

ax = axes[1, 0]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 1], "b-")
ax.set_ylabel("Rotor 2")

ax = axes[2, 0]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 2], "b-")
ax.set_ylabel("Rotor 3")
ax.set_xlabel("Time, sec")

ax = axes[0, 1]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 3], "b-")
ax.set_ylabel("Rotor 4")

ax = axes[1, 1]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 4], "b-")
ax.set_ylabel("Rotor 5")

ax = axes[2, 1]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 5], "b-")
ax.set_ylabel("Rotor 6")
ax.set_xlabel("Time, sec")

plt.tight_layout()
fig.subplots_adjust(wspace=0.3)
fig.align_ylabels(axes)

""" Figure 3 - Pusher input """
fig, axes = plt.subplots(2, 1, sharex=True)

ax = axes[0]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 6], "b-")
ax.set_ylabel("Pusher 1")
ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

ax = axes[1]
ax.plot(data_opt["t"], data_opt["ctrls"].squeeze(-1)[:, 7], "b-")
ax.set_ylabel("Pusher 2")
ax.set_xlabel("Time, sec")

plt.tight_layout()
fig.align_ylabels(axes)

""" Figure 5 - Thrust """
fig, axes = plt.subplots(2, 1, sharex=True)

ax = axes[0]
ax.plot(data_opt["t"], data_opt["Frd"], "r--")
ax.plot(data_opt["t"], data_opt["Fr"].squeeze(-1), "b-")
ax.set_ylabel(r"$F_{rotors}$, N")
ax.set_xlim(data_opt["t"][0], data_opt["t"][-1])

ax = axes[1]
ax.plot(data_opt["t"], data_opt["Fpd"], "r--")
ax.plot(data_opt["t"], data_opt["Fp"].squeeze(-1), "b-")
ax.set_ylabel(r"$F_{pushers}$, N")
ax.set_xlabel("Time, sec")

plt.tight_layout()
fig.align_ylabels(axes)

plt.show()


