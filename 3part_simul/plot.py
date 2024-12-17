import h5py
import matplotlib.pyplot as plt
import numpy as np

data = {}
with h5py.File("data.h5", "r") as f:
    data["t"] = f["flight_data"]["time"][:]
    data["pos"] = f["flight_data"]["cur_local_pos"][:]
    data["vel"] = f["flight_data"]["vel"][:]
    data["ang"] = f["flight_data"]["ang"][:]
    data["omega"] = f["flight_data"]["omega"][:]

    for key in f["flight_data"]:
        print(f"{key}: {f['flight_data'][key][:]}")


""" Figure 1 - States """
fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

""" Column 1 - States: Position """
ax = axes[0, 0]
ax.plot(data["t"], data["pos"][:, 0], "k-")
ax.set_ylabel(r"$x$, m")
ax.set_xlim(data["t"][0], data["t"][-1])

ax = axes[1, 0]
ax.plot(data["t"], data["pos"][:, 1], "k-")
ax.set_ylabel(r"$y$, m")

ax = axes[2, 0]
ax.plot(data["t"], data["pos"][:, 2], "k-")
ax.set_ylabel(r"$z$, m")

ax.set_xlabel("Time, sec")

""" Column 2 - States: Velocity """
ax = axes[0, 1]
ax.plot(data["t"], data["vel"][:, 0], "k-")
ax.set_ylabel(r"$v_x$, m/s")

ax = axes[1, 1]
ax.plot(data["t"], data["vel"][:, 1], "k-")
ax.set_ylabel(r"$v_y$, m/s")

ax = axes[2, 1]
ax.plot(data["t"], data["vel"][:, 2], "k-")
ax.set_ylabel(r"$v_z$, m/s")

ax.set_xlabel("Time, sec")

""" Column 3 - States: Euler angles """
ax = axes[0, 2]
ax.plot(data["t"], np.rad2deg(data["ang"][:, 0]), "k-")
ax.set_ylabel(r"$\phi$, deg")

ax = axes[1, 2]
ax.plot(data["t"], np.rad2deg(data["ang"][:, 1]), "k-")
ax.set_ylabel(r"$\theta$, deg")

ax = axes[2, 2]
ax.plot(data["t"], np.rad2deg(data["ang"][:, 2]), "k-")
ax.set_ylabel(r"$\psi$, deg")

ax.set_xlabel("Time, sec")

""" Column 4 - States: Angular rates """
ax = axes[0, 3]
ax.plot(data["t"], np.rad2deg(data["omega"][:, 0]), "k-")
ax.set_ylabel(r"$p$, deg/s")

ax = axes[1, 3]
ax.plot(data["t"], np.rad2deg(data["omega"][:, 1]), "k-")
ax.set_ylabel(r"$q$, deg/s")

ax = axes[2, 3]
ax.plot(data["t"], np.rad2deg(data["omega"][:, 2]), "k-")
ax.set_ylabel(r"$r$, deg/s")

ax.set_xlabel("Time, sec")

fig.tight_layout()
fig.subplots_adjust(wspace=0.3)
fig.align_ylabels(axes)

plt.show()
