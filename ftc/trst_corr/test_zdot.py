import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 14,
    }
)

# ── Load data ──────────────────────────────────────────────────────────────────
data_fwd = {}
with h5py.File("opt_forward_F.h5", "r") as f:
    data_fwd["tf"] = f["tf"][()]
    data_fwd["X"]  = f["X"][:]
    data_fwd["U"]  = f["U"][:]

data_bwd = {}
with h5py.File("opt_backward_F.h5", "r") as f:
    data_bwd["tf"] = f["tf"][()]
    data_bwd["X"]  = f["X"][:]
    data_bwd["U"]  = f["U"][:]

N = 200  # number of intervals (nodes = N+1)


def compute_zdot(data):
    """
    Compute zdot two ways:
      1. From Eq.(15): zdot_eq = -Vx * sin(theta) + Vz * cos(theta)
         theta(t) is interpolated from N interval values to N+1 node values
         using linear interpolation at interval midpoints.
      2. Numerical differentiation of z: zdot_num = diff(z) / dt

    Returns:
        tspan    : (N+1,) time array
        zdot_eq  : (N+1,) zdot from Eq.(15)
        zdot_num : (N,)   zdot from numerical differentiation
    """
    tf    = data["tf"]
    tspan = np.linspace(0, tf, N + 1)       # node times
    dt    = tf / N

    z     = data["X"][0, :]                  # (N+1,) state at nodes
    Vx    = data["X"][1, :]                  # (N+1,)
    Vz    = data["X"][2, :]                  # (N+1,)
    theta = data["U"][2, :]                  # (N,)   control at intervals

    # Interval midpoint times
    t_mid = (tspan[:-1] + tspan[1:]) / 2    # (N,)

    # Interpolate theta from midpoints to nodes via linear interpolation
    theta_nodes = np.interp(tspan, t_mid, theta)  # (N+1,)

    zdot_eq  = -Vx * np.sin(theta_nodes) + Vz * np.cos(theta_nodes)  # (N+1,)
    zdot_num = np.diff(z) / dt                                         # (N,)

    return tspan, zdot_eq, zdot_num, theta_nodes


# ── Compute ────────────────────────────────────────────────────────────────────
tspan_f, zdot_eq_f, zdot_num_f, theta_f = compute_zdot(data_fwd)
tspan_b, zdot_eq_b, zdot_num_b, theta_b = compute_zdot(data_bwd)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

# ── Forward transition ──
ax = axs[0]
ax.plot(tspan_f,       zdot_eq_f,  "b-",  linewidth=2,
        label=r"$\dot{z}$ from Eq.(15): $-V_x^B\sin\theta + V_z^B\cos\theta$")
ax.plot(tspan_f[:-1],  zdot_num_f, "r--", linewidth=1.5,
        label=r"$\dot{z}$ numerical differentiation of $z(t)$")
ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
ax.set_ylabel(r"$\dot{z}$, m/s", fontsize=14)
ax.set_title("Accelerating Transition", fontsize=14)
ax.set_xlim([0, data_fwd["tf"]])
ax.set_ylim([-1, 1])
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.4)

# ── Backward transition ──
ax = axs[1]
ax.plot(tspan_b,       zdot_eq_b,  "b-",  linewidth=2,
        label=r"$\dot{z}$ from Eq.(15): $-V_x^B\sin\theta + V_z^B\cos\theta$")
ax.plot(tspan_b[:-1],  zdot_num_b, "r--", linewidth=1.5,
        label=r"$\dot{z}$ numerical differentiation of $z(t)$")
ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
ax.set_ylabel(r"$\dot{z}$, m/s", fontsize=14)
ax.set_xlabel("Time, s", fontsize=14)
ax.set_title("Decelerating Transition", fontsize=14)
ax.set_xlim([0, data_bwd["tf"]])
ax.set_ylim([-1, 1])
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.4)

fig.suptitle(
    r"Verification that $\dot{z} \approx 0$ and Eq.(15) is satisfied"
    "\nalong the optimal trajectory",
    fontsize=13,
)
fig.tight_layout()
fig.savefig("zdot_verification.png", bbox_inches="tight", dpi=300)
print("Saved: zdot_verification.png")


# ── Representative time-point table ───────────────────────────────────────────
def print_table(data, label, n_points=6):
    tf      = data["tf"]
    tspan   = np.linspace(0, tf, N + 1)
    z       = data["X"][0, :]
    Vx      = data["X"][1, :]
    Vz      = data["X"][2, :]
    theta_u = data["U"][2, :]
    dt      = tf / N

    # Same interpolation as above
    t_mid      = (tspan[:-1] + tspan[1:]) / 2
    theta_nodes = np.interp(tspan, t_mid, theta_u)

    zdot_eq  = -Vx * np.sin(theta_nodes) + Vz * np.cos(theta_nodes)
    zdot_num = np.append(np.diff(z) / dt, np.nan)

    # Sample indices — skip last node (nan in zdot_num)
    indices = np.linspace(0, N - 1, n_points, dtype=int)

    print(f"\n{'='*75}")
    print(f"  {label}  (tf = {tf:.2f} s)")
    print(f"{'='*75}")
    print(f"  {'t (s)':>7}  {'Vx_B (m/s)':>12}  {'Vz_B (m/s)':>12}  "
          f"{'theta (deg)':>12}  {'zdot_eq':>10}  {'zdot_num':>10}")
    print(f"  {'-'*70}")
    for k in indices:
        th_deg = np.rad2deg(theta_nodes[k])
        print(f"  {tspan[k]:>7.2f}  {Vx[k]:>12.4f}  {Vz[k]:>12.4f}  "
              f"{th_deg:>12.4f}  {zdot_eq[k]:>10.6f}  {zdot_num[k]:>10.6f}")


print_table(data_fwd, "Accelerating Transition")
print_table(data_bwd, "Decelerating Transition")

plt.show()
