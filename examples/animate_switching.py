"""Supplemental-video animation for the LC62 mode-switching mission.

Top-left  : x-z side view of the LC62.  The airframe pitches per the logged
            attitude and rises/descends at the true altitude.  Live thrust
            arrows show each rotor station (front/mid/rear), the pusher, and the
            aerodynamic resultant force (all in N, one shared physical scale).
Top-right : (V, theta) transition corridor.  ATC for the first half, DTC for the
            second; the live state point traces its trajectory during the
            transitions.  Backgrounds ramp/cross-fade smoothly.
Bottom    : full-width global side view of the whole 0-1250 m journey with the
            flown path, a pitch-tracking vehicle marker, and a mode timeline.

Data source : data_opt_switch.h5  (vertical takeoff -> accelerating transition
              -> cruise -> decelerating transition -> hover -> vertical landing)

Individual rotor/pusher thrusts are reconstructed from the clean env/ctrls
through the LC62R thrust maps (env/th_r, env/th_p are uninitialised garbage);
verified against env/Fr, env/Fp to 0.0 N.  The aerodynamic force is recomputed
from the sim's own LC62R.B_Fuselage model (not stored in the log); at cruise its
vertical component is ~482 N ~= 1.17 m*g, confirming the recompute.

This script only READS data files.  Run inside the `ftc` conda environment.

Usage
-----
    python examples/animate_switching.py --closeup   # theta=0 vehicle zoom
    python examples/animate_switching.py             # preview PNGs
    python examples/animate_switching.py --mp4       # full mp4
"""

import argparse
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Ellipse
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
from matplotlib import animation
from scipy.interpolate import interp1d

from ftc.models.LC62R_lin import LC62R

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
H5 = os.path.join(ROOT, "data_opt_switch.h5")
ATC_NPZ = os.path.join(ROOT, "data", "corr_forward_wide.npz")
DTC_NPZ = os.path.join(ROOT, "data", "corr_backward.npz")
OUTDIR = os.path.join(HERE, "anim_out")

G = 9.81

# --- mission timing (from controller_opt_switch.get_ref) ---------------------
T_FTC0 = 5.0
T_FTC1 = 5.0 + 9.564651128192589      # 14.5647
T_CRZ1 = 25.0
T_BTC1 = 25.0 + 22.082337803707862    # 47.0823
T_HOV1 = 50.0
T_SWAP = T_CRZ1
RAMP = 0.5                            # corridor emphasis ramp [s]
XFADE_HW = 0.25                       # half-width of the ATC<->DTC cross-fade [s]

# rendering
STEP = 4
FPS = 25
DPI = 100
FIGSIZE = (16, 9)

# one shared physical arrow scale [m/N]; sized so the largest force (aero ~527 N)
# maps to ~2.6 m, keeping rotor/pusher arrows honestly proportional.
ARROW_SCALE = 2.6 / 527.0
BASE_HEAD = 9.0          # arrow-head mutation_scale (~60% of the old 15)
HEAD_MAX_FRAC = 0.30     # head length capped at this fraction of arrow length
LC62_3D = os.path.join(OUTDIR, "lc62_3d.png")

_CMD_TAB = np.array([0, 0.2, 0.255, 0.310, 0.365, 0.420,
                     0.475, 0.530, 0.585, 0.640, 0.695, 0.750])
_THP_TAB = np.array([0, 1.39, 4.22, 7.89, 12.36, 17.60,
                     23.19, 29.99, 39.09, 46.14, 52.67, 59.69])

MODE_NAMES = [
    "Vertical takeoff", "Accelerating transition", "Cruise",
    "Decelerating transition", "Hover", "Vertical landing",
]
# (name index, t0, t1); transition modes are 1 and 3
MODE_SEGS = [(0, 0.0, T_FTC0), (1, T_FTC0, T_FTC1), (2, T_FTC1, T_CRZ1),
             (3, T_CRZ1, T_BTC1), (4, T_BTC1, T_HOV1), (5, T_HOV1, 60.0)]


def mode_of(t):
    for k, t0, t1 in MODE_SEGS:
        if t < t1:
            return k
    return 5


def emphasis(t):
    """0 (faint) away from transitions, ramps to 1 (bright) around each."""
    def band(a, b):
        if a - RAMP <= t < a:
            return (t - (a - RAMP)) / RAMP
        if a <= t <= b:
            return 1.0
        if b < t <= b + RAMP:
            return 1.0 - (t - b) / RAMP
        return 0.0
    return max(band(T_FTC0, T_FTC1), band(T_CRZ1, T_BTC1))


def crossfade(t):
    """0 = ATC fully shown, 1 = DTC fully shown; linear over 2*XFADE_HW at swap."""
    if t <= T_SWAP - XFADE_HW:
        return 0.0
    if t >= T_SWAP + XFADE_HW:
        return 1.0
    return (t - (T_SWAP - XFADE_HW)) / (2 * XFADE_HW)


# --- LC62 geometry (canonical: ftc/models/LC62R.py) --------------------------
DX1, DX2, DX3 = 0.9815, 0.0235, 1.1235
RR = 0.762 / 2
RP = 0.525 / 2
STATION_XI = np.array([DX1, -DX2, -DX3])
ETA_ROT = 0.34
PUSH_XI = -DX3 / 2
C_FRONT = 0.2624
C_REAR = 0.5898

C_BODY = "0.82"
C_WING = "0.62"
C_ROTOR = "tab:pink"
C_STRUT = "0.35"
C_PUSH = "tab:purple"
C_RARROW = "tab:red"          # rotor thrust
C_PARROW = "tab:purple"       # pusher thrust
C_AARROW = "tab:blue"         # aerodynamic force


def load_data():
    with h5py.File(H5, "r") as f:
        t = f["env/t"][:]
        pos = f["env/plant/pos"][:, :, 0]
        vel = f["env/plant/vel"][:, :, 0]
        omega = f["env/plant/omega"][:]        # (N,3,1) kept 3D for the model
        ang = f["env/ang"][:, :, 0]
        veld = f["env/veld"][:, :, 0]
        angd = f["env/angd"][:, :, 0]
        ctrls = f["env/ctrls"][:, :, 0]
        pos3 = f["env/plant/pos"][:]
        vel3 = f["env/plant/vel"][:]
    theta = ang[:, 1]
    alt = -pos[:, 2]
    xpos = pos[:, 0]
    V = np.linalg.norm(vel, axis=1)
    Vd = np.linalg.norm(veld, axis=1)
    thetad = angd[:, 1]

    th_r = np.clip(np.polyval([-19281, 36503, -992.75, 0], ctrls[:, 0:6]) * G / 1000.0,
                   0.0, None)
    fp = interp1d(_CMD_TAB, _THP_TAB, fill_value="extrapolate")
    th_p = np.clip(fp(ctrls[:, 6:8]), 0.0, None)
    rotor = np.stack([0.5 * (th_r[:, 2] + th_r[:, 4]),
                      0.5 * (th_r[:, 0] + th_r[:, 1]),
                      0.5 * (th_r[:, 3] + th_r[:, 5])], axis=1)
    pusher = 0.5 * (th_p[:, 0] + th_p[:, 1])

    # aerodynamic resultant (body frame) from the sim's own B_Fuselage model
    plant = LC62R()
    N = len(t)
    aFx = np.zeros(N)
    aFz = np.zeros(N)
    for i in range(N):
        FM = plant.B_Fuselage(ctrls[i, 8:][:, None], pos3[i], vel3[i], omega[i])
        aFx[i] = FM[0, 0]
        aFz[i] = FM[2, 0]

    return dict(t=t, theta=theta, alt=alt, xpos=xpos, V=V, Vd=Vd, thetad=thetad,
                rotor=rotor, pusher=pusher, aFx=aFx, aFz=aFz)


def corridor(npz, theta_pad_to=None):
    d = np.load(npz)
    VT = d["VT_corr"]
    thc = np.rad2deg(d["theta_corr"])
    acc = np.array(d["acc"])
    if theta_pad_to is not None and theta_pad_to > thc[-1]:
        dth = thc[1] - thc[0]
        extra = np.arange(thc[-1] + dth, theta_pad_to + 1e-9, dth)
        if len(extra):
            pad = np.full((acc.shape[0], len(extra)), np.nan)
            for i in range(acc.shape[0]):
                fin = np.where(np.isfinite(acc[i]))[0]
                if len(fin) and fin[-1] == acc.shape[1] - 1:
                    pad[i, :] = acc[i, fin[-1]]
            acc = np.concatenate([acc, pad], axis=1)
            thc = np.concatenate([thc, extra])
    return dict(VT=VT, thc=thc, acc=acc)


# --- airframe silhouette in body frame (xi, eta) -----------------------------
def fuselage_xy():
    return np.array([
        [1.35, 0.00], [1.05, 0.15], [-0.90, 0.16], [-1.50, 0.10],
        [-1.78, 0.02], [-1.78, -0.02], [-1.50, -0.12], [-0.90, -0.16],
        [1.05, -0.13],
    ])


def rotor_bar_xy(xi):
    h = 0.035
    return np.array([[xi - RR, ETA_ROT + h], [xi + RR, ETA_ROT + h],
                     [xi + RR, ETA_ROT - h], [xi - RR, ETA_ROT - h]])


def strut_xy(xi):
    w = 0.03
    return np.array([[xi - w, ETA_ROT], [xi + w, ETA_ROT],
                     [xi + w, 0.02], [xi - w, 0.02]])


class SideView:
    """Top-left panel: pitching LC62 with rotor / pusher / aero force arrows."""

    def __init__(self, ax, ground=True):
        self.ax = ax
        ax.set_aspect("equal")
        ax.set_xlim(-4.8, 4.8)
        ax.set_ylim(-2.0, 13.0)          # extra sub-ground room for the legend
        ax.set_xlabel("body $x$ (centered), m", fontsize=12)
        ax.set_ylabel("altitude, m", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([0, 5, 10])
        if ground:
            ax.axhline(0.0, color="0.35", lw=2, zorder=0)
            ax.axhspan(-2.0, 0.0, color="0.85", zorder=0)

        self.struts = [Polygon(strut_xy(x), closed=True, fc=C_STRUT, ec="none",
                              zorder=2) for x in STATION_XI]
        self.bars = [Polygon(rotor_bar_xy(x), closed=True, fc=C_ROTOR, ec="0.15",
                            zorder=3) for x in STATION_XI]
        self.pusher = Ellipse((PUSH_XI, 0.0), width=0.09, height=2 * RP,
                             fc=C_PUSH, ec="0.15", zorder=7)
        self.body = Polygon(fuselage_xy(), closed=True, fc=C_BODY, ec="#333333",
                           lw=1.3, zorder=5)
        self.wing_r = Ellipse((-DX3 / 2, 0.0), width=C_REAR, height=0.12,
                             fc=C_WING, ec="#333333", zorder=6)
        self.wing_f = Ellipse((DX1 / 2, 0.0), width=C_FRONT, height=0.08,
                             fc=C_WING, ec="#333333", zorder=6)
        self.solid = [*self.struts, *self.bars, self.pusher, self.body,
                      self.wing_r, self.wing_f]
        for p in self.solid:
            ax.add_patch(p)

        self.r_arrows = [FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                        mutation_scale=15, color=C_RARROW, lw=2.5, zorder=9)
                        for _ in STATION_XI]
        self.p_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                        mutation_scale=15, color=C_PARROW, lw=2.5, zorder=9)
        self.a_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                        mutation_scale=15, color=C_AARROW, lw=2.5, zorder=8)
        for a in (*self.r_arrows, self.p_arrow, self.a_arrow):
            ax.add_artist(a)

        handles = [Line2D([0], [0], color=C_RARROW, lw=3, label="Rotor thrust"),
                   Line2D([0], [0], color=C_PARROW, lw=3, label="Pusher thrust"),
                   Line2D([0], [0], color=C_AARROW, lw=3, label="Aerodynamic force")]
        ax.legend(handles=handles, loc="lower left", ncol=1, fontsize=8,
                  frameon=True, framealpha=0.8, borderpad=0.4, handlelength=1.3,
                  labelspacing=0.3)

    def _ppu(self):
        """points per data-metre in this equal-aspect axes (for head sizing)."""
        try:
            h_px = self.ax.get_window_extent().height
            yr = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
            return h_px * 72.0 / self.ax.figure.dpi / yr
        except Exception:
            return 25.0

    def _set_arrow(self, arrow, base, tip, mag, thresh, ppu):
        if mag <= thresh:
            arrow.set_visible(False)         # fully hidden -> no stray head
            return
        arrow.set_visible(True)
        length_pts = np.hypot(tip[0] - base[0], tip[1] - base[1]) * ppu
        arrow.set_mutation_scale(min(BASE_HEAD, HEAD_MAX_FRAC * length_pts))
        arrow.set_positions(tuple(base), tuple(tip))

    def update(self, theta, alt, rotor, pusher, aFx, aFz):
        T = Affine2D().rotate(theta).translate(0.0, alt) + self.ax.transData
        for p in self.solid:
            p.set_transform(T)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        shift = np.array([0.0, alt])
        ppu = self._ppu()

        for i, a in enumerate(self.r_arrows):
            base = R @ np.array([STATION_XI[i], ETA_ROT + 0.04]) + shift
            self._set_arrow(a, base, base + R @ np.array([0.0, rotor[i] * ARROW_SCALE]),
                            rotor[i], 0.5, ppu)
        base = R @ np.array([PUSH_XI, 0.0]) + shift
        self._set_arrow(self.p_arrow, base,
                        base + R @ np.array([pusher * ARROW_SCALE, 0.0]),
                        pusher, 0.5, ppu)
        # aero resultant at the main (rear) wing; body (Fx, Fz) -> view (Fx, -Fz)
        base = R @ np.array([-DX3 / 2, 0.0]) + shift
        self._set_arrow(self.a_arrow, base,
                        base + R @ np.array([aFx * ARROW_SCALE, -aFz * ARROW_SCALE]),
                        np.hypot(aFx, aFz), 2.0, ppu)


class CorridorView:
    """Top-right panel: ATC/DTC corridor with smooth ramp + cross-fade."""

    def __init__(self, ax, atc, dtc):
        self.ax = ax
        ax.set_xlim(0, 45)
        ax.set_ylim(-30, 30)
        ax.set_xlabel(r"$V$, m/s", fontsize=13)
        ax.set_ylabel(r"$\theta$, deg", fontsize=13)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad((0, 0, 0, 0))
        # imshow blends as a single image -> clean partial-alpha fades, no seams
        self.atc_im = ax.imshow(
            np.ma.masked_invalid(atc["acc"].T), origin="lower", cmap=cmap,
            extent=[atc["VT"][0], atc["VT"][-1], atc["thc"][0], atc["thc"][-1]],
            aspect="auto", interpolation="bilinear", zorder=0)
        self.dtc_im = ax.imshow(
            np.ma.masked_invalid(-dtc["acc"].T), origin="lower", cmap=cmap,
            extent=[dtc["VT"][0], dtc["VT"][-1], dtc["thc"][0], dtc["thc"][-1]],
            aspect="auto", interpolation="bilinear", zorder=0)

        (self.traj,) = ax.plot([], [], "-", color="magenta", lw=2.5, zorder=4)
        (self.mark,) = ax.plot([], [], "o", mfc="magenta", mec="white",
                              mew=1.6, ms=13, zorder=6)

    def update(self, t, V_hist, th_hist, V_now, th_now):
        e = emphasis(t)
        w = crossfade(t)
        bright = 0.2 + 0.8 * e
        self.atc_im.set_alpha((1 - w) * bright)
        self.dtc_im.set_alpha(w * bright)
        self.ax.set_ylim(-30 + 20 * w, 30 - 20 * w)   # (-30,30) -> (-10,10)
        self.traj.set_data(np.clip(V_hist, 0, 45), th_hist)
        self.mark.set_data([min(V_now, 45.0)], [th_now])


class GlobalView:
    """Bottom strip: whole-journey side view + flown path + mode timeline."""

    MODE_FC = {0: "0.86", 1: "#8ecae6", 2: "0.80", 3: "#ffcaa8",
               4: "0.86", 5: "0.78"}   # transitions (1,3) saturated, others grey

    LABEL_FS = 9.5
    YMID = 7.0

    def __init__(self, ax, xbounds):
        self.ax = ax
        self.xb = xbounds                      # length-7 x boundaries
        ax.set_xlim(0, 1250)
        ax.set_ylim(0, 14.5)                    # headroom so the marker at alt 10
        ax.set_xlabel("range $x$, m", fontsize=12)   # doesn't overflow the strip
        ax.set_ylabel("altitude, m", fontsize=11)
        ax.set_yticks([0, 5, 10])

        bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
        # thin phases overlap in x (hover/landing ~0.5 m apart) -> stagger in y
        thin_y = {0: self.YMID, 4: 10.0, 5: 4.0}
        # takeoff/landing sit on the spines -> nudge just inside so text clears;
        # hover stays at its region centre
        thin_x = {0: 24.0, 4: 0.5 * (xbounds[4] + xbounds[5]), 5: 1226.0}
        self.labels = []
        for k, t0, t1 in MODE_SEGS:
            x0, x1 = xbounds[k], xbounds[k + 1]
            ax.axvspan(x0, x1, color=self.MODE_FC[k], zorder=0)
            if (x1 - x0) > 60:                 # wide: horizontal, top of region
                txt = ax.text(0.5 * (x0 + x1), 13.7, MODE_NAMES[k], ha="center",
                              va="top", fontsize=self.LABEL_FS, color="0.45",
                              zorder=5)
            else:                              # narrow: vertical, nudged inside
                txt = ax.text(thin_x[k], thin_y[k], MODE_NAMES[k],
                              ha="center", va="center", fontsize=self.LABEL_FS,
                              rotation=90, color="0.45", zorder=5, clip_on=False,
                              bbox=bbox)
            self.labels.append(txt)

        (self.path,) = ax.plot([], [], "-", color="0.4", lw=1.8, zorder=2)
        # position marker = the 3D LC62 snapshot (no rotation; attitude is the
        # left panel's job).  Sized to ~70% of the strip height.
        self._img = plt.imread(LC62_3D)
        self._oi = OffsetImage(self._img, zoom=0.3)
        self._ab = AnnotationBbox(self._oi, (0, 10), frameon=False, zorder=6,
                                  box_alignment=(0.5, 0.5), pad=0.0)
        ax.add_artist(self._ab)
        self._zoom_set = False

    def _fit_zoom(self):
        try:
            h_px = self.ax.get_window_extent().height
            self._oi.set_zoom(0.03 * h_px / self._img.shape[0])
            self._zoom_set = True
        except Exception:
            pass

    def update(self, x_hist, alt_hist, x_now, alt_now, theta, cur_mode):
        if not self._zoom_set:
            self._fit_zoom()
        self.path.set_data(x_hist, alt_hist)
        self._ab.xybox = (x_now, alt_now)
        self._ab.xy = (x_now, alt_now)
        for k, txt in enumerate(self.labels):
            txt.set_fontweight("bold" if k == cur_mode else "normal")
            txt.set_color("0.05" if k == cur_mode else "0.45")


def transition_kind(t):
    if T_FTC0 <= t < T_FTC1:
        return "FTC"
    if T_CRZ1 <= t < T_BTC1:
        return "BTC"
    return None


def traj_history(d, idx):
    t = d["t"][idx]
    kind = transition_kind(t)
    if kind == "FTC":                        # ATC -> actual response
        m = (d["t"] >= T_FTC0) & (d["t"] <= t)
        return d["V"][m], np.rad2deg(d["theta"][m])
    if kind == "BTC":                        # DTC -> optimal command
        m = (d["t"] >= T_CRZ1) & (d["t"] <= t)
        return d["Vd"][m], np.rad2deg(d["thetad"][m])
    return [], []


def mode_xbounds(d):
    bt = [0.0, T_FTC0, T_FTC1, T_CRZ1, T_BTC1, T_HOV1, 60.0]
    return [float(np.interp(tb, d["t"], d["xpos"])) for tb in bt]


def build_figure(d, atc, dtc):
    """Absolute-inch layout: the figure size is derived from the panels so there
    is no wasted margin on any side and the overall aspect is whatever the
    content needs (not forced to 16:9).  The corridor keeps a fixed, undistorted
    aspect; the global strip spans exactly the two top panels' combined width."""
    H_TOP, H_BOT = 4.5, 2.0                 # top-panel / bottom-strip heights [in]
    AR_CORR = 1.68                          # corridor width:height (undistorted)
    M_LAB, GAP, M_R = 0.66, 1.02, 0.14      # left labels / inter-panel / right
    T_TITLE, MID, B_LAB = 0.40, 1.15, 0.95  # top title / mid gap / bottom labels

    fig = plt.figure(figsize=(10, 8))       # provisional; resized below
    axL = fig.add_axes([0.1, 0.1, 0.3, 0.5])
    side = SideView(axL)                    # sets the side-view limits
    xr, yr = axL.get_xlim(), axL.get_ylim()
    w_left = (xr[1] - xr[0]) / (yr[1] - yr[0]) * H_TOP     # equal-aspect width
    w_corr = AR_CORR * H_TOP
    W = M_LAB + w_left + GAP + w_corr + M_R
    H = T_TITLE + H_TOP + MID + H_BOT + B_LAB
    fig.set_size_inches(W, H)

    def rect(x0, y0, w, h):
        return [x0 / W, y0 / H, w / W, h / H]

    y_top = B_LAB + H_BOT + MID
    axL.set_position(rect(M_LAB, y_top, w_left, H_TOP))
    axR = fig.add_axes(rect(M_LAB + w_left + GAP, y_top, w_corr, H_TOP))
    axG = fig.add_axes(rect(M_LAB, B_LAB, w_left + GAP + w_corr, H_BOT))
    corr = CorridorView(axR, atc, dtc)
    glob = GlobalView(axG, mode_xbounds(d))

    cx = (M_LAB + 0.5 * (w_left + GAP + w_corr)) / W
    txt = fig.text(cx, 0.16 / H, "", ha="center", va="center", fontsize=14)
    axL.set_title("LC62 side view  (pitch & forces)", fontsize=12)
    axR.set_title("Transition corridor  $(V,\\ \\theta)$", fontsize=12)
    axG.set_title("Global view  (mode timeline)", fontsize=12)
    return fig, side, corr, glob, txt


def render_frame(d, side, corr, glob, txt, idx):
    t = d["t"][idx]
    m = mode_of(t)
    side.update(d["theta"][idx], d["alt"][idx], d["rotor"][idx], d["pusher"][idx],
                d["aFx"][idx], d["aFz"][idx])
    Vh, thh = traj_history(d, idx)
    if t < T_SWAP:
        Vnow, thnow = d["V"][idx], np.rad2deg(d["theta"][idx])
    else:
        Vnow, thnow = d["Vd"][idx], np.rad2deg(d["thetad"][idx])
    corr.update(t, Vh, thh, Vnow, thnow)
    glob.update(d["xpos"][: idx + 1], d["alt"][: idx + 1],
                d["xpos"][idx], d["alt"][idx], d["theta"][idx], m)
    name = MODE_NAMES[m].replace(" ", r"\ ")
    txt.set_text(f"$t$ = {t:5.2f} s      Mode:  $\\bf{{{name}}}$")


def render_closeup():
    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    side = SideView(ax, ground=False)
    ax.set_xlim(-2.6, 2.4)
    ax.set_ylim(-1.1, 3.4)
    ax.set_ylabel("body $z$ (up), m", fontsize=12)
    ax.set_title("LC62 side view — geometry check ($\\theta=0$, illustrative)",
                 fontsize=12)
    side.update(0.0, 0.0, np.array([110.0, 95.0, 110.0]), 60.0, -75.0, -480.0)
    out = os.path.join(OUTDIR, "vehicle_closeup.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)
    plt.close(fig)


def render_previews(d, atc, dtc):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, side, corr, glob, txt = build_figure(d, atc, dtc)
    for tt in [3.0, 8.0, 12.0, 20.0, 35.0, 48.5, 55.0]:
        idx = int(round(tt / 0.01))
        render_frame(d, side, corr, glob, txt, idx)
        out = os.path.join(OUTDIR, f"frame_t{tt:04.1f}.png")
        fig.savefig(out, dpi=DPI)
        print("saved", out)
    plt.close(fig)


def render_mp4(d, atc, dtc):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, side, corr, glob, txt = build_figure(d, atc, dtc)
    frames = range(0, len(d["t"]), STEP)

    def _upd(idx):
        render_frame(d, side, corr, glob, txt, idx)
        return ()

    ani = animation.FuncAnimation(fig, _upd, frames=frames, blit=False)
    out = os.path.join(OUTDIR, "switching_animation.mp4")
    ani.save(out, writer="ffmpeg", fps=FPS, dpi=DPI, bitrate=6000)
    print("saved", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closeup", action="store_true")
    ap.add_argument("--mp4", action="store_true")
    args = ap.parse_args()
    if args.closeup:
        render_closeup()
        return
    d = load_data()
    atc = corridor(ATC_NPZ)
    dtc = corridor(DTC_NPZ, theta_pad_to=10.0)
    if args.mp4:
        render_mp4(d, atc, dtc)
    else:
        render_previews(d, atc, dtc)


if __name__ == "__main__":
    main()
