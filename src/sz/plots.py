"""Figures for the project (F1-F9).

Colors follow one fixed palette: a single-hue blue ramp for magnitudes,
a neutral gray for "no value", and ink colors for text and axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES = [SERIES_1, SERIES_2]
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def plot_regime_map(
    npz_path: Union[str, Path],
    cfg: Mapping[str, Any],
    out_path: Union[str, Path],
) -> Path:
    """T1.4 regime map: dominant frequency over (A, p), plus bifurcation diagrams.

    Left: spectral peak frequency of y for each (A, p); steady points
    (no oscillation) in neutral gray. Dashed line at the nominal A = 3.25.
    Right: one bifurcation diagram per p in cfg["bifurcation_p"]: maxima
    and minima of y against A (the steady value where y does not oscillate).
    """
    _style()
    d = np.load(npz_path)
    A, P = d["A"], d["p"]
    f_peak, steady = d["f_peak"], d["steady"]
    ext_A, ext_p, ext_y = d["ext_A"], d["ext_p"], d["ext_y"]
    bif_p = list(cfg["bifurcation_p"])

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(
        len(bif_p), 2, width_ratios=[1.15, 1], wspace=0.28, hspace=0.35,
        left=0.07, right=0.98, top=0.86, bottom=0.09,
    )

    # Left: frequency map
    ax = fig.add_subplot(gs[:, 0])
    dA = A[1] - A[0] if len(A) > 1 else 0.05
    dp = P[1] - P[0] if len(P) > 1 else 10.0
    extent = [A[0] - dA / 2, A[-1] + dA / 2, P[0] - dp / 2, P[-1] + dp / 2]
    ax.imshow(
        np.where(steady, 1.0, np.nan), origin="lower", aspect="auto",
        extent=extent, cmap=LinearSegmentedColormap.from_list("n", [NEUTRAL, NEUTRAL]),
        vmin=0, vmax=1, interpolation="nearest",
    )
    vmax = float(np.nanmax(np.where(steady, np.nan, f_peak))) if (~steady).any() else 1.0
    im = ax.imshow(
        np.where(steady, np.nan, f_peak), origin="lower", aspect="auto",
        extent=extent, cmap=SEQ_BLUE, vmin=0.0, vmax=np.ceil(vmax),
        interpolation="nearest",
    )
    ax.axvline(3.25, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(3.22, P[0], "nominal A = 3.25", color=INK_2, fontsize=8,
            va="bottom", ha="right")
    for pv in bif_p:
        ax.axhline(pv, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.set_xlabel("Excitatory gain A (mV)")
    ax.set_ylabel("Constant input p (Hz)")
    ax.set_title("Dominant frequency of y", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Peak frequency (Hz)", color=INK_2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(
        handles=[Patch(facecolor=NEUTRAL, edgecolor=AXIS, lw=0.5, label="Steady (no oscillation)")],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2,
    )

    # Right: bifurcation diagrams, shared A and y axes
    y_lo, y_hi = float(ext_y.min()), float(ext_y.max())
    pad = 0.05 * (y_hi - y_lo)
    axes = []
    for k, pv in enumerate(bif_p):
        axb = fig.add_subplot(gs[k, 1], sharex=axes[0] if axes else None)
        m = np.isclose(ext_p, pv)
        axb.scatter(ext_A[m], ext_y[m], s=2, color=SERIES_1, linewidths=0)
        axb.axvline(3.25, color=INK_2, lw=1, ls=(0, (4, 3)))
        axb.set_ylim(y_lo - pad, y_hi + pad)
        axb.set_xlim(extent[0], extent[1])
        axb.grid(axis="y", color=GRID, lw=0.6)
        axb.set_axisbelow(True)
        axb.text(0.01, 0.95, f"p = {pv:g} Hz", transform=axb.transAxes,
                 color=INK, fontsize=8, fontweight="bold", va="top")
        axb.set_ylabel("y (mV)")
        if k < len(bif_p) - 1:
            axb.tick_params(labelbottom=False)
        axes.append(axb)
    axes[0].set_title("Maxima and minima of y", loc="left")
    axes[-1].set_xlabel("Excitatory gain A (mV)")

    fig.suptitle(
        "Jansen-Rit regimes over gain and input (constant p, 10 s from rest, 2 s transient dropped)",
        x=0.07, ha="left", color=INK, fontsize=12, fontweight="bold",
    )
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_f1(
    npz_path: Union[str, Path],
    cfg: Mapping[str, Any],
    out_path: Union[str, Path],
) -> Path:
    """Figure F1: gain sweep at the chosen input level, both regimes.

    Left: bifurcation diagram (maxima and minima of y against A) at p = f1_p,
    with the example A values marked. Middle: example traces of y at 256 Hz.
    Right: Welch spectra of the example traces, with the seizure band
    (cfg["seizure_band_Hz"]) shaded.
    """
    _style()
    d = np.load(npz_path)
    p = float(d["p"])
    A_ex = d["example_A"]
    fs = float(d["fs_obs"])
    y_obs, f, psd = d["y_obs"], d["f"], d["psd"]
    A, ext_A, ext_y = d["A"], d["ext_A"], d["ext_y"]
    n_show = int(round(cfg["f1_trace_s"] * fs))
    t = np.arange(n_show) / fs
    labels = [f"A = {a:g} mV" for a in A_ex]

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(
        len(A_ex), 3, width_ratios=[1, 1.15, 1], wspace=0.3, hspace=0.25,
        left=0.06, right=0.98, top=0.8, bottom=0.12,
    )
    y_lo = min(float(ext_y.min()), float(y_obs.min()))
    y_hi = max(float(ext_y.max()), float(y_obs.max()))
    pad = 0.05 * (y_hi - y_lo)

    # Left: bifurcation diagram
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(ext_A, ext_y, s=3, color=INK_2, linewidths=0)
    for a, c, lab in zip(A_ex, SERIES, labels):
        ax.axvline(a, color=c, lw=2)
        ax.text(a + 0.04, y_hi + pad, lab, color=INK_2, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(y_lo - pad, y_hi + 2.5 * pad)
    ax.set_xlim(A[0], A[-1])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Excitatory gain A (mV)")
    ax.set_ylabel("y (mV)")
    ax.set_title("Maxima and minima of y", loc="left")

    # Middle: example traces, shared y-axis
    tr_axes = []
    for k, (c, lab) in enumerate(zip(SERIES, labels)):
        axt = fig.add_subplot(gs[k, 1], sharex=tr_axes[0] if tr_axes else None)
        axt.plot(t, y_obs[:n_show, k], color=c, lw=1.5)
        axt.set_ylim(y_lo - pad, y_hi + pad)
        axt.grid(axis="y", color=GRID, lw=0.6)
        axt.set_axisbelow(True)
        axt.text(0.01, 0.95, lab, transform=axt.transAxes, color=INK,
                 fontsize=8, fontweight="bold", va="top")
        axt.set_ylabel("y (mV)")
        if k < len(A_ex) - 1:
            axt.tick_params(labelbottom=False)
        tr_axes.append(axt)
    tr_axes[0].set_title(f"Example traces ({cfg['f1_trace_s']:g} s at 256 Hz)", loc="left")
    tr_axes[-1].set_xlabel("Time after transient (s)")

    # Right: spectra
    axs = fig.add_subplot(gs[:, 2])
    b_lo, b_hi = (float(v) for v in cfg["seizure_band_Hz"])
    axs.axvspan(b_lo, b_hi, color=NEUTRAL, zorder=0)
    axs.text(b_hi + 0.3, 0.98, f"seizure band ({b_lo:g}-{b_hi:g} Hz)",
             transform=axs.get_xaxis_transform(),
             color=INK_2, fontsize=8, ha="left", va="top")
    f_max = 20.0
    m = f <= f_max
    for k, (c, lab) in enumerate(zip(SERIES, labels)):
        axs.plot(f[m], psd[k, m], color=c, lw=2, label=lab)
    axs.set_xlim(0, f_max)
    axs.set_ylim(bottom=0)
    axs.grid(axis="y", color=GRID, lw=0.6)
    axs.set_axisbelow(True)
    axs.set_xlabel("Frequency (Hz)")
    axs.set_ylabel("Power (mV²/Hz)")
    axs.set_title("Welch spectrum of y", loc="left")
    axs.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

    fig.suptitle(
        f"F1. Gain sweep at p = {p:g} Hz: healthy (A = {A_ex[0]:g}) and seizure "
        f"(A = {A_ex[-1]:g}) regimes",
        x=0.06, ha="left", color=INK, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.06, 0.885,
        "Constant input, 10 s from rest, first 2 s dropped. Without input noise the "
        "healthy model sits at a steady state, so its trace is flat and its spectrum is zero.",
        color=INK_2, fontsize=9, ha="left",
    )
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Ordinal blue steps (light -> dark) for ordered series such as noise levels
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
 
 
def plot_noise_sweep(
    npz_path: Union[str, Path],
    cfg: Mapping[str, Any],
    out_path: Union[str, Path],
) -> Path:
    """T1.5 noise sweep: seizure-window fraction and seizure-band power vs A per sigma,
    plus example traces at the dataset A values.
 
    Lines are means over seeds; the shaded band on the left panel spans the
    min and max over seeds. Darker blue = larger sigma; sigma = 0 is gray dashed.
    """
    _style()
    d = np.load(npz_path)
    A, S = d["A"], d["sigma"]
    frac, band = d["frac"], d["band"]
    ex_A, ex_y = d["example_A"], d["example_y"]
    fs = float(d["fs_obs"])
    ex_s = float(d["example_sigma"])
 
    order = np.argsort(S)                         # small -> large sigma
    nonzero = [i for i in order if S[i] > 0]
    colors = {i: ORDINAL_BLUE[k] for k, i in enumerate(nonzero)}
 
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(
        len(ex_A), 3, width_ratios=[1, 1, 1.05], wspace=0.32, hspace=0.3,
        left=0.06, right=0.98, top=0.82, bottom=0.09,
    )
 
    def style_for(i):
        if S[i] == 0:
            return dict(color=MUTED, lw=1.5, ls=(0, (4, 3)))
        return dict(color=colors[i], lw=2)
 
    ax1 = fig.add_subplot(gs[:, 0])
    for i in order:
        ax1.plot(A, frac[i].mean(axis=1), label=f"σ = {S[i]:g}", **style_for(i))
        if S[i] > 0:
            ax1.fill_between(A, frac[i].min(axis=1), frac[i].max(axis=1),
                             color=colors[i], alpha=0.15, lw=0)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(A[0], A[-1])
    ax1.set_xlabel("Excitatory gain A (mV)")
    ax1.set_ylabel(f"Share of {cfg['window_s']:g} s windows with a spike")
    ax1.set_title("Seizure-window fraction", loc="left")
 
    ax2 = fig.add_subplot(gs[:, 1], sharex=ax1)
    for i in nonzero:  # sigma = 0 has ~zero power below onset, off a log axis
        ax2.plot(A, band[i].mean(axis=1), **style_for(i))
    ax2.set_yscale("log")
    ax2.set_xlabel("Excitatory gain A (mV)")
    b_lo, b_hi = cfg["seizure_band_Hz"]
    ax2.set_ylabel(f"Power in {b_lo:g}-{b_hi:g} Hz (mV², log)")
    ax2.set_title("Seizure-band power", loc="left")
 
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax1.legend(loc="upper left", title="Input noise (Hz·√s)", title_fontsize=8)
 
    n = ex_y.shape[1]
    t = np.arange(n) / fs
    y_lo, y_hi = float(ex_y.min()), float(ex_y.max())
    pad = 0.06 * (y_hi - y_lo)
    axes = []
    for k, a in enumerate(ex_A):
        axt = fig.add_subplot(gs[k, 2], sharex=axes[0] if axes else None)
        axt.plot(t, ex_y[k], color=SERIES_1, lw=1)
        axt.axhline(float(cfg["spike_threshold_mV"]), color=MUTED, lw=0.8, ls=(0, (2, 2)))
        axt.set_ylim(y_lo - pad, y_hi + pad)
        axt.grid(axis="y", color=GRID, lw=0.6)
        axt.set_axisbelow(True)
        axt.text(0.01, 0.95, f"A = {a:g} mV", transform=axt.transAxes, color=INK,
                 fontsize=8, fontweight="bold", va="top",
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        axt.set_ylabel("y (mV)")
        if k < len(ex_A) - 1:
            axt.tick_params(labelbottom=False)
        axes.append(axt)
    axes[0].set_title(f"Dataset A values at σ = {ex_s:g}", loc="left")
    axes[-1].set_xlabel("Time after transient (s)")
 
    fig.suptitle(
        f"Input-noise sweep at p̄ = {cfg['p_mean']:g} Hz: where the seizure regime starts "
        f"for each σ",
        x=0.06, ha="left", color=INK, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.06, 0.88,
        f"Euler–Maruyama at 2048 Hz, {cfg['duration_s']:g} s per run after a "
        f"{cfg['transient_s']:g} s transient, {frac.shape[2]} seeds per point; "
        "lines are seed means, shading on the left spans the seeds.\n"
        f"Right: dotted line = {cfg['spike_threshold_mV']:g} mV spike threshold. "
        "σ = 0 is left off the power panel (no power below onset).",
        color=INK_2, fontsize=9, ha="left",
    )
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
 