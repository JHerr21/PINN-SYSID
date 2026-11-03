"""T1.4: regime map over excitatory gain A and constant input p.

For each (A, p) on the configured grid, simulates the Jansen-Rit model with
constant input p from the zero state (RK4 at 2048 Hz, many grid points per
batch), drops the transient, and records:
    - peak-to-peak amplitude of y = y1 - y2 at 2048 Hz
    - the local maxima and minima of y at 2048 Hz (bifurcation diagrams)
    - the Welch spectrum of y decimated to 256 Hz
    - the spectral peak frequency and the half-power band around it

Writes to a new run folder under results/:
    config.yaml, meta.json         (from sz.runs.make_run_dir)
    gain_sweep.npz                 arrays for re-plotting
    metrics.json                   per-(A, p) summary table
    regime_map.png                 frequency map + bifurcation diagrams
    f1.npz, F1.png                 figure F1: bifurcation diagram, example traces
                                   and spectra at the chosen input level f1_p

Usage:
    python experiments/01_simulator.py --config configs/gain_sweep.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy import signal

from sz.config import load_config
from sz.integrate import FS_OBS, FS_SIM, decimate, rk4
from sz.model import N_STATES, make_params, output
from sz.plots import plot_f1, plot_regime_map
from sz.runs import REPO_ROOT, make_run_dir, set_seed

MAX_BATCH = 320  # grid points simulated together (~300 MB of states at 10 s)


def grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


def extrema(y: np.ndarray):
    """Values of the local maxima and minima of a 1-D signal."""
    i_max = signal.argrelextrema(y, np.greater)[0]
    i_min = signal.argrelextrema(y, np.less)[0]
    return y[i_max], y[i_min]


def half_power_band(f: np.ndarray, pxx: np.ndarray):
    """Peak frequency and the contiguous band around it where power >= peak / 2.

    The 0 Hz bin is excluded from the peak search.
    """
    k = 1 + int(np.argmax(pxx[1:]))
    half = pxx[k] / 2.0
    lo = k
    while lo > 1 and pxx[lo - 1] >= half:
        lo -= 1
    hi = k
    while hi < len(pxx) - 1 and pxx[hi + 1] >= half:
        hi += 1
    return float(f[k]), float(f[lo]), float(f[hi])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "gain_sweep.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir = make_run_dir(cfg)
    torch.set_default_dtype(torch.float64)

    A = grid(cfg["A_min"], cfg["A_max"], cfg["A_step"])
    P = grid(cfg["p_min"], cfg["p_max"], cfg["p_step"])
    n_A, n_p = len(A), len(P)
    AA, PP = np.meshgrid(A, P)                 # (n_p, n_A)
    A_flat, P_flat = AA.ravel(), PP.ravel()
    n_pts = A_flat.size

    dt = 1.0 / FS_SIM
    n_steps = int(round(cfg["duration_s"] * FS_SIM))
    drop_sim = int(round(cfg["transient_s"] * FS_SIM))
    drop_obs = int(round(cfg["transient_s"] * FS_OBS))
    tol = cfg["steady_tol_mV"]

    f = None
    psd = None
    p2p = np.zeros(n_pts)
    f_peak = np.zeros(n_pts)
    f_lo = np.zeros(n_pts)
    f_hi = np.zeros(n_pts)
    ext_A, ext_p, ext_y = [], [], []

    print(f"Simulating {n_A} A x {n_p} p = {n_pts} points, {cfg['duration_s']} s each")
    t0 = time.time()
    for start in range(0, n_pts, MAX_BATCH):
        idx = np.arange(start, min(start + MAX_BATCH, n_pts))
        theta = make_params(A=torch.tensor(A_flat[idx]))
        p_steps = torch.tensor(P_flat[idx]).expand(n_steps, -1)
        x0 = torch.zeros(len(idx), N_STATES)
        X = rk4(x0, theta, p_steps, dt=dt)               # (n_steps + 1, b, 6)
        y_sim = output(X).numpy()
        del X

        y_obs = decimate(y_sim[1:]).numpy()[drop_obs:]   # (n_obs, b)
        y_sim = y_sim[1 + drop_sim:]
        f_b, psd_b = signal.welch(
            y_obs, fs=FS_OBS, nperseg=cfg["welch_nperseg"], axis=0, detrend="constant"
        )
        if psd is None:
            f = f_b
            psd = np.zeros((n_pts, len(f)))
        psd[idx] = psd_b.T

        for j, i in enumerate(idx):
            y = y_sim[:, j]
            p2p[i] = y.max() - y.min()
            f_peak[i], f_lo[i], f_hi[i] = half_power_band(f, psd[i])
            if p2p[i] < tol:
                vals = np.array([y.mean()])
            else:
                y_max, y_min = extrema(y)
                vals = np.concatenate([y_max, y_min])
            ext_A += [A_flat[i]] * len(vals)
            ext_p += [P_flat[i]] * len(vals)
            ext_y += list(vals)
        print(f"  {idx[-1] + 1}/{n_pts} done ({time.time() - t0:.0f} s)")

    steady = p2p < tol
    np.savez(
        run_dir / "gain_sweep.npz",
        A=A, p=P, f=f,
        psd=psd.reshape(n_p, n_A, -1),
        p2p=p2p.reshape(n_p, n_A),
        f_peak=f_peak.reshape(n_p, n_A),
        f_lo=f_lo.reshape(n_p, n_A),
        f_hi=f_hi.reshape(n_p, n_A),
        steady=steady.reshape(n_p, n_A),
        ext_A=np.array(ext_A), ext_p=np.array(ext_p), ext_y=np.array(ext_y),
    )

    rows = [
        {
            "A": float(A_flat[i]),
            "p": float(P_flat[i]),
            "steady": bool(steady[i]),
            "peak_to_peak_mV": float(p2p[i]),
            "peak_freq_Hz": None if steady[i] else float(f_peak[i]),
            "half_power_lo_Hz": None if steady[i] else float(f_lo[i]),
            "half_power_hi_Hz": None if steady[i] else float(f_hi[i]),
        }
        for i in range(n_pts)
    ]
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"per_point": rows}, fh, indent=2)

    plot_regime_map(run_dir / "gain_sweep.npz", cfg, run_dir / "regime_map.png")

    # Figure F1: example traces and spectra at the chosen input level
    p_f1 = float(cfg["f1_p"])
    A_ex = np.array(cfg["example_A"], dtype=float)
    X = rk4(
        torch.zeros(len(A_ex), N_STATES),
        make_params(A=torch.tensor(A_ex)),
        p_f1, dt=dt, n_steps=n_steps,
    )
    y_ex_sim = output(X).numpy()
    del X
    y_ex_obs = decimate(y_ex_sim[1:]).numpy()[drop_obs:]      # (n_obs, n_ex)
    f_ex, psd_ex = signal.welch(
        y_ex_obs, fs=FS_OBS, nperseg=cfg["welch_nperseg"], axis=0, detrend="constant"
    )
    ip = int(np.argmin(np.abs(P - p_f1)))
    np.savez(
        run_dir / "f1.npz",
        p=p_f1, example_A=A_ex, fs_obs=FS_OBS,
        y_obs=y_ex_obs, f=f_ex, psd=psd_ex.T,
        A=A, ext_A=np.array(ext_A)[np.isclose(ext_p, P[ip])],
        ext_y=np.array(ext_y)[np.isclose(ext_p, P[ip])],
    )
    plot_f1(run_dir / "f1.npz", cfg, run_dir / "F1.png")

    print(f"\nAt p = {P[ip]:g} Hz: spectral peak and half-power band per A")
    print(f"{'A (mV)':>7} {'p2p (mV)':>9} {'peak (Hz)':>10} {'half-power (Hz)':>17}")
    for ia, a in enumerate(A):
        k = ip * n_A + ia
        if steady[k]:
            print(f"{a:7.2f} {p2p[k]:9.3f} {'steady':>10}")
        else:
            print(f"{a:7.2f} {p2p[k]:9.3f} {f_peak[k]:10.2f} {f_lo[k]:8.2f} - {f_hi[k]:<6.2f}")

    print("\nPeak frequency (Hz) per (p, A); '.' = steady")
    shown = A[:: max(1, n_A // 12)]
    print("p \\ A " + "".join(f"{a:6.2f}" for a in shown))
    for ip, pv in enumerate(P):
        cells = []
        for a in shown:
            ia = int(np.argmin(np.abs(A - a)))
            k = ip * n_A + ia
            cells.append(f"{'.':>6}" if steady[k] else f"{f_peak[k]:6.1f}")
        print(f"{pv:5.0f} " + "".join(cells))
    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()