"""T1.5: input-noise (sigma) sweep at the chosen mean input.

For each (A, sigma) on the configured grid and each of n_seeds input-noise
realizations, simulates the Jansen-Rit model with Euler-Maruyama
(p = p_mean + sigma * white noise) from the zero state, drops the
transient, and records from y = y1 - y2 at 256 Hz:
    - seizure-window fraction: share of window_s windows whose max(y)
      exceeds spike_threshold_mV
    - seizure-band power: Welch power summed over seizure_band_Hz

Writes to a new run folder under results/:
    config.yaml, meta.json         (from sz.runs.make_run_dir)
    noise_sweep.npz                arrays for re-plotting
    metrics.json                   per-(A, sigma) means over seeds
    noise_sweep.png                figure

Usage:
    python experiments/01_noise_sweep.py --config configs/noise_sweep.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import signal

from sz.config import load_config
from sz.data import simulate_records
from sz.integrate import FS_OBS
from sz.plots import plot_noise_sweep
from sz.runs import REPO_ROOT, make_run_dir, set_seed

MAX_BATCH = 160  # records simulated together (~400 MB of states at 22 s)


def grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "noise_sweep.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir = make_run_dir(cfg)

    A = grid(cfg["A_min"], cfg["A_max"], cfg["A_step"])
    S = np.array(cfg["sigma_values"], dtype=float)
    n_A, n_S, n_seed = len(A), len(S), int(cfg["n_seeds"])

    # One record per (sigma, A, seed); run seed = base seed + flat index
    SS, AA, KK = np.meshgrid(np.arange(n_S), np.arange(n_A), np.arange(n_seed), indexing="ij")
    s_idx, a_idx, k_idx = SS.ravel(), AA.ravel(), KK.ravel()
    run_seeds = int(cfg["seed"]) + np.arange(s_idx.size)
    n_rec = s_idx.size

    win = int(round(cfg["window_s"] * FS_OBS))
    b_lo, b_hi = (float(v) for v in cfg["seizure_band_Hz"])
    thr = float(cfg["spike_threshold_mV"])
    n_show = int(round(cfg["example_trace_s"] * FS_OBS))

    frac = np.zeros(n_rec)
    band = np.zeros(n_rec)
    examples = {}
    ex_A = [float(a) for a in cfg["example_A"]]
    ex_s = float(cfg["example_sigma"])

    print(f"Simulating {n_A} A x {n_S} sigma x {n_seed} seeds = {n_rec} records, "
          f"{cfg['duration_s']} s each")
    t0 = time.time()
    for sigma_i in range(n_S):
        rows = np.flatnonzero(s_idx == sigma_i)
        for start in range(0, rows.size, MAX_BATCH):
            idx = rows[start:start + MAX_BATCH]
            rec = simulate_records(
                A[a_idx[idx]], run_seeds[idx], cfg["p_mean"], S[sigma_i],
                cfg["duration_s"], cfg["transient_s"],
            )
            y = rec["y_clean"]                                   # (b, n_obs)
            n_win = y.shape[1] // win
            wmax = y[:, : n_win * win].reshape(len(idx), n_win, win).max(axis=2)
            frac[idx] = (wmax > thr).mean(axis=1)
            f, psd = signal.welch(y, fs=FS_OBS, nperseg=cfg["welch_nperseg"],
                                  axis=1, detrend="constant")
            m = (f >= b_lo) & (f <= b_hi)
            band[idx] = psd[:, m].sum(axis=1) * (f[1] - f[0])
            if np.isclose(S[sigma_i], ex_s):
                for j, i in enumerate(idx):
                    a = float(A[a_idx[i]])
                    if k_idx[i] == 0 and any(np.isclose(a, e) for e in ex_A):
                        examples[a] = y[j, :n_show]
        print(f"  sigma = {S[sigma_i]:.3f} done ({time.time() - t0:.0f} s)")

    frac = frac.reshape(n_S, n_A, n_seed)
    band = band.reshape(n_S, n_A, n_seed)
    ex_keys = sorted(examples)
    np.savez(
        run_dir / "noise_sweep.npz",
        A=A, sigma=S, frac=frac, band=band, fs_obs=FS_OBS,
        example_A=np.array(ex_keys), example_y=np.array([examples[a] for a in ex_keys]),
        example_sigma=ex_s,
    )

    rows = []
    for si, s in enumerate(S):
        for ai, a in enumerate(A):
            rows.append({
                "sigma": float(s), "A": float(a),
                "seizure_window_frac_mean": float(frac[si, ai].mean()),
                "seizure_window_frac_min": float(frac[si, ai].min()),
                "seizure_window_frac_max": float(frac[si, ai].max()),
                "seizure_band_power_mean_mV2": float(band[si, ai].mean()),
            })
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"per_point": rows}, fh, indent=2)

    plot_noise_sweep(run_dir / "noise_sweep.npz", cfg, run_dir / "noise_sweep.png")

    print("\nSeizure-window fraction (mean over seeds) per (sigma, A)")
    shown = A[::5]
    print("sigma \\ A " + "".join(f"{a:6.2f}" for a in shown))
    for si, s in enumerate(S):
        cells = [f"{frac[si, int(np.argmin(np.abs(A - a)))].mean():6.2f}" for a in shown]
        print(f"{s:9.3f} " + "".join(cells))
    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()
