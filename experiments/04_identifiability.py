"""T2.1-T2.2: fit-error landscapes and local sensitivity / Fisher information.

Loads one reference record (configs/identifiability.yaml: by default the
nominal cell, A = 4.0, seed 0). The record is split into back-to-back
windows of window_s; every model run restarts from the record's true state
at each window start and is driven by the record's realized input (see
sz.identifiability for the bookkeeping).

T2.1: for each parameter pair, a n_grid x n_grid grid over
true value * (1 -/+ rel_range) on both axes, all other parameters at their
true values; the error is the NMSE of the clean output y = y1 - y2.

T2.2: J = d y / d log(theta) for sensitivity_params with
torch.func.jacfwd at the true parameters; Fisher information
F = J^T J / noise_sd^2 with the record's observation-noise SD; its
eigenvalues and eigenvectors, the Cramer-Rao relative SDs and the
parameter correlation matrix from F^-1.

Writes to a new run folder under results/:
    config.yaml, meta.json         (from sz.runs.make_run_dir)
    identifiability.npz            landscapes, J, F and its summary, for re-plotting
    metrics.json                   landscape summary, eigenvalues, eigenvectors,
                                   relative SDs, correlation matrix
    F2.png                         figure F2

Usage:
    python experiments/04_identifiability.py --config configs/identifiability.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sz.config import load_config
from sz.identifiability import (
    fisher_information,
    fisher_summary,
    landscape,
    load_reference,
    make_windows,
    output_sensitivity,
    relative_grid,
    window_obs_from_seconds,
)
from sz.plots import plot_f2
from sz.runs import REPO_ROOT, make_run_dir, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "identifiability.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    ref = load_reference(REPO_ROOT / cfg["dataset"], record=int(cfg["record"]))
    theta = ref["theta"]
    W = window_obs_from_seconds(float(cfg["window_s"]))
    win = make_windows(ref["x"], ref["p"], W)
    n_win = int(win["start"].numel())
    print(f"Reference {cfg['dataset']} record {cfg['record']}: A = {theta['A']:g} mV, "
          f"noise SD {ref['noise_sd']:.4g} mV; {n_win} windows of {cfg['window_s']:g} s")

    run_dir = make_run_dir(cfg)
    arrays = {
        "noise_sd": np.float64(ref["noise_sd"]),
        "window_start": win["start"].numpy(),
    }
    for name, v in theta.items():
        arrays[f"theta_true_{name}"] = np.float64(v)

    # T2.1: fit-error landscapes
    pairs = [tuple(str(n) for n in pr) for pr in cfg["pairs"]]
    arrays["pair_names"] = np.array(pairs)
    n_grid = int(cfg["n_grid"])
    rel = float(cfg["rel_range"])
    land_metrics = []
    for k, (n1, n2) in enumerate(pairs):
        v1 = relative_grid(theta[n1], rel, n_grid)
        v2 = relative_grid(theta[n2], rel, n_grid)
        t0 = time.time()
        E = landscape(win, theta, (n1, n2), (v1, v2), max_batch=int(cfg["max_batch"]))
        c = n_grid // 2
        i, j = np.unravel_index(int(np.argmin(E)), E.shape)
        print(f"  T2.1 ({n1}, {n2}): {n_grid}x{n_grid} grid in {time.time() - t0:.1f} s; "
              f"NMSE at truth {E[c, c]:.3g}, min {E[i, j]:.3g} at "
              f"{n1} = {v1[i]:.4g}, {n2} = {v2[j]:.4g}; max {E.max():.3g}")
        arrays[f"v1_{k}"] = v1
        arrays[f"v2_{k}"] = v2
        arrays[f"nmse_{k}"] = E
        arrays[f"true_{k}"] = np.array([theta[n1], theta[n2]])
        land_metrics.append({
            "pair": [n1, n2],
            "true": [theta[n1], theta[n2]],
            "nmse_at_true": float(E[c, c]),
            "nmse_min": float(E[i, j]),
            "argmin": [float(v1[i]), float(v2[j])],
            "nmse_max": float(E.max()),
        })

    # T2.2: sensitivity and Fisher information
    names = [str(n) for n in cfg["sensitivity_params"]]
    t0 = time.time()
    y, J = output_sensitivity(win, theta, names)
    F = fisher_information(J, ref["noise_sd"])
    S = fisher_summary(F)
    print(f"  T2.2: jacfwd over {len(names)} parameters, {J.shape[0]} outputs in "
          f"{time.time() - t0:.1f} s; condition number {S['cond']:.3g}")
    arrays.update({
        "sens_names": np.array(names),
        "y_sens": y.numpy(),
        "J": J.numpy(),
        "F": F.numpy(),
        "eigvals": S["eigvals"],
        "eigvecs": S["eigvecs"],
        "cov": S["cov"],
        "rel_sd": S["rel_sd"],
        "corr": S["corr"],
    })

    print("    eigenvalue   eigenvector (largest loadings)")
    for i, w in enumerate(S["eigvals"]):
        v = S["eigvecs"][:, i]
        top = np.argsort(-np.abs(v))[:3]
        print(f"    {w:10.4g}   " + ", ".join(f"{names[j]} {v[j]:+.2f}" for j in top))
    print("    Cramer-Rao relative SD (all listed parameters estimated together):")
    for n, s in zip(names, S["rel_sd"]):
        print(f"      {n:>3}: {100 * s:.3g}%")

    np.savez_compressed(run_dir / "identifiability.npz", **arrays)
    metrics = {
        "reference": {
            "dataset": cfg["dataset"],
            "record": int(cfg["record"]),
            "theta_true": theta,
            "noise_sd_mV": ref["noise_sd"],
            "window_s": float(cfg["window_s"]),
            "n_windows": n_win,
        },
        "landscapes": land_metrics,
        "sensitivity": {
            "params": names,
            "eigvals": S["eigvals"].tolist(),
            "eigvecs": {n: S["eigvecs"][j].tolist() for j, n in enumerate(names)},
            "condition_number": S["cond"],
            "rel_sd_crlb": dict(zip(names, S["rel_sd"].tolist())),
            "corr": {n: dict(zip(names, S["corr"][j].tolist())) for j, n in enumerate(names)},
        },
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_f2(run_dir / "identifiability.npz", cfg, run_dir / "F2.png")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
