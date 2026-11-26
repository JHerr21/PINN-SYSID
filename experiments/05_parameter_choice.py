"""T2.3: compare candidate sets of estimated parameters.

For every (cell, A) in configs/parameter_choice.yaml, loads the dataset
<data_dir>/<cell>/A<A>_seed<k> and computes the Fisher information F of the
log parameters exactly as in T2.2 (windows of window_s from the true states,
realized input, the record's own observation-noise SD). Then, for every
subset of subset_sizes parameters that contains must_include, holds the other
parameters at their true values and computes from F's block for the subset:
    - the Cramer-Rao lower bound on each estimated parameter's relative SD
    - the parameter correlations, and the largest |correlation|
    - the condition number of the block

Writes to a new run folder under results/:
    config.yaml, meta.json         (from sz.runs.make_run_dir)
    parameter_choice.npz           F for every (cell, A), for re-analysis
    metrics.json                   every subset, every (cell, A)
    subsets.md                     summary table, subsets sorted by their
                                   largest |correlation| over all A values

Usage:
    python experiments/05_parameter_choice.py --config configs/parameter_choice.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from sz.config import load_config
from sz.identifiability import (
    fisher_information,
    load_reference,
    make_windows,
    output_sensitivity,
    subset_summary,
    window_obs_from_seconds,
)
from sz.runs import REPO_ROOT, make_run_dir, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "parameter_choice.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    names = [str(n) for n in cfg["params"]]
    cells = [str(c) for c in cfg["cells"]]
    A_values = [float(a) for a in cfg["A_values"]]
    W = window_obs_from_seconds(float(cfg["window_s"]))
    data_dir = REPO_ROOT / cfg["data_dir"]
    k, rec = int(cfg["seed_index"]), int(cfg["record"])

    run_dir = make_run_dir(cfg)

    # Fisher information for every (cell, A)
    F_all = np.zeros((len(cells), len(A_values), len(names), len(names)))
    noise_sd = np.zeros((len(cells), len(A_values)))
    duration = np.zeros((len(cells), len(A_values)))
    for ci, cell in enumerate(cells):
        for ai, A in enumerate(A_values):
            path = data_dir / cell / f"A{A:.2f}_seed{k}"
            ref = load_reference(path, record=rec)
            if abs(ref["theta"]["A"] - A) > 1e-12:
                raise ValueError(f"{path}: theta_true A = {ref['theta']['A']}, expected {A}")
            win = make_windows(ref["x"], ref["p"], W)
            t0 = time.time()
            _, J = output_sensitivity(win, ref["theta"], names)
            F_all[ci, ai] = fisher_information(J, ref["noise_sd"]).numpy()
            noise_sd[ci, ai] = ref["noise_sd"]
            duration[ci, ai] = float(ref["meta"]["duration_s"])
            print(f"  F for {cell:<12} A = {A:.2f}: noise SD {ref['noise_sd']:.4g} mV, "
                  f"{win['start'].numel()} windows, {time.time() - t0:.1f} s")

    # Every subset that contains must_include
    must = [str(n) for n in cfg["must_include"]]
    subsets = [
        list(s)
        for m in cfg["subset_sizes"]
        for s in itertools.combinations(names, int(m))
        if all(n in s for n in must)
    ]
    results = []
    for S in subsets:
        per = []
        for ci, cell in enumerate(cells):
            for ai, A in enumerate(A_values):
                s = subset_summary(F_all[ci, ai], names, S)
                per.append({
                    "cell": cell,
                    "A": A,
                    "rel_sd": s["rel_sd"],
                    "max_rel_sd": max(s["rel_sd"].values()),
                    "max_corr": s["max_corr"],
                    "cond": s["cond"],
                    "corr": s["corr"].tolist(),
                })
        worst_sd = max(per, key=lambda r: r["max_rel_sd"])
        results.append({
            "subset": S,
            "max_corr_by_A": {f"{A:g}": max(r["max_corr"] for r in per if r["A"] == A)
                              for A in A_values},
            "max_corr": max(r["max_corr"] for r in per),
            "max_rel_sd": worst_sd["max_rel_sd"],
            "max_rel_sd_at": {"cell": worst_sd["cell"], "A": worst_sd["A"]},
            "per_cell_A": per,
        })
    results.sort(key=lambda r: (len(r["subset"]), r["max_corr"]))

    # Summary table
    head = ("| Subset | " + " | ".join(f"max abs corr, A = {A:g}" for A in A_values)
            + " | Worst bound on relative SD | Where |")
    rule = "|" + "---|" * (len(A_values) + 3)
    lines = [
        f"Cells: {', '.join(cells)}. Parameters not in a subset are fixed at their true values. "
        f"Correlations do not depend on the noise level, so the nominal and noisy cells give "
        f"the same values; each column shows the largest over all cells.",
        "",
        head,
        rule,
    ]
    for r in results:
        cols = [f"{r['max_corr_by_A'][f'{A:g}']:.3f}" for A in A_values]
        where = f"{r['max_rel_sd_at']['cell']}, A = {r['max_rel_sd_at']['A']:g}"
        lines.append(f"| {', '.join(r['subset'])} | " + " | ".join(cols)
                     + f" | {100 * r['max_rel_sd']:.2f}% | {where} |")
    (run_dir / "subsets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSubsets sorted by largest |correlation| over all A values and cells:")
    print(f"  {'subset':<10} " + " ".join(f"A={A:<5g}" for A in A_values)
          + "  worst rel SD (where)")
    for r in results:
        cols = " ".join(f"{r['max_corr_by_A'][f'{A:g}']:7.3f}" for A in A_values)
        print(f"  {'+'.join(r['subset']):<10} {cols}  {100 * r['max_rel_sd']:6.2f}% "
              f"({r['max_rel_sd_at']['cell']}, A = {r['max_rel_sd_at']['A']:g})")

    np.savez_compressed(
        run_dir / "parameter_choice.npz",
        F=F_all, noise_sd=noise_sd, duration_s=duration,
        cells=np.array(cells), A_values=np.array(A_values), params=np.array(names),
    )
    metrics = {
        "params": names,
        "cells": cells,
        "A_values": A_values,
        "noise_sd_mV": {c: dict(zip([f"{A:g}" for A in A_values], noise_sd[ci].tolist()))
                        for ci, c in enumerate(cells)},
        "subsets": results,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
