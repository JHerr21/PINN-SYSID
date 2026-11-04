"""T1.5: generate every synthetic dataset the H1 sweep needs.

H1 varies one factor at a time around a nominal condition. Only duration,
SNR and number of records change the data; the initialization-offset and
input-knowledge conditions reuse the nominal datasets. The data cells are:
    nominal                         nominal duration, SNR and record count
    duration_<d>s                   each non-nominal duration
    snr_<s>dB                       each non-nominal SNR
    records_<r>                     each non-nominal record count

For each true A value and seed index, max(n_records) clean records of
max(durations_s) are simulated once. Each cell takes the first n_records
records, keeps the first duration_s of each, and adds observation noise at
its SNR. Seeds (fixed formulas, also saved with every dataset):
    sim seed    = seed + 1000 * a_index + 10 * seed_index + record
    noise seed  = 500000 + sim seed

Output, one pair of files per (cell, A, seed index):
    data/synthetic/<cell>/A<A>_seed<k>.npz
        t        (n_obs,)               observation times, s
        y        (R, n_obs)             noisy observations, 256 Hz
        y_clean  (R, n_obs)             noise-free output, 256 Hz
        x        (R, n_obs, 6)          true states at the observation times
        p        (R, n_sim)             realized input at 2048 Hz
    data/synthetic/<cell>/A<A>_seed<k>.json
        true parameters, seeds, noise SD per record, settings, git commit

Usage:
    python experiments/02_datasets.py --config configs/datasets.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sz.config import load_config
from sz.data import add_observation_noise, save_dataset, simulate_records
from sz.integrate import FS_OBS, FS_SIM
from sz.model import make_params
from sz.runs import REPO_ROOT, git_commit, set_seed

MAX_BATCH = 100  # clean records simulated together


def sim_seed(base: int, a_index: int, seed_index: int, record: int) -> int:
    return int(base + 1000 * a_index + 10 * seed_index + record)


def noise_seed(s: int) -> int:
    return 500000 + int(s)


def data_cells(cfg):
    """One-factor-at-a-time cells that change the data: name -> (duration, snr, n_records)."""
    nom = cfg["nominal"]
    d0, s0, r0 = float(nom["duration_s"]), float(nom["snr_db"]), int(nom["n_records"])
    cells = {"nominal": (d0, s0, r0)}
    for d in cfg["durations_s"]:
        if float(d) != d0:
            cells[f"duration_{float(d):g}s"] = (float(d), s0, r0)
    for s in cfg["snr_db"]:
        if float(s) != s0:
            cells[f"snr_{float(s):g}dB"] = (d0, float(s), r0)
    for r in cfg["n_records"]:
        if int(r) != r0:
            cells[f"records_{int(r)}"] = (d0, s0, int(r))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "datasets.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_root = REPO_ROOT / cfg["out_dir"]
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(
            f"{out_root} already contains files. Move or delete it first; "
            "datasets are never overwritten."
        )

    A_vals = [float(a) for a in cfg["A_values"]]
    n_seed = int(cfg["n_seeds"])
    base = int(cfg["seed"])
    T_max = max(float(d) for d in cfg["durations_s"])
    R_max = max(int(r) for r in cfg["n_records"])
    cells = data_cells(cfg)
    commit = git_commit()

    # Simulate every clean record once at full length
    keys = [(ai, k, r) for ai in range(len(A_vals)) for k in range(n_seed) for r in range(R_max)]
    clean = {}
    t0 = time.time()
    print(f"Simulating {len(keys)} clean records of {T_max:g} s "
          f"(p = {cfg['p_mean']:g} Hz, sigma = {cfg['sigma']:g})")
    for start in range(0, len(keys), MAX_BATCH):
        chunk = keys[start:start + MAX_BATCH]
        rec = simulate_records(
            [A_vals[ai] for ai, _, _ in chunk],
            [sim_seed(base, ai, k, r) for ai, k, r in chunk],
            cfg["p_mean"], cfg["sigma"], T_max, cfg["transient_s"],
        )
        for j, key in enumerate(chunk):
            clean[key] = {name: arr[j] for name, arr in rec.items()}
        print(f"  {start + len(chunk)}/{len(keys)} done ({time.time() - t0:.0f} s)")

    # Build and save each dataset
    n_saved = 0
    for cell, (dur, snr, n_rec) in cells.items():
        n_obs = int(round(dur * FS_OBS))
        n_sim = int(round(dur * FS_SIM))
        for ai, a in enumerate(A_vals):
            for k in range(n_seed):
                y, y_clean, x, p, sds, s_seeds, n_seeds_ = [], [], [], [], [], [], []
                for r in range(n_rec):
                    c = clean[(ai, k, r)]
                    yc = c["y_clean"][:n_obs]
                    s = sim_seed(base, ai, k, r)
                    yn, sd = add_observation_noise(yc, snr, noise_seed(s))
                    y.append(yn)
                    y_clean.append(yc)
                    x.append(c["x"][:n_obs])
                    p.append(c["p"][:n_sim])
                    sds.append(sd)
                    s_seeds.append(s)
                    n_seeds_.append(noise_seed(s))
                arrays = {
                    "t": np.arange(n_obs) / FS_OBS,
                    "y": np.stack(y), "y_clean": np.stack(y_clean),
                    "x": np.stack(x), "p": np.stack(p),
                }
                meta = {
                    "cell": cell,
                    "A_true": a,
                    "theta_true": make_params(A=a),
                    "seed_index": k,
                    "sim_seeds": s_seeds,
                    "noise_seeds": n_seeds_,
                    "noise_sd_mV": sds,
                    "duration_s": dur,
                    "snr_db": snr,
                    "n_records": n_rec,
                    "p_mean_Hz": float(cfg["p_mean"]),
                    "sigma_Hz_sqrt_s": float(cfg["sigma"]),
                    "transient_s": float(cfg["transient_s"]),
                    "fs_obs_Hz": FS_OBS,
                    "fs_sim_Hz": FS_SIM,
                    "x0": "zeros",
                    "git_commit": commit,
                    "config": Path(args.config).name,
                }
                save_dataset(out_root / cell / f"A{a:.2f}_seed{k}", arrays, meta)
                n_saved += 1

    print(f"Saved {n_saved} datasets in {len(cells)} cells under {out_root}")
    for cell, (dur, snr, n_rec) in cells.items():
        print(f"  {cell:<14} {dur:g} s, {snr:g} dB, {n_rec} record(s)")


if __name__ == "__main__":
    main()
