"""Synthetic dataset generator: simulation, observation noise at a set SNR, save/load.

A clean record is simulated with Euler-Maruyama at 2048 Hz from the zero
state, with input p(t) = p_mean + sigma * white noise. The first
transient_s seconds are dropped. The record keeps:
    y_clean   EEG output y = y1 - y2, anti-alias filtered and decimated to 256 Hz
    x         true states at the 256 Hz observation times (every 8th 2048 Hz
              sample, not filtered)
    p         realized input at 2048 Hz over the recorded period
              (p[k] is held over the 2048 Hz step k)

Observation noise is white Gaussian with standard deviation chosen so that
    SNR_dB = 10 * log10(var(y_clean) / noise_sd^2)
over the record that is actually kept.

Every random draw has its own seed, so a record is the same whether it is
simulated alone or in a batch:
    sim seed    drives the input noise
    noise seed  drives the observation noise
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sz.integrate import DECIM, FS_OBS, FS_SIM, decimate, euler_maruyama
from sz.model import N_STATES, make_params, output


def input_noise(seeds: Sequence[int], n_steps: int) -> torch.Tensor:
    """Standard-normal input noise, one column per seed, shape (n_steps, len(seeds))."""
    cols = [
        torch.randn(n_steps, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(int(s)))
        for s in seeds
    ]
    return torch.stack(cols, dim=1)


def simulate_records(
    A: Sequence[float],
    sim_seeds: Sequence[int],
    p_mean: float,
    sigma: float,
    duration_s: float,
    transient_s: float,
    theta_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate clean records, one per (A[i], sim_seeds[i]), as one batch.

    Args:
        A: excitatory gain per record, mV.
        sim_seeds: input-noise seed per record (same length as A).
        p_mean: mean input, Hz.
        sigma: input noise intensity, Hz * sqrt(s).
        duration_s: recorded length after the transient, s.
        transient_s: length dropped at the start, s.
        theta_overrides: other parameters to change from the standard values
            (A is set per record and may not appear here).

    Returns:
        dict with
            y_clean  (n_rec, n_obs)       output at 256 Hz
            x        (n_rec, n_obs, 6)    true states at the 256 Hz times
            p        (n_rec, n_sim)       realized input at 2048 Hz
        where n_obs = duration_s * 256 and n_sim = duration_s * 2048.
    """
    A = [float(a) for a in A]
    if len(A) != len(sim_seeds):
        raise ValueError("A and sim_seeds must have the same length")
    overrides = dict(theta_overrides or {})
    if "A" in overrides:
        raise ValueError("Set A through the A argument, not theta_overrides")

    n_rec = len(A)
    n_drop = int(round(transient_s * FS_SIM))
    n_keep = int(round(duration_s * FS_SIM))
    if n_drop % DECIM or n_keep % DECIM:
        raise ValueError("transient_s and duration_s must be multiples of 1/256 s")
    n_steps = n_drop + n_keep

    theta = make_params(A=torch.tensor(A, dtype=torch.float64), **overrides)
    x0 = torch.zeros(n_rec, N_STATES, dtype=torch.float64)
    noise = input_noise(sim_seeds, n_steps)
    X, p_real = euler_maruyama(
        x0, theta, p_mean, sigma, n_steps=n_steps, noise=noise
    )                                                    # (n_steps + 1, n_rec, 6)

    y_sim = output(X[1:])                                # sample k = state after step k
    y_obs = decimate(y_sim)                              # (n_steps / 8, n_rec)
    y_clean = y_obs[n_drop // DECIM:].T.numpy()          # (n_rec, n_obs)

    # States at the observation times: every DECIM-th 2048 Hz sample, starting
    # at the first (the same instants that decimate keeps)
    x = X[1:][::DECIM][n_drop // DECIM:]                 # (n_obs, n_rec, 6)
    x = x.permute(1, 0, 2).numpy()

    p = p_real[n_drop:].T.numpy()                        # (n_rec, n_sim)
    return {"y_clean": np.ascontiguousarray(y_clean),
            "x": np.ascontiguousarray(x),
            "p": np.ascontiguousarray(p)}


def add_observation_noise(
    y_clean: np.ndarray, snr_db: float, seed: int
) -> Tuple[np.ndarray, float]:
    """Add white Gaussian noise to one record at the given SNR (dB).

    The noise is seed's standard-normal sequence scaled to the required
    standard deviation, so records that share a seed get the same noise
    pattern at every SNR and duration (common random numbers).

    Returns (y_noisy, noise_sd).
    """
    y_clean = np.asarray(y_clean, dtype=np.float64)
    var = float(np.var(y_clean))
    noise_sd = math.sqrt(var / (10.0 ** (snr_db / 10.0)))
    z = np.random.default_rng(int(seed)).standard_normal(y_clean.shape)
    return y_clean + noise_sd * z, noise_sd


def _dataset_files(path: Union[str, Path]) -> Tuple[Path, Path]:
    """The .npz and .json files for a dataset path given with or without an
    extension. Dots inside the name (e.g. A3.25_seed0) are kept."""
    path = Path(path)
    if path.suffix in (".npz", ".json"):
        path = path.with_suffix("")
    return path.parent / (path.name + ".npz"), path.parent / (path.name + ".json")


def save_dataset(
    path: Union[str, Path],
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> Path:
    """Save arrays to <path>.npz and metadata to <path>.json. Never overwrites."""
    npz, js = _dataset_files(path)
    for f in (npz, js):
        if f.exists():
            raise FileExistsError(f"Dataset file already exists: {f}")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    with js.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return npz


def load_dataset(path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a dataset saved by save_dataset. Returns (arrays, meta)."""
    npz, js = _dataset_files(path)
    with np.load(npz) as d:
        arrays = {k: d[k] for k in d.files}
    with js.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return arrays, meta
