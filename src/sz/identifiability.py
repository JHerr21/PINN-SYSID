"""Identifiability analysis: fit-error landscapes (T2.1) and local sensitivity
with Fisher information (T2.2), and bounds for estimated subsets (T2.3).

Both analyses compare model output to a reference record in short windows.
Each window starts from the record's true state at its first observation
and is driven by the record's realized input p(t) (plan decision D2), so
errors do not build up across the whole record. This matches what the
identification methods see: the PINN fits each window with its own network
(decision D4) and the UKF corrects its state with every observation.

Window bookkeeping (see sz.data): in a record,
    x[m]   state at observation m, i.e. after 2048 Hz step n_drop + 1 + 8m
    p[j]   realized input held over 2048 Hz step n_drop + j
so a window that starts at observation m begins at x[m], is driven by
p[1 + 8m : 1 + 8(m + W)], and predicts observations m + 1 ... m + W.
Windows are back to back from m = 0. Integration uses sz.integrate.euler,
the same scheme that made the data, so the true parameters reproduce the
record's states exactly.

The reference output is the clean EEG output y = y1 - y2 of the true states
at the observation times.

Fit error (T2.1):
    NMSE = mean over windows and samples of (y_model - y_ref)^2 / var(y_ref)

Sensitivity (T2.2): J[i, j] = d y_i / d log(theta_j), computed with
torch.func.jacfwd. Log parameters make the columns relative sensitivities
(change in y per relative change in theta_j), so parameters with different
units are comparable. With white observation noise of standard deviation
sigma_obs, the Fisher information is
    F = J^T J / sigma_obs^2
Its inverse is the Cramer-Rao lower bound on the covariance of the
log-parameter estimates: sqrt(diag(F^-1)) is the smallest achievable
relative standard deviation, and normalizing F^-1 gives the parameter
correlation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from torch.func import jacfwd

from sz.data import load_dataset
from sz.integrate import DECIM, FS_OBS, euler
from sz.model import DEFAULT_PARAMS, output


def load_reference(path: Union[str, Path], record: int = 0) -> Dict[str, Any]:
    """Load one record of a saved dataset as the reference.

    Returns a dict with
        x          (n_obs, 6) float64 tensor, true states at the observation times
        p          (n_sim,) float64 tensor, realized input at 2048 Hz
        theta      dict of the true parameter values (floats)
        noise_sd   observation-noise standard deviation of the record, mV
        meta       the dataset's metadata
    """
    arrays, meta = load_dataset(path)
    n_rec = arrays["x"].shape[0]
    if not 0 <= record < n_rec:
        raise IndexError(f"record {record} out of range: dataset has {n_rec} record(s)")
    return {
        "x": torch.as_tensor(arrays["x"][record], dtype=torch.float64),
        "p": torch.as_tensor(arrays["p"][record], dtype=torch.float64),
        "theta": {k: float(v) for k, v in meta["theta_true"].items()},
        "noise_sd": float(meta["noise_sd_mV"][record]),
        "meta": meta,
    }


def make_windows(
    x: torch.Tensor, p: torch.Tensor, window_obs: int
) -> Dict[str, torch.Tensor]:
    """Split a record into back-to-back windows of window_obs observations.

    Args:
        x: (n_obs, 6) true states at the observation times.
        p: (n_sim,) realized input at 2048 Hz.
        window_obs: observations per window (128 = 0.5 s at 256 Hz).

    Returns a dict with
        x0      (n_win, 6)                   true state at each window start
        p       (window_obs * 8, n_win)      input over each window
        y_ref   (window_obs, n_win)          clean output y1 - y2 the window predicts
        start   (n_win,)                     index of each window's start observation
    The number of windows is the largest n_win with
    1 + 8 * n_win * window_obs <= n_sim and n_win * window_obs < n_obs.
    """
    window_obs = int(window_obs)
    if window_obs < 1:
        raise ValueError("window_obs must be at least 1")
    n_obs, n_sim = x.shape[0], p.shape[0]
    n_win = min((n_sim - 1) // (DECIM * window_obs), (n_obs - 1) // window_obs)
    if n_win < 1:
        raise ValueError(f"record too short for one window of {window_obs} observations")
    start = torch.arange(n_win) * window_obs
    n_steps = DECIM * window_obs
    p_win = torch.stack([p[1 + DECIM * m: 1 + DECIM * m + n_steps] for m in start.tolist()], dim=1)
    y_ref = torch.stack([output(x[m + 1: m + 1 + window_obs]) for m in start.tolist()], dim=1)
    return {"x0": x[start], "p": p_win, "y_ref": y_ref, "start": start}


def simulate_windows(
    x0: torch.Tensor,
    theta: Mapping[str, Any],
    p_win: torch.Tensor,
    batch_shape: Tuple[int, ...] = (),
) -> torch.Tensor:
    """Simulate every window from its start state; return the output at the
    observation times.

    Args:
        x0: (n_win, 6) start states.
        theta: parameters; each value a float or a tensor broadcastable to
            batch_shape + (1,) (one value per batch member, shared by all windows).
        p_win: (n_steps, n_win) input, n_steps a multiple of 8.
        batch_shape: leading batch shape of the parameter sets.

    Returns:
        (window_obs,) + batch_shape + (n_win,) output y1 - y2 at the
        observation times of each window.
    """
    n_steps = p_win.shape[0]
    if n_steps % DECIM:
        raise ValueError("p_win must have a multiple of 8 steps per window")
    x_start = x0.expand(tuple(batch_shape) + tuple(x0.shape)).clone()
    X = euler(x_start, theta, p_win, n_steps=n_steps)
    return output(X[DECIM::DECIM])


def landscape(
    windows: Mapping[str, torch.Tensor],
    theta_true: Mapping[str, float],
    names: Tuple[str, str],
    values: Tuple[Sequence[float], Sequence[float]],
    max_batch: int = 200,
) -> np.ndarray:
    """Fit-error landscape over a grid of two parameters (T2.1).

    Every other parameter is held at its true value.

    Args:
        windows: output of make_windows.
        theta_true: true parameter values.
        names: the two parameter names, e.g. ("A", "B").
        values: grid values for each of the two parameters.
        max_batch: grid points simulated together (memory bound).

    Returns:
        NMSE array of shape (len(values[0]), len(values[1])):
        nmse[i, j] is the error at names[0] = values[0][i], names[1] = values[1][j].
    """
    n1, n2 = names
    if n1 == n2:
        raise ValueError("the two parameters must differ")
    for n in names:
        if n not in DEFAULT_PARAMS:
            raise KeyError(f"Unknown parameter: {n!r}")
    v1 = torch.as_tensor(np.asarray(values[0], dtype=np.float64))
    v2 = torch.as_tensor(np.asarray(values[1], dtype=np.float64))
    G1, G2 = torch.meshgrid(v1, v2, indexing="ij")
    g1, g2 = G1.reshape(-1), G2.reshape(-1)

    y_ref = windows["y_ref"]
    var_ref = float(y_ref.var())
    out = torch.empty(g1.numel(), dtype=torch.float64)
    for s in range(0, g1.numel(), int(max_batch)):
        sl = slice(s, s + int(max_batch))
        b = g1[sl].numel()
        theta = dict(theta_true)
        theta[n1] = g1[sl].unsqueeze(-1)              # (b, 1): shared by all windows
        theta[n2] = g2[sl].unsqueeze(-1)
        y = simulate_windows(windows["x0"], theta, windows["p"], batch_shape=(b,))
        err = (y - y_ref.unsqueeze(1)) ** 2           # (W, b, n_win)
        out[sl] = err.mean(dim=(0, 2)) / var_ref
    return out.reshape(len(v1), len(v2)).numpy()


def output_sensitivity(
    windows: Mapping[str, torch.Tensor],
    theta_true: Mapping[str, float],
    names: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Relative output sensitivities with torch.func.jacfwd (T2.2).

    Args:
        windows: output of make_windows.
        theta_true: true parameter values (the point of linearization).
        names: the parameters to differentiate with respect to; the others
            stay at their true values.

    Returns:
        (y, J): y is the model output at the true parameters, the
        (window_obs, n_win) output of simulate_windows flattened to (n,)
        with n = window_obs * n_win; J is (n, len(names)) with
        J[i, j] = d y_i / d log(theta_j).
    """
    names = list(names)
    for n in names:
        if n not in DEFAULT_PARAMS:
            raise KeyError(f"Unknown parameter: {n!r}")
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    log_theta0 = torch.log(torch.tensor([float(theta_true[n]) for n in names],
                                        dtype=torch.float64))

    def f(log_theta: torch.Tensor) -> torch.Tensor:
        theta = dict(theta_true)
        for k, n in enumerate(names):
            theta[n] = torch.exp(log_theta[k])
        return simulate_windows(windows["x0"], theta, windows["p"]).reshape(-1)

    y = f(log_theta0)
    J = jacfwd(f)(log_theta0)
    return y, J


def fisher_information(J: torch.Tensor, noise_sd: float) -> torch.Tensor:
    """F = J^T J / noise_sd^2 for white Gaussian observation noise."""
    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive")
    return J.T @ J / float(noise_sd) ** 2


def fisher_summary(F: torch.Tensor) -> Dict[str, np.ndarray]:
    """Eigen-decomposition of F and the Cramer-Rao covariance it implies.

    Returns a dict with
        eigvals   (k,) eigenvalues, largest first
        eigvecs   (k, k) unit eigenvectors as columns, same order; each
                  column's sign is set so its largest-magnitude entry is positive
        cond      largest / smallest eigenvalue
        cov       F^-1, the lower bound on the log-parameter covariance
        rel_sd    sqrt(diag(F^-1)), lower bound on each relative standard deviation
        corr      parameter correlation matrix from F^-1
    Raises ValueError if F is not positive definite.
    """
    F = 0.5 * (F + F.T)
    w, V = torch.linalg.eigh(F)
    if float(w.min()) <= 0.0:
        raise ValueError(
            f"Fisher information is not positive definite (smallest eigenvalue {float(w.min()):.3g}); "
            "at least one parameter combination has no effect on the output"
        )
    order = torch.argsort(w, descending=True)
    w, V = w[order], V[:, order]
    idx = V.abs().argmax(dim=0)
    V = V * torch.sign(V[idx, torch.arange(V.shape[1])])
    cov = (V / w) @ V.T
    sd = torch.sqrt(torch.diag(cov))
    corr = cov / sd[:, None] / sd[None, :]
    return {
        "eigvals": w.numpy(),
        "eigvecs": V.numpy(),
        "cond": float(w[0] / w[-1]),
        "cov": cov.numpy(),
        "rel_sd": sd.numpy(),
        "corr": corr.numpy(),
    }


def relative_grid(center: float, rel_range: float, n: int) -> np.ndarray:
    """n values spaced evenly over center * (1 - rel_range) ... center * (1 + rel_range).

    n must be odd so the center itself is a grid point.
    """
    n = int(n)
    if n < 3 or n % 2 == 0:
        raise ValueError("n must be an odd number >= 3 so the true value is on the grid")
    return float(center) * (1.0 + float(rel_range) * np.linspace(-1.0, 1.0, n))


def window_obs_from_seconds(window_s: float) -> int:
    """Observations per window for a window length in seconds (256 Hz)."""
    w = window_s * FS_OBS
    if abs(w - round(w)) > 1e-9:
        raise ValueError("window_s must be a multiple of 1/256 s")
    return int(round(w))


def subset_summary(
    F: Union[torch.Tensor, np.ndarray],
    names: Sequence[str],
    subset: Sequence[str],
) -> Dict[str, Any]:
    """Cramer-Rao bounds when only `subset` is estimated (T2.3).

    The parameters not in the subset are held at their true values, so the
    bound comes from the subset's own block of F: cov = (F_SS)^-1, not the
    subset's block of F^-1 (which would let the other parameters vary too).

    Args:
        F: (k, k) Fisher information for the log parameters `names`.
        names: parameter names in the order of F's rows.
        subset: the estimated parameters (at least two, all in names).

    Returns a dict with
        rel_sd     {name: lower bound on its relative standard deviation}
        corr       (m, m) correlation matrix of the subset, in subset order
        max_corr   largest |off-diagonal correlation|
        cond       condition number of F_SS
    """
    F = np.asarray(F, dtype=np.float64)
    names = list(names)
    subset = list(subset)
    if len(subset) < 2:
        raise ValueError("subset needs at least two parameters")
    if len(set(subset)) != len(subset):
        raise ValueError("subset names must be unique")
    missing = [n for n in subset if n not in names]
    if missing:
        raise KeyError(f"not in names: {missing}")
    idx = [names.index(n) for n in subset]
    F_ss = F[np.ix_(idx, idx)]
    w = np.linalg.eigvalsh(F_ss)
    if w.min() <= 0.0:
        raise ValueError(f"F block for {subset} is not positive definite")
    cov = np.linalg.inv(F_ss)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    off = np.abs(corr[~np.eye(len(subset), dtype=bool)])
    return {
        "rel_sd": dict(zip(subset, sd.tolist())),
        "corr": corr,
        "max_corr": float(off.max()),
        "cond": float(w.max() / w.min()),
    }
