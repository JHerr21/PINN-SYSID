"""Integrators for the Jansen-Rit model and decimation to the observation rate.

Simulation runs at FS_SIM = 2048 Hz and is decimated to FS_OBS = 256 Hz.

Inputs p(t) and u(t) are given as one value per step and held constant
within the step (zero-order hold).

Stochastic input is a Gaussian SDE: p(t) = p_mean + sigma * xi(t), with
xi white noise. Euler-Maruyama adds A * a * sigma * sqrt(dt) * N(0, 1) to
y4 each step. This equals an Euler step with the realized input
p_k = p_mean + sigma * N(0, 1) / sqrt(dt), which is returned so the
realized p(t) is known to the identification methods.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
from scipy import signal

from sz.model import vector_field

FS_SIM = 2048  # Hz, simulation rate
FS_OBS = 256   # Hz, observation and control rate
DECIM = FS_SIM // FS_OBS  # 8


def _as_steps(v: Any, n_steps: Optional[int], ref: torch.Tensor) -> torch.Tensor:
    """Convert an input to a tensor with one value per step (first axis)."""
    t = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    if t.ndim == 0:
        if n_steps is None:
            raise ValueError("n_steps is required when p and u are both scalars")
        t = t.expand(n_steps)
    return t


def _resolve_inputs(
    p: Any, u: Any, n_steps: Optional[int], x0: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    p_t = torch.as_tensor(p, dtype=x0.dtype, device=x0.device)
    u_t = torch.as_tensor(u, dtype=x0.dtype, device=x0.device)
    if n_steps is None:
        lengths = [t.shape[0] for t in (p_t, u_t) if t.ndim > 0]
        if not lengths:
            raise ValueError("n_steps is required when p and u are both scalars")
        n_steps = lengths[0]
    p_t = _as_steps(p_t, n_steps, x0)
    u_t = _as_steps(u_t, n_steps, x0)
    for name, t in (("p", p_t), ("u", u_t)):
        if t.shape[0] != n_steps:
            raise ValueError(f"{name} has {t.shape[0]} steps, expected {n_steps}")
    return p_t, u_t, n_steps


def rk4(
    x0: torch.Tensor,
    theta: Mapping[str, Any],
    p: Any,
    u: Any = 0.0,
    dt: float = 1.0 / FS_SIM,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    """Classical 4th-order Runge-Kutta, inputs held constant within each step.

    Args:
        x0: initial state, shape (..., 6).
        theta: parameters (see sz.model.make_params).
        p: external input in Hz, scalar or shape (n_steps, ...).
        u: stimulation in Hz, scalar or shape (n_steps, ...).
        dt: step size in s.
        n_steps: number of steps; needed only if p and u are both scalars.

    Returns:
        States, shape (n_steps + 1, ..., 6), including x0 at index 0.
    """
    p_t, u_t, n_steps = _resolve_inputs(p, u, n_steps, x0)
    x = x0
    states = [x]
    for k in range(n_steps):
        pk, uk = p_t[k], u_t[k]
        k1 = vector_field(x, theta, pk, uk)
        k2 = vector_field(x + 0.5 * dt * k1, theta, pk, uk)
        k3 = vector_field(x + 0.5 * dt * k2, theta, pk, uk)
        k4 = vector_field(x + dt * k3, theta, pk, uk)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(x)
    return torch.stack(states, dim=0)


def euler(
    x0: torch.Tensor,
    theta: Mapping[str, Any],
    p: Any,
    u: Any = 0.0,
    dt: float = 1.0 / FS_SIM,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    """Forward Euler, inputs held constant within each step.

    Same arguments and return shape as rk4.
    """
    p_t, u_t, n_steps = _resolve_inputs(p, u, n_steps, x0)
    x = x0
    states = [x]
    for k in range(n_steps):
        x = x + dt * vector_field(x, theta, p_t[k], u_t[k])
        states.append(x)
    return torch.stack(states, dim=0)


def euler_maruyama(
    x0: torch.Tensor,
    theta: Mapping[str, Any],
    p_mean: float,
    sigma: float,
    n_steps: int,
    u: Any = 0.0,
    dt: float = 1.0 / FS_SIM,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Euler-Maruyama for p(t) = p_mean + sigma * white noise.

    Args:
        x0: initial state, shape (6,).
        theta: parameters (see sz.model.make_params).
        p_mean: mean input in Hz.
        sigma: noise intensity in Hz * sqrt(s).
        n_steps: number of steps.
        u: stimulation in Hz, scalar or shape (n_steps,).
        dt: step size in s.
        generator: torch.Generator for the noise; None uses the global
            torch RNG (seeded by sz.runs.set_seed).

    Returns:
        (states, p_realized): states of shape (n_steps + 1, 6) including
        x0, and the realized input p_k of shape (n_steps,), held over
        step k.
    """
    noise = torch.randn(
        n_steps, dtype=x0.dtype, device=x0.device, generator=generator
    )
    p_realized = p_mean + sigma * noise / math.sqrt(dt)
    states = euler(x0, theta, p_realized, u, dt=dt, n_steps=n_steps)
    return states, p_realized


def decimate(y: Any, factor: int = DECIM) -> torch.Tensor:
    """Anti-alias filter and downsample along the first (time) axis.

    Uses a zero-phase FIR low-pass filter (scipy.signal.decimate), then
    keeps every factor-th sample. The default factor takes 2048 Hz to
    256 Hz. Returns a float64 tensor; the result is not differentiable.
    """
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)
    out = signal.decimate(y_np, factor, ftype="fir", axis=0, zero_phase=True)
    return torch.as_tensor(np.ascontiguousarray(out), dtype=torch.float64)
