"""Jansen-Rit neural mass model.

States x = (y0, y1, y2, y3, y4, y5):
    y0  pyramidal population
    y1  excitatory interneuron population
    y2  inhibitory interneuron population
    y3-y5  derivatives of y0-y2

Equations (Jansen & Rit, 1995), with stimulation u(t) added to the
external excitatory input p(t):

    dy0/dt = y3
    dy1/dt = y4
    dy2/dt = y5
    dy3/dt = A a S(y1 - y2) - 2 a y3 - a^2 y0
    dy4/dt = A a [p(t) + u(t) + C2 S(C1 y0)] - 2 a y4 - a^2 y1
    dy5/dt = B b C4 S(C3 y0) - 2 b y5 - b^2 y2

    S(v) = 2 e0 / (1 + exp(r (v0 - v)))

EEG output: y = y1 - y2.

Units: A, B, v0, y in mV; a, b, e0 in 1/s; p, u in Hz; r in 1/mV.

This is the single model definition used by every component (simulator,
identifiability, PINN, UKF, control). It uses only out-of-place torch
operations so it works with autograd and torch.func (jacfwd, vmap).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

N_STATES = 6

# Standard values (Jansen & Rit, 1995)
DEFAULT_PARAMS: Dict[str, float] = {
    "A": 3.25,   # mV, excitatory synaptic gain
    "B": 22.0,   # mV, inhibitory synaptic gain
    "a": 100.0,  # 1/s, excitatory rate constant
    "b": 50.0,   # 1/s, inhibitory rate constant
    "C": 135.0,  # connectivity constant; C1 = C, C2 = 0.8C, C3 = C4 = 0.25C
    "e0": 2.5,   # 1/s, half the maximum firing rate
    "v0": 6.0,   # mV, firing threshold
    "r": 0.56,   # 1/mV, sigmoid steepness
}


def make_params(**overrides: Any) -> Dict[str, Any]:
    """Return the standard parameters with any given values replaced.

    Values may be floats or tensors (e.g. a tensor A that requires grad).
    Unknown parameter names raise KeyError.
    """
    unknown = set(overrides) - set(DEFAULT_PARAMS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")
    theta: Dict[str, Any] = dict(DEFAULT_PARAMS)
    theta.update(overrides)
    return theta


def connectivity(C: Any):
    """Return (C1, C2, C3, C4) = (C, 0.8C, 0.25C, 0.25C)."""
    return C, 0.8 * C, 0.25 * C, 0.25 * C


def sigmoid(v: torch.Tensor, e0: Any, v0: Any, r: Any) -> torch.Tensor:
    """Potential-to-rate sigmoid S(v) = 2 e0 / (1 + exp(r (v0 - v)))."""
    return 2.0 * e0 / (1.0 + torch.exp(r * (v0 - v)))


def vector_field(
    x: torch.Tensor,
    theta: Mapping[str, Any],
    p: Any,
    u: Any = 0.0,
) -> torch.Tensor:
    """Jansen-Rit vector field f(x, theta, p, u).

    Args:
        x: states, shape (..., 6).
        theta: full parameter mapping (see make_params).
        p: external excitatory input in Hz, scalar or tensor broadcastable
           to x[..., 0].
        u: stimulation input in Hz, scalar or tensor broadcastable to
           x[..., 0]. Added to p.

    Returns:
        dx/dt, shape (..., 6).
    """
    A, B = theta["A"], theta["B"]
    a, b = theta["a"], theta["b"]
    e0, v0, r = theta["e0"], theta["v0"], theta["r"]
    C1, C2, C3, C4 = connectivity(theta["C"])

    y0, y1, y2 = x[..., 0], x[..., 1], x[..., 2]
    y3, y4, y5 = x[..., 3], x[..., 4], x[..., 5]

    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = A * a * sigmoid(y1 - y2, e0, v0, r) - 2.0 * a * y3 - a**2 * y0
    dy4 = A * a * (p + u + C2 * sigmoid(C1 * y0, e0, v0, r)) - 2.0 * a * y4 - a**2 * y1
    dy5 = B * b * C4 * sigmoid(C3 * y0, e0, v0, r) - 2.0 * b * y5 - b**2 * y2

    return torch.stack(torch.broadcast_tensors(dy0, dy1, dy2, dy3, dy4, dy5), dim=-1)


def output(x: torch.Tensor) -> torch.Tensor:
    """EEG output y = y1 - y2, shape (...)."""
    return x[..., 1] - x[..., 2]
