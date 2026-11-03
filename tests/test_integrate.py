"""T1.3: integrators and decimation."""

import math

import pytest
import torch

from sz.integrate import DECIM, FS_OBS, FS_SIM, decimate, euler, euler_maruyama, rk4
from sz.model import N_STATES, make_params, output

torch.set_default_dtype(torch.float64)

X0 = torch.zeros(N_STATES)


@pytest.mark.parametrize("A", [3.25, 4.0])
def test_rk4_step_halving_under_1_percent(A):
    """Halving the RK4 step changes y by < 1% (relative L2) on shared samples."""
    theta = make_params(A=A)
    T = 5.0  # s
    dt = 1.0 / FS_SIM
    n = int(round(T / dt))
    y_coarse = output(rk4(X0, theta, 220.0, dt=dt, n_steps=n))
    y_fine = output(rk4(X0, theta, 220.0, dt=dt / 2, n_steps=2 * n))[::2]
    rel = torch.linalg.norm(y_coarse - y_fine) / torch.linalg.norm(y_fine)
    assert rel.item() < 0.01


def test_rk4_shapes_and_initial_state():
    theta = make_params()
    p = torch.full((100,), 220.0)
    X = rk4(X0, theta, p)
    assert X.shape == (101, N_STATES)
    torch.testing.assert_close(X[0], X0)


def test_rk4_input_length_mismatch_raises():
    with pytest.raises(ValueError):
        rk4(X0, make_params(), torch.full((10,), 220.0), torch.zeros(9))


def test_inputs_held_per_step():
    """A step change in u at step k only affects states after step k."""
    theta = make_params()
    n, k = 200, 120
    u = torch.zeros(n)
    u[k:] = 50.0
    X_base = rk4(X0, theta, 220.0, 0.0, n_steps=n)
    X_step = rk4(X0, theta, 220.0, u)
    torch.testing.assert_close(X_step[: k + 1], X_base[: k + 1])
    assert not torch.allclose(X_step[k + 1], X_base[k + 1])


def test_batched_rk4_matches_single():
    theta = make_params()
    x0s = torch.stack([X0, X0 + 0.1])
    Xb = rk4(x0s, theta, 220.0, n_steps=50)
    for i in range(2):
        torch.testing.assert_close(Xb[:, i], rk4(x0s[i], theta, 220.0, n_steps=50))


def test_euler_maruyama_zero_noise_equals_euler():
    theta = make_params()
    X_em, p_real = euler_maruyama(X0, theta, 220.0, 0.0, n_steps=500)
    torch.testing.assert_close(X_em, euler(X0, theta, 220.0, n_steps=500))
    torch.testing.assert_close(p_real, torch.full((500,), 220.0))


def test_euler_maruyama_reproducible_and_realized_input():
    theta = make_params()
    n, sigma = 4000, 5.0
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    X1, p1 = euler_maruyama(X0, theta, 220.0, sigma, n_steps=n, generator=g1)
    X2, p2 = euler_maruyama(X0, theta, 220.0, sigma, n_steps=n, generator=g2)
    torch.testing.assert_close(X1, X2)
    torch.testing.assert_close(p1, p2)

    # Re-running Euler with the realized input reproduces the trajectory
    torch.testing.assert_close(euler(X0, theta, p1), X1)

    # Realized input has mean p_mean and std sigma / sqrt(dt)
    dt = 1.0 / FS_SIM
    assert p1.mean().item() == pytest.approx(220.0, abs=4 * sigma / math.sqrt(dt * n))
    assert p1.std().item() == pytest.approx(sigma / math.sqrt(dt), rel=0.05)


def test_decimate_rate_and_filtering():
    assert FS_SIM // DECIM == FS_OBS == 256
    t = torch.arange(4 * FS_SIM) / FS_SIM  # 4 s
    low = torch.sin(2 * math.pi * 10.0 * t)    # passes
    high = torch.sin(2 * math.pi * 300.0 * t)  # above the 128 Hz Nyquist of 256 Hz

    d_low = decimate(low)
    d_high = decimate(high)
    assert d_low.shape == (4 * FS_OBS,)

    inner = slice(FS_OBS // 2, -FS_OBS // 2)  # skip filter edges
    t_obs = torch.arange(4 * FS_OBS) / FS_OBS
    torch.testing.assert_close(
        d_low[inner], torch.sin(2 * math.pi * 10.0 * t_obs)[inner], rtol=0, atol=1e-2
    )
    assert d_high[inner].abs().max().item() < 0.05


def test_decimate_multichannel_time_axis_first():
    X = torch.randn(FS_SIM, 3)
    assert decimate(X).shape == (FS_OBS, 3)
