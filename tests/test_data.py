"""T1.5: synthetic dataset generator."""

import math

import numpy as np
import pytest
import torch

from sz.data import (
    add_observation_noise,
    load_dataset,
    save_dataset,
    simulate_records,
)
from sz.integrate import FS_OBS, FS_SIM, euler_maruyama
from sz.model import make_params, output

KW = dict(p_mean=90.0, sigma=1.0, duration_s=1.0, transient_s=0.5)


def test_shapes():
    r = simulate_records([3.25, 4.0], [1, 2], **KW)
    assert r["y_clean"].shape == (2, FS_OBS)
    assert r["x"].shape == (2, FS_OBS, 6)
    assert r["p"].shape == (2, FS_SIM)


def test_record_same_alone_or_in_batch():
    batch = simulate_records([3.25, 4.0, 3.75], [11, 12, 13], **KW)
    alone = simulate_records([4.0], [12], **KW)
    for k in ("y_clean", "x", "p"):
        np.testing.assert_array_equal(batch[k][1], alone[k][0])


def test_same_seed_reproduces_and_different_seed_differs():
    a = simulate_records([3.25], [5], **KW)
    b = simulate_records([3.25], [5], **KW)
    c = simulate_records([3.25], [6], **KW)
    np.testing.assert_array_equal(a["y_clean"], b["y_clean"])
    assert not np.array_equal(a["y_clean"], c["y_clean"])


def test_states_and_input_match_a_direct_simulation():
    """x at the observation times and p equal a direct Euler-Maruyama run."""
    r = simulate_records([4.0], [7], **KW)
    n_drop = int(KW["transient_s"] * FS_SIM)
    n_steps = n_drop + int(KW["duration_s"] * FS_SIM)
    noise = torch.randn(n_steps, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(7))
    X, p = euler_maruyama(torch.zeros(6, dtype=torch.float64), make_params(A=4.0),
                          KW["p_mean"], KW["sigma"], n_steps, noise=noise)
    np.testing.assert_allclose(r["p"][0], p[n_drop:].numpy())
    np.testing.assert_allclose(r["x"][0], X[1:][::8][n_drop // 8:].numpy())
    # y_clean is the filtered output; it stays close to y1 - y2 at those times
    y_direct = output(torch.as_tensor(r["x"][0])).numpy()
    rel = np.linalg.norm(r["y_clean"][0] - y_direct) / np.linalg.norm(y_direct - y_direct.mean())
    assert rel < 0.05


def test_input_statistics():
    r = simulate_records([3.25], [3], p_mean=90.0, sigma=1.0, duration_s=4.0, transient_s=0.5)
    p = r["p"][0]
    sd = 1.0 * math.sqrt(FS_SIM)
    assert p.mean() == pytest.approx(90.0, abs=4 * sd / math.sqrt(p.size))
    assert p.std() == pytest.approx(sd, rel=0.03)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_records([3.25, 4.0], [1], **KW)
    with pytest.raises(ValueError):
        simulate_records([3.25], [1], theta_overrides={"A": 4.0}, **KW)
    with pytest.raises(ValueError):
        simulate_records([3.25], [1], p_mean=90.0, sigma=1.0, duration_s=1.001, transient_s=0.5)


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0, 0.0])
def test_observation_noise_hits_snr(snr_db):
    y = np.sin(np.linspace(0, 20, 2560)) * 3.0 + 1.0
    y_noisy, sd = add_observation_noise(y, snr_db, seed=1)
    assert 10 * np.log10(np.var(y) / sd**2) == pytest.approx(snr_db)
    measured = 10 * np.log10(np.var(y) / np.var(y_noisy - y))
    assert measured == pytest.approx(snr_db, abs=0.5)


def test_observation_noise_common_random_numbers():
    y = np.sin(np.linspace(0, 20, 2560))
    n10 = add_observation_noise(y, 10.0, seed=3)[0] - y
    n0 = add_observation_noise(y, 0.0, seed=3)[0] - y
    np.testing.assert_allclose(n0 / np.std(n0), n10 / np.std(n10), rtol=0, atol=1e-12)


def test_save_load_roundtrip_and_no_overwrite(tmp_path):
    arrays = {"y": np.arange(5.0), "x": np.ones((5, 6))}
    meta = {"theta_true": make_params(A=4.0), "sim_seeds": [1, 2], "snr_db": 10.0}
    save_dataset(tmp_path / "d" / "ds", arrays, meta)
    a2, m2 = load_dataset(tmp_path / "d" / "ds")
    np.testing.assert_array_equal(a2["y"], arrays["y"])
    np.testing.assert_array_equal(a2["x"], arrays["x"])
    assert m2 == meta
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path / "d" / "ds", arrays, meta)


def test_dotted_names_are_kept(tmp_path):
    """Names like A3.25_seed0 must not lose the part after the dot."""
    arrays = {"y": np.arange(3.0)}
    for a in (3.25, 3.75):
        save_dataset(tmp_path / f"A{a:.2f}_seed0", arrays, {"A_true": a})
    assert (tmp_path / "A3.25_seed0.npz").exists()
    assert (tmp_path / "A3.75_seed0.json").exists()
    _, meta = load_dataset(tmp_path / "A3.75_seed0")
    assert meta["A_true"] == 3.75
    _, meta = load_dataset(tmp_path / "A3.25_seed0.npz")
    assert meta["A_true"] == 3.25
