"""T2.1-T2.2: windowed fit-error landscapes, sensitivity and Fisher information."""

import math

import numpy as np
import pytest
import torch

from sz.data import save_dataset, simulate_records
from sz.identifiability import (
    fisher_information,
    fisher_summary,
    landscape,
    load_reference,
    make_windows,
    output_sensitivity,
    relative_grid,
    simulate_windows,
    subset_summary,
    window_obs_from_seconds,
)
from sz.model import make_params, output

KW = dict(p_mean=90.0, sigma=1.326, duration_s=1.0, transient_s=0.5)
W = 32  # observations per window in these tests (0.125 s)


@pytest.fixture(scope="module")
def rec():
    r = simulate_records([4.0], [7], **KW)
    return {"x": torch.as_tensor(r["x"][0]), "p": torch.as_tensor(r["p"][0])}


@pytest.fixture(scope="module")
def win(rec):
    return make_windows(rec["x"], rec["p"], W)


def test_window_shapes(rec, win):
    n_win = (rec["x"].shape[0] - 1) // W          # 255 // 32 = 7
    assert win["x0"].shape == (n_win, 6)
    assert win["p"].shape == (8 * W, n_win)
    assert win["y_ref"].shape == (W, n_win)
    assert win["start"].tolist() == [k * W for k in range(n_win)]


def test_true_parameters_reproduce_the_record_exactly(rec, win):
    """Window bookkeeping: x[m] and p[1 + 8m:] give back x[m + 1 ...]."""
    y = simulate_windows(win["x0"], make_params(A=4.0), win["p"])
    torch.testing.assert_close(y, win["y_ref"], rtol=0, atol=1e-10)


def test_landscape_zero_only_at_truth(win):
    theta = make_params(A=4.0)
    vA = relative_grid(4.0, 0.05, 5)
    vB = relative_grid(22.0, 0.05, 5)
    E = landscape(win, theta, ("A", "B"), (vA, vB))
    assert E.shape == (5, 5)
    assert E[2, 2] < 1e-20
    off = np.delete(E.ravel(), 12)
    assert (off > 1e-6).all()


def test_landscape_is_the_same_for_any_batch_size(win):
    theta = make_params(A=4.0)
    v1 = relative_grid(4.0, 0.03, 5)
    v2 = relative_grid(100.0, 0.03, 3)
    E1 = landscape(win, theta, ("A", "a"), (v1, v2), max_batch=1000)
    E2 = landscape(win, theta, ("A", "a"), (v1, v2), max_batch=4)
    np.testing.assert_allclose(E1, E2, rtol=1e-12, atol=0)


def test_landscape_point_matches_direct_nmse(win):
    theta = make_params(A=4.0)
    E = landscape(win, theta, ("A", "C"), ([4.1], [130.0]))
    y = simulate_windows(win["x0"], make_params(A=4.1, C=130.0), win["p"])
    expected = float(((y - win["y_ref"]) ** 2).mean() / win["y_ref"].var())
    assert E[0, 0] == pytest.approx(expected, rel=1e-12)


def test_landscape_rejects_bad_pairs(win):
    theta = make_params(A=4.0)
    with pytest.raises(ValueError):
        landscape(win, theta, ("A", "A"), ([4.0], [4.0]))
    with pytest.raises(KeyError):
        landscape(win, theta, ("A", "Z"), ([4.0], [1.0]))


def test_sensitivity_matches_finite_differences(win):
    names = ["A", "B", "a", "b", "C", "e0", "v0", "r"]
    theta = make_params(A=4.0)
    y, J = output_sensitivity(win, theta, names)
    assert J.shape == (win["y_ref"].numel(), len(names))
    torch.testing.assert_close(y, win["y_ref"].reshape(-1), rtol=0, atol=1e-10)
    eps = 1e-6
    for j, n in enumerate(names):
        up = dict(theta, **{n: theta[n] * math.exp(eps)})
        dn = dict(theta, **{n: theta[n] * math.exp(-eps)})
        fd = (simulate_windows(win["x0"], up, win["p"])
              - simulate_windows(win["x0"], dn, win["p"])).reshape(-1) / (2 * eps)
        rel = float((fd - J[:, j]).norm() / J[:, j].norm())
        assert rel < 1e-5, f"{n}: relative difference {rel:.2e}"


def test_fisher_information_scaling():
    J = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    F = fisher_information(J, 0.5)
    torch.testing.assert_close(F, J.T @ J / 0.25)
    with pytest.raises(ValueError):
        fisher_information(J, 0.0)


def test_fisher_summary_two_by_two():
    """For a 2x2 F, corr(F^-1) = -F12 / sqrt(F11 F22) and sd_i = sqrt((F^-1)_ii)."""
    F = torch.tensor([[4.0, 1.5], [1.5, 1.0]], dtype=torch.float64)
    S = fisher_summary(F)
    np.testing.assert_allclose(S["eigvals"], np.sort(np.linalg.eigvalsh(F.numpy()))[::-1])
    assert S["eigvals"][0] >= S["eigvals"][1]
    np.testing.assert_allclose(S["cov"], np.linalg.inv(F.numpy()), rtol=1e-12)
    assert S["corr"][0, 1] == pytest.approx(-1.5 / math.sqrt(4.0 * 1.0))
    np.testing.assert_allclose(np.diag(S["corr"]), 1.0)
    np.testing.assert_allclose(S["rel_sd"], np.sqrt(np.diag(np.linalg.inv(F.numpy()))))
    assert S["cond"] == pytest.approx(S["eigvals"][0] / S["eigvals"][1])
    # eigenvectors: unit length, F v = lambda v, largest entry positive
    V = S["eigvecs"]
    np.testing.assert_allclose(F.numpy() @ V, V * S["eigvals"], atol=1e-12)
    for k in range(2):
        assert V[np.argmax(np.abs(V[:, k])), k] > 0


def test_fisher_summary_rejects_singular():
    F = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        fisher_summary(F)


def test_relative_grid():
    g = relative_grid(22.0, 0.05, 41)
    assert g.size == 41
    assert g[20] == pytest.approx(22.0)
    assert g[0] == pytest.approx(22.0 * 0.95)
    assert g[-1] == pytest.approx(22.0 * 1.05)
    with pytest.raises(ValueError):
        relative_grid(22.0, 0.05, 40)


def test_window_obs_from_seconds():
    assert window_obs_from_seconds(0.5) == 128
    with pytest.raises(ValueError):
        window_obs_from_seconds(0.001)


def test_load_reference_round_trip(tmp_path):
    r = simulate_records([4.0, 4.0], [7, 8], **KW)
    meta = {"theta_true": make_params(A=4.0), "noise_sd_mV": [0.3, 0.4]}
    save_dataset(tmp_path / "A4.00_seed0", r, meta)
    ref = load_reference(tmp_path / "A4.00_seed0", record=1)
    np.testing.assert_array_equal(ref["x"].numpy(), r["x"][1])
    np.testing.assert_array_equal(ref["p"].numpy(), r["p"][1])
    assert ref["theta"]["A"] == 4.0
    assert ref["noise_sd"] == 0.4
    with pytest.raises(IndexError):
        load_reference(tmp_path / "A4.00_seed0", record=2)



# T2.3: subsets with the other parameters fixed

def test_subset_summary_uses_the_block_of_F_not_the_block_of_F_inverse():
    F = np.array([[4.0, 1.5, 0.8], [1.5, 2.0, 0.6], [0.8, 0.6, 1.0]])
    names = ["A", "B", "C"]
    s = subset_summary(F, names, ["A", "B"])
    cov = np.linalg.inv(F[:2, :2])
    assert s["rel_sd"]["A"] == pytest.approx(math.sqrt(cov[0, 0]))
    assert s["rel_sd"]["B"] == pytest.approx(math.sqrt(cov[1, 1]))
    assert s["max_corr"] == pytest.approx(abs(-1.5 / math.sqrt(4.0 * 2.0)))
    # fixing C gives a smaller bound than letting it vary
    assert s["rel_sd"]["A"] < math.sqrt(np.linalg.inv(F)[0, 0])
    w = np.linalg.eigvalsh(F[:2, :2])
    assert s["cond"] == pytest.approx(w.max() / w.min())


def test_subset_summary_full_set_matches_fisher_summary():
    F = torch.tensor([[4.0, 1.5], [1.5, 1.0]], dtype=torch.float64)
    s = subset_summary(F, ["A", "B"], ["B", "A"])          # order follows the subset
    S = fisher_summary(F)
    assert s["rel_sd"]["A"] == pytest.approx(S["rel_sd"][0])
    assert s["rel_sd"]["B"] == pytest.approx(S["rel_sd"][1])
    np.testing.assert_allclose(s["corr"], S["corr"][::-1, ::-1])


def test_subset_summary_rejects_bad_subsets():
    F = np.eye(3)
    with pytest.raises(ValueError):
        subset_summary(F, ["A", "B", "C"], ["A"])
    with pytest.raises(ValueError):
        subset_summary(F, ["A", "B", "C"], ["A", "A"])
    with pytest.raises(KeyError):
        subset_summary(F, ["A", "B", "C"], ["A", "Z"])
    with pytest.raises(ValueError):
        subset_summary(np.ones((3, 3)), ["A", "B", "C"], ["A", "B"])