"""T1.2: the vector field matches the Jansen-Rit equations term by term."""

import math

import pytest
import torch

from sz.model import (
    DEFAULT_PARAMS,
    N_STATES,
    connectivity,
    make_params,
    output,
    sigmoid,
    vector_field,
)

torch.set_default_dtype(torch.float64)


def S_ref(v, P):
    """Sigmoid written out with Python floats."""
    return 2.0 * P["e0"] / (1.0 + math.exp(P["r"] * (P["v0"] - v)))


def f_ref(x, P, p, u):
    """Jansen-Rit equations written out term by term with Python floats."""
    A, B, a, b, C = P["A"], P["B"], P["a"], P["b"], P["C"]
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    y0, y1, y2, y3, y4, y5 = x
    return [
        y3,
        y4,
        y5,
        A * a * S_ref(y1 - y2, P) - 2 * a * y3 - a**2 * y0,
        A * a * (p + u + C2 * S_ref(C1 * y0, P)) - 2 * a * y4 - a**2 * y1,
        B * b * C4 * S_ref(C3 * y0, P) - 2 * b * y5 - b**2 * y2,
    ]


def test_standard_values():
    assert DEFAULT_PARAMS == {
        "A": 3.25, "B": 22.0, "a": 100.0, "b": 50.0,
        "C": 135.0, "e0": 2.5, "v0": 6.0, "r": 0.56,
    }
    assert connectivity(135.0) == (135.0, 108.0, 33.75, 33.75)


def test_make_params_overrides_and_rejects_unknown():
    theta = make_params(A=4.0)
    assert theta["A"] == 4.0 and theta["B"] == 22.0
    with pytest.raises(KeyError):
        make_params(D=1.0)


def test_sigmoid_values():
    P = DEFAULT_PARAMS
    e0, v0, r = P["e0"], P["v0"], P["r"]
    assert sigmoid(torch.tensor(v0), e0, v0, r).item() == pytest.approx(e0)
    assert sigmoid(torch.tensor(-1e3), e0, v0, r).item() == pytest.approx(0.0)
    assert sigmoid(torch.tensor(1e3), e0, v0, r).item() == pytest.approx(2 * e0)
    for v in (-5.0, 0.0, 3.3, 10.0):
        assert sigmoid(torch.tensor(v), e0, v0, r).item() == pytest.approx(S_ref(v, P))


@pytest.mark.parametrize("A", [3.25, 4.0])
def test_vector_field_matches_equations(A):
    P = make_params(A=A)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        x = torch.randn(N_STATES, generator=gen) * torch.tensor([0.1, 20, 15, 5, 500, 500])
        p = 120.0 + 200.0 * torch.rand(1, generator=gen).item()
        u = 50.0 * torch.randn(1, generator=gen).item()
        got = vector_field(x, P, p, u)
        want = torch.tensor(f_ref(x.tolist(), P, p, u))
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-9)


def test_zero_state():
    P = DEFAULT_PARAMS
    p = 220.0
    S0 = S_ref(0.0, P)
    C1, C2, C3, C4 = connectivity(P["C"])
    got = vector_field(torch.zeros(N_STATES), P, p)
    want = torch.tensor([
        0.0, 0.0, 0.0,
        P["A"] * P["a"] * S0,
        P["A"] * P["a"] * (p + C2 * S0),
        P["B"] * P["b"] * C4 * S0,
    ])
    torch.testing.assert_close(got, want)


def test_u_adds_to_p():
    x = torch.tensor([0.05, 15.0, 10.0, 1.0, -30.0, 20.0])
    P = DEFAULT_PARAMS
    torch.testing.assert_close(vector_field(x, P, 230.0, 0.0), vector_field(x, P, 220.0, 10.0))


def test_output_is_y1_minus_y2():
    x = torch.tensor([[0.1, 12.0, 9.5, 0.0, 0.0, 0.0], [0.0, -1.0, 2.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(output(x), torch.tensor([2.5, -3.0]))


def test_batched_states_match_single():
    gen = torch.Generator().manual_seed(1)
    X = torch.randn(5, N_STATES, generator=gen)
    p = torch.linspace(120.0, 320.0, 5)
    batched = vector_field(X, DEFAULT_PARAMS, p)
    for i in range(5):
        torch.testing.assert_close(batched[i], vector_field(X[i], DEFAULT_PARAMS, p[i]))


def test_autograd_and_jacfwd():
    x = torch.tensor([0.05, 15.0, 10.0, 1.0, -30.0, 20.0])
    A = torch.tensor(3.25, requires_grad=True)
    theta = make_params(A=A)
    vector_field(x, theta, 220.0).sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad)

    J = torch.func.jacfwd(lambda z: vector_field(z, DEFAULT_PARAMS, 220.0))(x)
    assert J.shape == (N_STATES, N_STATES)
    eps = 1e-6
    for j in range(N_STATES):
        dx = torch.zeros(N_STATES)
        dx[j] = eps
        fd = (vector_field(x + dx, DEFAULT_PARAMS, 220.0)
              - vector_field(x - dx, DEFAULT_PARAMS, 220.0)) / (2 * eps)
        torch.testing.assert_close(J[:, j], fd, rtol=1e-6, atol=1e-5)
