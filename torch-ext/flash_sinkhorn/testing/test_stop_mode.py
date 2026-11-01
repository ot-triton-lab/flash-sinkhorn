"""Tests for the marginal stopping rules (stop_mode="marginal_l1" / "marginal_linf")."""

import warnings

import pytest
import torch

import flash_sinkhorn._autograd as autograd_module
from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.multiscale._preprocess import round_to_tf32
from flash_sinkhorn.sinkhorn_solvers import (
    _marginal_residual,
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EPS = 0.05


def _problem(n=600, m=500, d=3, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, d, device="cuda", generator=g)
    y = torch.randn(m, d, device="cuda", generator=g) * 0.7 + 0.5
    a = torch.rand(n, device="cuda", generator=g) + 0.5
    b = torch.rand(m, device="cuda", generator=g) + 0.5
    return x, y, a / a.sum(), b / b.sum()


def _dense_residuals(x, y, a, b, f, g, eps, cost_scale=1.0, ott_convention=False):
    x, y, a, b, f, g = (t.double() for t in (x, y, a, b, f, g))
    log_p = (f[:, None] + g[None, :] - cost_scale * torch.cdist(x, y) ** 2) / eps
    if not ott_convention:
        log_p = log_p + a.log()[:, None] + b.log()[None, :]
    p = log_p.exp()
    return (p.sum(1) - a).abs(), (p.sum(0) - b).abs()


def _l1(x, y, a, b, f, g, eps, **kwargs):
    row, col = _dense_residuals(x, y, a, b, f, g, eps, **kwargs)
    return float(row.sum() + col.sum())


def _linf(x, y, a, b, f, g, eps, **kwargs):
    row, col = _dense_residuals(x, y, a, b, f, g, eps, **kwargs)
    return float(torch.maximum(row.max(), col.max()))


def _alternating(x, y, a, b, n_iters, **kwargs):
    return sinkhorn_flashstyle_alternating(x, y, a, b, eps=EPS, n_iters=n_iters, allow_tf32=False,
                                           autotune=False, **kwargs)


def test_residual_matches_dense_plan_alternating():
    x, y, a, b = _problem()
    f_k, g_k = _alternating(x, y, a, b, 7)
    f_next, _ = _alternating(x, y, a, b, 8)
    row, col = _dense_residuals(x, y, a, b, f_k, g_k, EPS)
    assert float(col.sum()) < 1e-5
    for mode, expected in (("marginal_l1", row.sum()), ("marginal_linf", row.max())):
        got = _marginal_residual(a, f_k, f_next, EPS, mode)
        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-7)


@pytest.mark.parametrize("ott_convention", [False, True])
@pytest.mark.parametrize("mode,measure", [("marginal_l1", _l1), ("marginal_linf", _linf)])
def test_alternating_stops_at_marginal_threshold(ott_convention, mode, measure):
    x, y, a, b = _problem()
    tol = 1e-4 if mode == "marginal_l1" else 1e-5
    f, g, used = _alternating(x, y, a, b, 5000, threshold=tol, check_every=1, stop_mode=mode,
                              return_n_iters=True, ott_convention=ott_convention)
    assert used < 5000
    assert measure(x, y, a, b, f, g, EPS, ott_convention=ott_convention) <= tol
    f_prev, g_prev = _alternating(x, y, a, b, used - 2, ott_convention=ott_convention)
    assert measure(x, y, a, b, f_prev, g_prev, EPS, ott_convention=ott_convention) > tol


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("mode,measure", [("marginal_l1", _l1), ("marginal_linf", _linf)])
def test_symmetric_stops_at_marginal_threshold(fused, mode, measure):
    x, y, a, b = _problem()
    tol = 1e-4 if mode == "marginal_l1" else 1e-5
    common = dict(eps_list=[EPS] * 5000, allow_tf32=False, fused=fused, last_extrapolation=False,
                  autotune=False)
    f, g, used = sinkhorn_flashstyle_symmetric(x, y, a, b, threshold=tol, check_every=1, stop_mode=mode,
                                               return_n_iters=True, **common)
    assert used < 5000
    certified = dict(common, eps_list=[EPS] * (used - 2))
    f_cert, g_cert = sinkhorn_flashstyle_symmetric(x, y, a, b, **certified)
    assert measure(x, y, a, b, f_cert, g_cert, EPS) <= tol * 1.01
    f_prev, g_prev = sinkhorn_flashstyle_symmetric(x, y, a, b, **dict(common, eps_list=[EPS] * (used - 3)))
    assert measure(x, y, a, b, f_prev, g_prev, EPS) > tol
    assert measure(x, y, a, b, f, g, EPS) <= tol * 1.1
    looser = sinkhorn_flashstyle_symmetric(x, y, a, b, threshold=tol * 100, check_every=1, stop_mode=mode,
                                           return_n_iters=True, **common)[2]
    assert looser < used


def test_potential_and_marginal_rules_stop_differently():
    x, y, a, b = _problem()
    *_, by_potential = _alternating(x, y, a, b, 5000, threshold=1e-4, check_every=1, return_n_iters=True)
    *_, by_marginal = _alternating(x, y, a, b, 5000, threshold=1e-4, check_every=1, stop_mode="marginal_l1",
                                   return_n_iters=True)
    assert by_potential != by_marginal


def test_samples_loss_threads_stop_mode():
    x, y, a, b = _problem()
    tol = 1e-4
    common = dict(backend="alternating", use_epsilon_scaling=False, eps=EPS, n_iters=5000, potentials=True,
                  allow_tf32=False, truncate_tf32=False, autotune=False, threshold=tol, inner_iterations=1)
    f, g = SamplesLoss(stop_mode="marginal_l1", **common)(a, x, b, y)
    assert _l1(x, y, a, b, f, g, EPS) <= tol
    f, g = SamplesLoss(**common)(a, x, b, y)
    assert _l1(x, y, a, b, f, g, EPS) > tol


def test_invalid_stop_modes_are_rejected():
    x, y, a, b = _problem(n=64, m=64)
    with pytest.raises(ValueError, match="stop_mode must be one of"):
        SamplesLoss(stop_mode="marginal")
    with pytest.raises(ValueError, match="balanced"):
        SamplesLoss(stop_mode="marginal_l1", reach=1.0)
    with pytest.raises(ValueError, match="multiscale"):
        SamplesLoss(backend="multiscale", stop_mode="marginal_l1")
    with pytest.raises(ValueError, match="balanced"):
        _alternating(x, y, a, b, 10, threshold=1e-3, stop_mode="marginal_l1", reach_y=1.0)
    with pytest.raises(ValueError, match="stop_mode must be one of"):
        sinkhorn_flashstyle_symmetric(x, y, a, b, eps_list=[EPS] * 10, stop_mode="tv")


def test_zero_weight_residual_ignores_nonfinite_padded_potentials():
    weights = torch.tensor([0.25, 0.75, 0.0], device="cuda")
    before = torch.tensor([0.1, 0.2, -torch.inf], device="cuda")
    after = torch.tensor([0.12, 0.19, -torch.inf], device="cuda")
    for mode in ("marginal_l1", "marginal_linf"):
        got = _marginal_residual(weights, before, after, EPS, mode)
        expected = _marginal_residual(weights[:2], before[:2], after[:2], EPS, mode)
        torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("backend", ["alternating", "symmetric"])
def test_stop_mode_without_threshold_preserves_fixed_iterations(backend):
    x, y, a, b = _problem(n=79, m=91)
    common = dict(backend=backend, eps=EPS, n_iters=23, use_epsilon_scaling=False, allow_tf32=False,
                  truncate_tf32=False, autotune=False, potentials=True)
    expected = SamplesLoss(**common)(a, x, b, y)
    for mode in ("marginal_l1", "marginal_linf"):
        with pytest.warns(UserWarning, match="no effect without threshold"):
            actual = SamplesLoss(stop_mode=mode, **common)(a, x, b, y)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_marginal_rule_with_padding_and_half_cost():
    x, y, a, b = _problem(n=79, m=91)
    loss = SamplesLoss(backend="alternating", eps=EPS, n_iters=5000, use_epsilon_scaling=False,
                       allow_tf32=False, truncate_tf32=False, potentials=True, pad_to_multiple=128, autotune=False,
                       half_cost=True, threshold=1e-4, inner_iterations=1, stop_mode="marginal_l1")
    f, g = loss(a, x, b, y)
    assert f.shape == a.shape and g.shape == b.shape
    assert _l1(x, y, a, b, f, g, EPS, cost_scale=0.5) <= 1e-4


@pytest.mark.parametrize("iterations,check_every,message", [
    (3, 1, "not reached by the marginal_l1 residual"),
    (1, 10, "no marginal_l1 check ran within 1 updates"),
    (1, 1, "from the second update on"),
])
def test_marginal_rule_reports_unmet_threshold(iterations, check_every, message):
    x, y, a, b = _problem(n=79, m=91)
    with pytest.warns(RuntimeWarning, match=message):
        _alternating(x, y, a, b, iterations, threshold=1e-4, check_every=check_every, stop_mode="marginal_l1")


def test_symmetric_reports_missing_check_at_target_eps():
    x, y, a, b = _problem(n=79, m=91)
    with pytest.warns(RuntimeWarning, match="no marginal_l1 check ran at the target eps.*eps_list"):
        sinkhorn_flashstyle_symmetric(x, y, a, b, eps_list=[0.5, 0.2, EPS], allow_tf32=False, autotune=False,
                                      threshold=1e-4, check_every=10, stop_mode="marginal_l1")


@pytest.mark.parametrize("backend,last_extrapolation", [
    ("alternating", True), ("symmetric", True), ("symmetric", False),
])
def test_cost_path_threads_stop_mode(monkeypatch, backend, last_extrapolation):
    seen = []
    for name in ("sinkhorn_flashstyle_alternating", "sinkhorn_flashstyle_symmetric"):
        solver = getattr(autograd_module, name)

        def recording(*args, _solver=solver, **kwargs):
            seen.append(kwargs.get("stop_mode"))
            return _solver(*args, **kwargs)
        monkeypatch.setattr(autograd_module, name, recording)
    x, y, a, b = _problem(n=79, m=91)
    x = x.clone().requires_grad_()
    loss = SamplesLoss(backend=backend, eps=EPS, n_iters=50, use_epsilon_scaling=False, allow_tf32=False,
                       truncate_tf32=False, autotune=False, last_extrapolation=last_extrapolation,
                       threshold=1e-3, inner_iterations=1, stop_mode="marginal_linf")
    loss(a, x, b, y).backward()
    assert seen and set(seen) == {"marginal_linf"}
    assert torch.isfinite(x.grad).all()


def test_threshold_below_rounding_floor_warns():
    x, y, a, b = _problem()
    with pytest.warns(RuntimeWarning, match="rounding floor"):
        _alternating(x + 30, y + 30, a, b, 2, threshold=1e-6, stop_mode="marginal_l1")
    x16, y16, a16, b16 = _problem(d=16)
    with pytest.warns(RuntimeWarning, match="rounding floor"):
        sinkhorn_flashstyle_symmetric(x16, y16, a16, b16, eps_list=[EPS] * 2, allow_tf32=True, autotune=False,
                                      threshold=1e-4, stop_mode="marginal_l1")


def test_tf32_rounded_coordinates_have_fp32_floor():
    x, y, a, b = _problem()
    x, y = round_to_tf32(x), round_to_tf32(y)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*rounding floor")
        sinkhorn_flashstyle_symmetric(x, y, a, b, eps_list=[EPS] * 2, allow_tf32=True, autotune=False,
                                      threshold=1e-4, check_every=100, stop_mode="marginal_l1")


def test_marginal_rule_rejects_invalid_weights():
    x, y, a, b = _problem(n=79, m=91)
    a = a.clone()
    a[0] = -a[0]
    with pytest.raises(ValueError, match="nonnegative weights a"):
        _alternating(x, y, a, b, 2, threshold=1e-3, stop_mode="marginal_l1")
