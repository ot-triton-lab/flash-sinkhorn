"""Balanced, unbalanced and semi-unbalanced OT against a dense float64 reference.

Both backends at a fixed eps, run to convergence: the potentials, the cost, the gradients with respect to the points
and the weights, and the x-only Hessian-vector product must match the dense solution of the same problem. A few target
points are outliers, so a relaxed marginal changes the answer. Further tests cover the closed-form translation of
semi-unbalanced potentials, zero weights, early stopping, batched potentials and the half-cost scaling identity.
"""

from __future__ import annotations

import functools
import warnings

import pytest
import torch

import flash_sinkhorn._autograd as autograd_module
import flash_sinkhorn.samples_loss as samples_loss_module
from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.sinkhorn_solvers import _translate_semi_unbalanced, sinkhorn_flashstyle_symmetric

from .reference_sinkhorn import unbalanced_hvp_x_ref, unbalanced_sinkhorn_ref, unbalanced_value_and_grad_ref

EPS = 0.09
N_ITERS = 20000
SETTINGS = {
    "balanced": {},
    "reach": {"reach": 1.0},
    "asymmetric": {"reach_x": 0.5, "reach_y": 5.0},
    "reach_x only": {"reach_x": 1.0},
    "reach_y only": {"reach_y": 1.0},
}
RELAXED = [name for name in SETTINGS if name != "balanced"]


def _rho(setting):
    reach_x = setting.get("reach_x", setting.get("reach"))
    reach_y = setting.get("reach_y", setting.get("reach"))
    return (None if reach_x is None else reach_x ** 2), (None if reach_y is None else reach_y ** 2)


def _kwargs(backend, name):
    return dict(backend=backend, use_epsilon_scaling=False, eps=EPS, n_iters=N_ITERS, allow_tf32=False,
                **SETTINGS[name])


@functools.lru_cache(maxsize=None)
def _problem():
    torch.manual_seed(0)
    n = 256
    x = torch.randn(n, 3, device="cuda")
    y = torch.randn(n, 3, device="cuda")
    y[:13] += 5.0
    a = torch.full((n,), 1.0 / n, device="cuda")
    b = torch.full((n,), 1.0 / n, device="cuda")
    return x, y, a, b


@functools.lru_cache(maxsize=None)
def _reference(name):
    x, y, a, b = (t.double() for t in _problem())
    rho_x, rho_y = _rho(SETTINGS[name])
    f, g = unbalanced_sinkhorn_ref(x, y, a, b, EPS, N_ITERS, rho_x, rho_y)
    value, grad_x, P = unbalanced_value_and_grad_ref(x, y, a, b, f, g, EPS, rho_x, rho_y)
    return f, g, value.item(), grad_x, P


def _plan(f, g):
    x, y, a, b = (t.double() for t in _problem())
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1)
    return a[:, None] * b[None, :] * torch.exp((f[:, None] + g[None, :] - C) / EPS)


@pytest.mark.parametrize("name", list(SETTINGS))
def test_reference_is_converged(name):
    """The dense plan meets the optimality conditions: a strict marginal equals its weights, a relaxed one equals the
    weights times exp(-potential / rho). The balanced fixture converges slowest (3e-6 after 20000 iterations)."""
    _, _, a, b = (t.double() for t in _problem())
    f, g, _, _, P = _reference(name)
    rho_x, rho_y = _rho(SETTINGS[name])
    r_expected = a if rho_x is None else a * torch.exp(-f / rho_x)
    c_expected = b if rho_y is None else b * torch.exp(-g / rho_y)
    assert (P.sum(dim=1) - r_expected).abs().sum() < 1e-5
    assert (P.sum(dim=0) - c_expected).abs().sum() < 1e-5


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("name", list(SETTINGS))
def test_matches_dense_reference(backend, name):
    x, y, a, b = _problem()
    f_ref, g_ref, value_ref, grad_x_ref, P_ref = _reference(name)
    kwargs = _kwargs(backend, name)

    f, g = SamplesLoss("sinkhorn", potentials=True, **kwargs)(a, x, b, y)
    if name == "balanced":
        # Balanced potentials are unique up to one global gauge in exact arithmetic; weak coupling makes group
        # shifts poorly conditioned. Check the plan they define instead.
        P = _plan(f.double(), g.double())
        assert (P.sum(dim=1) - a.double()).abs().sum() < 1e-4
        assert (P.sum(dim=0) - b.double()).abs().sum() < 1e-4
    else:
        assert (f.double() - f_ref).abs().max() < 2e-4
        assert (g.double() - g_ref).abs().max() < 2e-4

    xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    cost = SamplesLoss("sinkhorn", **kwargs)(a, xr, b, yr)
    assert abs(cost.item() - value_ref) <= 1e-5 * max(1.0, abs(value_ref))

    grad_x, grad_y = torch.autograd.grad(cost, [xr, yr])
    grad_y_ref = 2 * (P_ref.sum(dim=0)[:, None] * y.double() - P_ref.T @ x.double())
    assert (grad_x.double() - grad_x_ref).norm() <= 1e-3 * grad_x_ref.norm()
    assert (grad_y.double() - grad_y_ref).norm() <= 1e-3 * grad_y_ref.norm()


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("name", RELAXED)
def test_weight_gradients(backend, name):
    """d OT / d a and d OT / d b against the envelope derivative of the dense solution. normalize=False passes the
    probability weights through unchanged, so no normalization chain rule enters. Balanced weight gradients are the
    potentials, defined up to a constant, and are left out."""
    x, y, a, b = _problem()
    f_ref, g_ref, _, _, P_ref = _reference(name)
    rho_x, rho_y = _rho(SETTINGS[name])
    ar, br = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    cost = SamplesLoss("sinkhorn", normalize=False, **_kwargs(backend, name))(ar, x, br, y)
    grad_a, grad_b = torch.autograd.grad(cost, [ar, br])

    def envelope(w, pot, mass, rho, other_total):
        phi = pot if rho is None else rho * (1 - torch.exp(-pot / rho))
        return phi - EPS * (mass / w - other_total)

    A, B = a.double(), b.double()
    grad_a_ref = envelope(A, f_ref, P_ref.sum(dim=1), rho_x, B.sum())
    grad_b_ref = envelope(B, g_ref, P_ref.sum(dim=0), rho_y, A.sum())
    assert (grad_a.double() - grad_a_ref).abs().max() <= 1e-3 * grad_a_ref.abs().max()
    assert (grad_b.double() - grad_b_ref).abs().max() <= 1e-3 * grad_b_ref.abs().max()


@pytest.mark.parametrize("name", RELAXED)
def test_hvp_x(name):
    """Double backward with respect to x against the dense implicit-function Hessian-vector product."""
    x, y, a, b = _problem()
    f_ref, g_ref, _, _, _ = _reference(name)
    rho_x, rho_y = _rho(SETTINGS[name])
    v = torch.randn(x.shape, generator=torch.Generator().manual_seed(1)).to(x.device)
    xr = x.clone().requires_grad_(True)
    cost = SamplesLoss("sinkhorn", **_kwargs("symmetric", name))(a, xr, b, y)
    (grad,) = torch.autograd.grad(cost, xr, create_graph=True)
    (hv,) = torch.autograd.grad((grad * v).sum(), xr)
    X, Y, A, B = (t.double() for t in (x, y, a, b))
    hv_ref = unbalanced_hvp_x_ref(X, Y, A, B, f_ref, g_ref, v.double(), EPS, rho_x, rho_y)
    assert (hv.double() - hv_ref).norm() <= 1e-2 * hv_ref.norm()


@pytest.mark.parametrize("name", list(SETTINGS))
def test_half_cost_scaling(name):
    """Halving the cost, eps and the marginal penalties (reach / sqrt 2) halves the value, the gradient and the
    Hessian-vector product. The balanced Schur system contains eps * tau2, so double tau2 when halving eps to
    preserve the regularized system. Stable regularization and confirmed CG convergence keep this a scaling
    check; nearly zero regularization can make its result depend on the kernel configuration."""
    x, y, a, b = _problem()
    halved = {key: value / 2 ** 0.5 for key, value in SETTINGS[name].items()}
    common = dict(use_epsilon_scaling=False, n_iters=2000, allow_tf32=False)
    losses = (SamplesLoss("sinkhorn", eps=EPS, hvp_tau2=1e-5, **common, **SETTINGS[name]),
              SamplesLoss("sinkhorn", eps=EPS / 2, half_cost=True, hvp_tau2=2e-5, **common, **halved))
    v = torch.randn(x.shape, generator=torch.Generator().manual_seed(2)).to(x.device)
    out = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for loss in losses:
            xr = x.clone().requires_grad_(True)
            cost = loss(a, xr, b, y)
            (grad,) = torch.autograd.grad(cost, xr, create_graph=True)
            (hv,) = torch.autograd.grad((grad * v).sum(), xr)
            out.append((cost.item(), grad.detach(), hv))
    assert not [w for w in caught if "conjugate gradients" in str(w.message)]
    (c1, g1, h1), (c2, g2, h2) = out
    assert abs(c2 - 0.5 * c1) <= 1e-5 * abs(c1)
    assert (g2 - 0.5 * g1).norm() <= 1e-4 * g1.norm()
    assert (h2 - 0.5 * h1).norm() <= 1e-3 * h1.norm()


def test_translation_ignores_zero_weights():
    """A zero weight is stored as log-weight -1e5 and must not enter the translation sums: here the shift is 0."""
    log_a, log_b = torch.zeros(1), torch.tensor([0.0, -1e5])
    f_hat, g_hat = torch.zeros(1), torch.tensor([0.0, -100001.0])
    f_new, g_new = _translate_semi_unbalanced(f_hat, g_hat, torch.zeros(1), torch.zeros(2), log_a, log_b, None, 1.0)
    assert torch.equal(f_new, f_hat) and torch.equal(g_new, g_hat)


@pytest.mark.parametrize("relaxed", ["source", "target"])
def test_zero_weight_point_far_away(relaxed):
    """A zero-weight point far from the rest must not turn the semi-unbalanced cost or its gradient into nan: all
    the mass travels from 11 to 0 (or back), at cost 121."""
    near = torch.zeros(1, 3, device="cuda")
    pair = torch.tensor([[11.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda")
    one, w = torch.ones(1, device="cuda"), torch.tensor([1.0, 0.0], device="cuda")
    kwargs = dict(use_epsilon_scaling=False, eps=EPS, n_iters=2000, allow_tf32=False)
    if relaxed == "target":
        a, x, b, y = one, near, w, pair
        kwargs["reach_y"] = 1.0
    else:
        a, x, b, y = w, pair, one, near
        kwargs["reach_x"] = 1.0
    xr = x.clone().requires_grad_(True)
    cost = SamplesLoss("sinkhorn", **kwargs)(a, xr, b, y)
    (grad,) = torch.autograd.grad(cost, xr)
    assert torch.isfinite(cost) and abs(cost.item() - 121.0) <= 1e-3 * 121.0
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("fused", [True, False])
def test_early_stopping_waits_for_the_target_eps(fused):
    """A threshold must not end the epsilon-scaling descent before it reaches the target eps."""
    x, y, a, b = _problem()
    kwargs = dict(blur=EPS ** 0.5, scaling=0.9, allow_tf32=False, fused=fused)
    f1, g1 = sinkhorn_flashstyle_symmetric(x, y, a, b, **kwargs)
    f2, g2 = sinkhorn_flashstyle_symmetric(x, y, a, b, threshold=0.1, check_every=1, **kwargs)
    assert torch.allclose(f1, f2) and torch.allclose(g1, g2)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("name", ["reach_x only", "reach_y only"])
def test_translated_potentials_meet_the_mass_condition(fused, name):
    """After the closed-form translation the relaxed side satisfies <w, exp(-pot / rho)> = 1, the mass of the strict
    side, for both the returned and the prelast potentials."""
    x, y, a, b = _problem()
    rho_x, rho_y = _rho(SETTINGS[name])
    f, g, f_prelast, g_prelast = sinkhorn_flashstyle_symmetric(
        x, y, a, b, blur=EPS ** 0.5, scaling=0.9, rho_x=rho_x, rho_y=rho_y, allow_tf32=False, fused=fused,
        return_prelast=True)
    for f_, g_ in ((f, g), (f_prelast, g_prelast)):
        mass = (b * torch.exp(-g_ / rho_y)).sum() if rho_y is not None else (a * torch.exp(-f_ / rho_x)).sum()
        assert abs(mass.item() - 1.0) < 1e-4


def test_batched_potentials_follow_the_backend():
    """Batched potentials=True must run the requested backend: few iterations, where alternating and symmetric
    iterates still differ."""
    x, y, _, _ = _problem()
    xb, yb = torch.stack([x, x + 0.1]), torch.stack([y, y])
    kwargs = dict(backend="alternating", use_epsilon_scaling=False, eps=EPS, n_iters=20, allow_tf32=False,
                  potentials=True, reach=1.0)
    fb, gb = SamplesLoss("sinkhorn", **kwargs)(xb, yb)
    for i in range(2):
        f, g = SamplesLoss("sinkhorn", **kwargs)(xb[i], yb[i])
        assert torch.allclose(fb[i], f) and torch.allclose(gb[i], g)


def test_alternating_receives_the_stopping_rule(monkeypatch):
    """threshold and inner_iterations reach the alternating solver, on the cost and on the potentials path."""
    seen = []
    for module in (autograd_module, samples_loss_module):
        original = module.sinkhorn_flashstyle_alternating

        def spy(*args, _original=original, **kwargs):
            seen.append((kwargs.get("threshold"), kwargs.get("check_every")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "sinkhorn_flashstyle_alternating", spy)
    x, y, a, b = _problem()
    kwargs = dict(backend="alternating", use_epsilon_scaling=False, eps=EPS, n_iters=50, threshold=1e-3,
                  inner_iterations=7)
    SamplesLoss("sinkhorn", **kwargs)(a, x, b, y)
    SamplesLoss("sinkhorn", potentials=True, **kwargs)(a, x, b, y)
    assert seen == [(1e-3, 7), (1e-3, 7)]


@functools.lru_cache(maxsize=None)
def _gaussian_reference(name):
    """Two Gaussian clouds in 16 dimensions with no outliers, and their converged semi-unbalanced cost at blur 0.1."""
    torch.manual_seed(0)
    x = torch.randn(256, 16, device="cuda")
    y = torch.randn(256, 16, device="cuda")
    a = torch.full((256,), 1.0 / 256, device="cuda")
    b = torch.full((256,), 1.0 / 256, device="cuda")
    X, Y, A, B = x.double(), y.double(), a.double(), b.double()
    rho_x, rho_y = _rho(SETTINGS[name])
    f = g = None
    for eps in torch.logspace(1.0, -2.0, 400).tolist():  # anneal, then converge at eps = 0.01
        f, g = unbalanced_sinkhorn_ref(X, Y, A, B, eps, 1, rho_x, rho_y, f_init=f, g_init=g)
    f, g = unbalanced_sinkhorn_ref(X, Y, A, B, 0.01, 30000, rho_x, rho_y, f_init=f, g_init=g)
    value, _, P = unbalanced_value_and_grad_ref(X, Y, A, B, f, g, 0.01, rho_x, rho_y)
    # The expected value is only trusted once the plan meets the optimality conditions.
    r_expected = A if rho_x is None else A * torch.exp(-f / rho_x)
    c_expected = B if rho_y is None else B * torch.exp(-g / rho_y)
    assert (P.sum(dim=1) - r_expected).abs().sum() < 1e-5
    assert (P.sum(dim=0) - c_expected).abs().sum() < 1e-5
    return (x, y, a, b), value.item()


@pytest.mark.parametrize("name", ["reach_x only", "reach_y only"])
def test_semi_unbalanced_epsilon_scaling(name):
    """With one marginal relaxed, Sinkhorn converges slowly along a shift (f + c, g - c) of the potentials. The
    solvers remove that shift in closed form at every step, so an epsilon-scaling run lands near the converged
    value: without it, scaling 0.9 gave 6.09 against 11.47 on this problem."""
    (x, y, a, b), value_ref = _gaussian_reference(name)
    cost = SamplesLoss("sinkhorn", blur=0.1, scaling=0.9, allow_tf32=False, **SETTINGS[name])(a, x, b, y)
    assert abs(cost.item() - value_ref) <= 1e-2 * abs(value_ref)


def test_ott_convention_refuses_relaxed_marginals():
    """The OTT-convention branch of the alternating solver is balanced only and says so."""
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

    x, y, a, b = _problem()
    with pytest.raises(NotImplementedError):
        sinkhorn_flashstyle_alternating(x, y, a, b, eps=EPS, n_iters=5, rho_y=1.0, ott_convention=True)
