"""Gradients and Hessian-vector products of SamplesLoss beyond converged parity, and the warnings given when a
requested convergence check is not confirmed. Each test fails on 0.4.0. Tolerances are those of
test_unbalanced_reference.py.

- A loss inside a nonlinear function: the double backward also returns the derivative with respect to the incoming
  gradient scale, so the HVP of L^2 is 2 L H v + 2 grad L <grad L, v>.
- Before convergence, with the final extrapolation, the gradient is GeomLoss's gradient through the stopped last
  update (balanced or one common reach, GeomLoss's relaxed factor removed), and the three problems of the debiased
  cost share one epsilon schedule, as in GeomLoss. It is not the total derivative of the finite run.
- A solve whose threshold is never confirmed, and an HVP whose conjugate gradients do not confirm convergence, warn;
  conjugate gradients confirm convergence on the true residual.
- Clouds shifted far from the origin agree with the centred clouds within tolerance; centering in fp32 keeps the
  tested fp16 and bf16 gaps to fp32 resolution and does not overflow fp16.
- With a label cost, lambda_x weights the whole feature term, and problems above the size where the solver switches
  to separate kernels still solve.
- A user eps_list is the schedule that runs.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric

from .test_unbalanced_reference import _kwargs, _problem


def _value_and_grads(loss, a, x, b, y):
    xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    cost = loss(a, xr, b, yr)
    grad_x, grad_y = torch.autograd.grad(cost, [xr, yr])
    return cost.item(), grad_x, grad_y


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("half_cost", [False, True])
def test_hvp_of_a_composed_loss(backend, half_cost):
    """One point to one point: L = cs (x - y)^2, so d^2 (L^2) / dx^2 = 12 cs^2 at x - y = 1. The missing chain-rule
    term gave 4 cs^2; 0.4.0 also had the half-cost prefactor defect and gives -3 in the half-cost case. The Schur
    system is zero here, so the conjugate gradients must also confirm convergence (no warning)."""
    x = torch.ones(1, 1, device="cuda", requires_grad=True)
    y = torch.zeros_like(x)
    loss = SamplesLoss("sinkhorn", backend=backend, half_cost=half_cost, use_epsilon_scaling=False, eps=0.25,
                       n_iters=50, allow_tf32=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (grad,) = torch.autograd.grad(loss(x, y) ** 2, x, create_graph=True)
        (hv,) = torch.autograd.grad(grad.sum(), x)
    exact = 12 * (0.5 if half_cost else 1.0) ** 2
    assert abs(hv.item() - exact) <= 1e-5 * exact
    assert not [w for w in caught if "conjugate gradients" in str(w.message)]


@pytest.mark.parametrize("reach", [None, 0.3])
def test_gradient_before_convergence_matches_geomloss(reach):
    """At GeomLoss's default scaling 0.5 and blur 0.1 the solve is not converged; the debiased gradient must still be
    GeomLoss's. GeomLoss's relaxed gradients carry a factor (rho + eps/2) / (rho + eps), which is divided out.
    Separate self schedules (0.4.0) gave 6.2e-4 here, the mass factor at the prelast potentials 1.2e-1."""
    geomloss = pytest.importorskip("geomloss")
    gen = torch.Generator().manual_seed(89)
    x = (0.12 * torch.randn(8, 2, generator=gen)).cuda()
    y = (0.4 * torch.randn(9, 2, generator=gen) + 1.1).cuda()
    a = torch.full((8,), 1 / 8, device="cuda")
    b = torch.full((9,), 1 / 9, device="cuda")
    k = 1.0 if reach is None else (reach ** 2 + 0.01 / 2) / (reach ** 2 + 0.01)
    xr = x.double().clone().requires_grad_(True)
    reference = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.1, reach=reach, backend="tensorized")
    (g_ref,) = torch.autograd.grad(reference(a.double(), xr, b.double(), y.double()), [xr])
    g_ref = g_ref / k
    ours = SamplesLoss("sinkhorn", blur=0.1, reach=reach, half_cost=True, debias=True, allow_tf32=False)
    _, gx, _ = _value_and_grads(ours, a, x, b, y)
    assert (gx.double() - g_ref).norm() <= 1e-4 * g_ref.norm()


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
def test_a_solve_above_its_threshold_warns(backend):
    x, y, a, b = _problem()
    short = SamplesLoss("sinkhorn", backend=backend, use_epsilon_scaling=False, eps=0.01, n_iters=20,
                        threshold=1e-12, inner_iterations=5, allow_tf32=False)
    with pytest.warns(RuntimeWarning, match="not confirmed"):
        short(a, x, b, y)
    converged = SamplesLoss("sinkhorn", backend=backend, use_epsilon_scaling=False, eps=0.25, n_iters=5000,
                            threshold=1e-4, inner_iterations=10, allow_tf32=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converged(a, x, b, y)
    assert not [w for w in caught if "not confirmed" in str(w.message)]


def test_an_hvp_whose_conjugate_gradients_stop_early_warns():
    x, y, a, b = _problem()
    loss = SamplesLoss("sinkhorn", hvp_max_cg_iter=1, **_kwargs("symmetric", "balanced"))
    xr = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(a, xr, b, y), xr, create_graph=True)
    with pytest.warns(RuntimeWarning, match="conjugate gradients"):
        torch.autograd.grad((grad * torch.ones_like(grad)).sum(), xr)


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("name", ["balanced", "reach"])
def test_clouds_far_from_the_origin(backend, name):
    """Translating both clouds by 100 leaves the problem unchanged; without centering, 0.4.0 moved the value by up
    to 720x and the gradient by up to 6.3 in relative norm, because alpha = |x|^2 lost the precision eps needs."""
    x, y, a, b = _problem()
    loss = SamplesLoss("sinkhorn", **_kwargs(backend, name))
    c0, gx0, gy0 = _value_and_grads(loss, a, x, b, y)
    c1, gx1, gy1 = _value_and_grads(loss, a, x + 100.0, b, y + 100.0)
    assert abs(c1 - c0) <= 1e-5 * max(1.0, abs(c0))
    assert (gx1 - gx0).norm() <= 1e-3 * gx0.norm()
    assert (gy1 - gy0).norm() <= 1e-3 * gy0.norm()


def test_a_user_eps_list_is_the_schedule():
    """0.4.0's symmetric backend ignored eps_list (and diameter) and ran the default schedule."""
    torch.manual_seed(0)
    x = torch.rand(300, 2, device="cuda")
    y = torch.rand(250, 2, device="cuda") + 0.3
    a = torch.full((300,), 1 / 300, device="cuda")
    b = torch.full((250,), 1 / 250, device="cuda")
    schedule = [1.0, 0.3, 0.1, 0.03] + [0.01] * 3
    f, g = sinkhorn_flashstyle_symmetric(x, y, a, b, eps_list=schedule, allow_tf32=False)
    expected = ((a * f).sum() + (b * g).sum()).item()
    cost = SamplesLoss("sinkhorn", blur=0.1, eps_list=schedule, allow_tf32=False)(a, x, b, y).item()
    assert abs(cost - expected) <= 1e-5 * max(1.0, abs(expected))
    f2, g2 = SamplesLoss("sinkhorn", blur=0.1, eps_list=schedule, allow_tf32=False, potentials=True)(a, x, b, y)
    assert (f2 - f).abs().max() < 2e-4 and (g2 - g).abs().max() < 2e-4


def test_centering_keeps_low_precision_inputs_exact():
    """Centering runs in fp32: in fp16 the shift would erase the 0.125 gap (0.0 and 0.125 both round to the same value
    near -980) and turn finite +-65504 into -inf."""
    from flash_sinkhorn.samples_loss import _center_coords

    a = torch.tensor([0.01, 0.01, 0.98], device="cuda")
    b = torch.ones(1, device="cuda")
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.tensor([[0.0], [0.125], [1000.0]], device="cuda", dtype=dtype)
        y = torch.tensor([[1000.0]], device="cuda", dtype=dtype)
        xc, _ = _center_coords(x, y, a, b, batched=False)
        resolution = torch.finfo(torch.float32).eps * xc.abs().max().item()  # fp32 rounding of the centred values
        assert abs((xc[1] - xc[0]).item() - 0.125) <= resolution
    x = torch.tensor([[-65504.0], [65504.0]], device="cuda", dtype=torch.float16)
    y = torch.tensor([[65504.0]], device="cuda", dtype=torch.float16)
    xc, yc = _center_coords(x, y, torch.tensor([0.01, 0.99], device="cuda"), b, batched=False)
    assert torch.isfinite(xc).all() and torch.isfinite(yc).all()


@pytest.mark.parametrize("half_cost", [False, True])
@pytest.mark.parametrize("lambda_x", [2.0, 0.0])
def test_label_cost_weights_the_whole_feature_term(half_cost, lambda_x):
    """One point to one point with W = 0: the cost is cs * lambda_x * |x - y|^2 and the gradient 2 cs lambda_x (x - y).
    The shifts used to leave lambda_x out of |x|^2 + |y|^2: 1.5 instead of 2 at lambda_x = 2, 0.5 instead of 0."""
    x = torch.ones(1, 1, device="cuda", requires_grad=True)
    y = torch.zeros_like(x)
    labels = torch.zeros(1, device="cuda", dtype=torch.long)
    loss = SamplesLoss("sinkhorn", half_cost=half_cost, use_epsilon_scaling=False, eps=0.25, n_iters=30,
                       allow_tf32=False, label_cost_matrix=torch.zeros(1, 1, device="cuda"), lambda_x=lambda_x,
                       lambda_y=1.0)
    value = loss(x, y, label_x=labels, label_y=labels)
    (grad,) = torch.autograd.grad(value, x)
    expected = (0.5 if half_cost else 1.0) * lambda_x
    assert abs(value.item() - expected) <= 1e-5 * max(1.0, expected)
    assert abs(grad.item() - 2 * expected) <= 1e-5 * max(1.0, 2 * expected)


def test_label_cost_solves_above_the_separate_kernel_threshold():
    # From 30,000 points the solver switches to separate kernels, which have no label cost; with an active label term
    # it must keep the fused kernel instead of raising. Identical labelled clouds have a zero debiased cost.
    torch.manual_seed(0)
    x = torch.rand(30_016, 2, device="cuda")
    labels = torch.randint(0, 3, (len(x),), device="cuda")
    W = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], device="cuda")
    loss = SamplesLoss(blur=0.1, half_cost=True, debias=True, label_cost_matrix=W, lambda_x=1.0, lambda_y=1.0,
                       autotune=False)
    value = loss(x, x, label_x=labels, label_y=labels)
    assert torch.isfinite(value)
    assert abs(value.item()) < 1e-5


def test_conjugate_gradients_confirm_on_the_true_residual():
    """An fp32 SPD system with condition number 1e5: the recursively updated residual drifts below the tolerance while
    b - A x stays far above it (0.4.0 reported convergence at a true residual of 4e-3 against 5e-6)."""
    from flash_sinkhorn.cg import conjugate_gradient

    gen = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=gen))
    A = (q * torch.logspace(0, 5, 32)) @ q.T
    rhs = torch.randn(32, generator=gen)
    x, info = conjugate_gradient(lambda v: A @ v, rhs, max_iter=1000, rtol=1e-6, atol=1e-6)
    true_residual = torch.linalg.vector_norm(rhs - A @ x).item()
    tol = max(1e-6, 1e-6 * info.cg_initial_residual)
    assert info.cg_converged == (true_residual <= tol)
    assert math.isclose(info.cg_residual, true_residual, rel_tol=1e-6)
