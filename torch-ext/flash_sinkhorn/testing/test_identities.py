"""Swapping the two measures, an exact identity of entropic OT, on both backends at a fixed eps run to convergence.

OT(a, x; b, y) with (reach_x, reach_y) must equal OT(b, y; a, x) with the reaches swapped. This guards the order in
which the source and target penalties are applied, including the closed-form translation of semi-unbalanced
potentials. Tolerances are those of test_unbalanced_reference.py.
"""

from __future__ import annotations

import pytest
import torch

from flash_sinkhorn import SamplesLoss

from .test_unbalanced_reference import EPS, SETTINGS, _kwargs, _problem


def _value_and_grads(loss, a, x, b, y):
    xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    cost = loss(a, xr, b, yr)
    grad_x, grad_y = torch.autograd.grad(cost, [xr, yr])
    return cost.item(), grad_x, grad_y


def _plan(x, y, a, b, f, g):
    C = ((x.double()[:, None, :] - y.double()[None, :, :]) ** 2).sum(dim=-1)
    return a.double()[:, None] * b.double()[None, :] * torch.exp((f.double()[:, None] + g.double()[None, :] - C) / EPS)


@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("name", list(SETTINGS))
def test_swapping_the_measures(backend, name):
    """The potentials swap and so do the gradients. Balanced potentials are unique only up to a constant, so the plan
    is compared instead."""
    x, y, a, b = _problem()
    kwargs = _kwargs(backend, name)
    swapped = {{"reach_x": "reach_y", "reach_y": "reach_x"}.get(k, k): v for k, v in kwargs.items()}
    f, g = SamplesLoss("sinkhorn", potentials=True, **kwargs)(a, x, b, y)
    g2, f2 = SamplesLoss("sinkhorn", potentials=True, **swapped)(b, y, a, x)
    if name == "balanced":
        P2 = _plan(x, y, a, b, f2, g2)
        assert (P2.sum(dim=1) - a.double()).abs().sum() < 1e-4
        assert (P2.sum(dim=0) - b.double()).abs().sum() < 1e-4
        assert (P2 - _plan(x, y, a, b, f, g)).abs().sum() < 1e-4
    else:
        assert (f2 - f).abs().max() < 2e-4
        assert (g2 - g).abs().max() < 2e-4

    c, gx, gy = _value_and_grads(SamplesLoss("sinkhorn", **kwargs), a, x, b, y)
    c2, gy2, gx2 = _value_and_grads(SamplesLoss("sinkhorn", **swapped), b, y, a, x)
    assert abs(c2 - c) <= 1e-5 * max(1.0, abs(c))
    assert (gx2 - gx).norm() <= 1e-3 * gx.norm()
    assert (gy2 - gy).norm() <= 1e-3 * gy.norm()
