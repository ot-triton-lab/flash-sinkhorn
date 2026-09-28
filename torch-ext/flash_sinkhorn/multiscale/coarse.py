"""Coarse stage: annealed Sinkhorn on cell centroids, the lift to the points, and the stall rule.

Potentials are kept shifted, ``f_hat = f - ||x||^2``, for the full squared-Euclidean cost.
Both half-steps of an update read the same incoming pair (symmetric updates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import torch

from ..kernels._common import log_weights
from ..kernels.sinkhorn_flashstyle_sqeuclid import flashsinkhorn_lse, flashsinkhorn_lse_fused
from .geometry import Cells, pad_columns

# Stall rule for coarse updates continued at the target epsilon.
CONTINUATION_BATCH = 64
CONTINUATION_MAX_UPDATES = 2048
CONTINUATION_MIN_PROGRESS = 0.01
CONTINUATION_THRESHOLD_FRACTION = 0.5  # stop once D <= tau/2 ...
CONTINUATION_THRESHOLD_CHECKS = 2      # ... on this many consecutive batches
CONTINUATION_SLOW_CHECKS = 3           # or after this many consecutive slow batches

# Fixed launch of the lift (no autotuning on this rectangular shape).
LIFT_LAUNCH = dict(autotune=False, block_m=128, block_n=128, block_k=16, num_warps=4, num_stages=2)


@dataclass
class CoarseState:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    log_a: torch.Tensor
    log_b: torch.Tensor
    f_hat: torch.Tensor
    g_hat: torch.Tensor
    averaged_steps: int = 0


@torch.no_grad()
def coarse_state(cells_x: Cells, cells_y: Cells) -> CoarseState:
    """Zero physical potentials on the centroids, stored shifted."""
    x, y = pad_columns(cells_x.points), pad_columns(cells_y.points)
    alpha, beta = (x ** 2).sum(dim=1), (y ** 2).sum(dim=1)
    return CoarseState(x, y, cells_x.mass, cells_y.mass, alpha, beta,
                       log_weights(cells_x.mass), log_weights(cells_y.mass),
                       -alpha.clone(), -beta.clone())


@torch.no_grad()
def coarse_updates(state: CoarseState, schedule: Sequence[float], *, alpha: float = 0.5) -> CoarseState:
    """Symmetric updates at each epsilon of ``schedule``; ``alpha`` = 1 replaces, 1/2 averages."""
    f, g = state.f_hat, state.g_hat
    for eps in schedule:
        eps = float(eps)
        kw = dict(cost_scale=1.0, damping=1.0, allow_tf32=True, use_exp2=True, autotune=True)
        fc = flashsinkhorn_lse(state.x, state.y, (g + eps * state.log_b) / eps, eps, **kw)
        gc = flashsinkhorn_lse(state.y, state.x, (f + eps * state.log_a) / eps, eps, **kw)
        f, g = (fc, gc) if alpha == 1.0 else (0.5 * f + 0.5 * fc, 0.5 * g + 0.5 * gc)
    averaged = state.averaged_steps + (len(schedule) if alpha == 0.5 else 0)
    return replace(state, f_hat=f, g_hat=g, averaged_steps=averaged)


@torch.no_grad()
def lift(geo: dict, state: CoarseState, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One half-step from each point against the coarse records; physical potentials in input order."""
    # Form the physical coarse potentials, then shift them for the point-to-cell transforms.
    f_c, g_c = state.f_hat + state.alpha, state.g_hat + state.beta
    x_s, y_s = geo["x_s"], geo["y_s"]
    xc, yc = state.x, state.y
    f_hat_c = f_c.float() - (xc * xc).sum(dim=1)
    g_hat_c = g_c.float() - (yc * yc).sum(dim=1)
    kw = dict(cost_scale=1.0, damping=1.0, allow_tf32=True, use_exp2=True, **LIFT_LAUNCH)
    f_hat = flashsinkhorn_lse_fused(x_s, yc, g_hat_c, torch.log(state.b), float(eps), **kw)
    g_hat = flashsinkhorn_lse_fused(y_s, xc, f_hat_c, torch.log(state.a), float(eps), **kw)
    f = (f_hat + (x_s * x_s).sum(dim=1)).contiguous()
    g = (g_hat + (y_s * y_s).sum(dim=1)).contiguous()
    return f[:geo["n"]][geo["inv_perm_x"]], g[:geo["m"]][geo["inv_perm_y"]]


def _stall_side(old: torch.Tensor, new: torch.Tensor, mass: torch.Tensor, eps: float):
    """Mass-weighted ``|log|`` marginal change of one side, in fp64, or None when invalid.

    Invalid means nonfinite potentials or masses, negative or zero total mass, or a
    marginal change ``exp(ell) - 1`` that overflows; the stall rule then discards the batch.
    """
    old64, new64, m = old.double(), new.double(), mass.double()
    ell = (-2.0 / eps) * (new64 - old64)
    value = (m * ell.abs()).sum()
    change = (m * torch.expm1(ell).abs()).sum()
    valid = bool(torch.isfinite(old64).all() & torch.isfinite(new64).all() & torch.isfinite(m).all()
                 & torch.isfinite(value) & torch.isfinite(change) & (m >= 0).all()) and float(m.sum()) > 0
    return float(value) if valid else None


def stall_statistic(before: CoarseState, after: CoarseState, eps: float) -> Optional[float]:
    """D = larger mass-weighted |log| marginal change of one averaged update, or None if invalid."""
    row = _stall_side(before.f_hat, after.f_hat, before.a, eps)
    col = _stall_side(before.g_hat, after.g_hat, before.b, eps)
    if row is None or col is None or not math.isfinite(row) or not math.isfinite(col):
        return None
    return max(row, col)


@torch.no_grad()
def continue_coarse(state: CoarseState, eps: float, tau: float) -> Tuple[CoarseState, bool, int]:
    """Extra averaged updates at eps in batches until D stalls; returns (state, committed, updates).

    An invalid statistic discards the extra batches and returns the incoming state uncommitted.
    """
    threshold = CONTINUATION_THRESHOLD_FRACTION * tau
    current, previous = state, None
    threshold_streak = slow_streak = steps = 0
    while steps < CONTINUATION_MAX_UPDATES:
        before = coarse_updates(current, (eps,) * (CONTINUATION_BATCH - 1))
        after = coarse_updates(before, (eps,))
        steps += CONTINUATION_BATCH
        D = stall_statistic(before, after, eps)
        if D is None:
            return state, False, steps
        threshold_streak = threshold_streak + 1 if D <= threshold else 0
        if previous is not None:
            slow = previous == 0 or (previous - D) / previous < CONTINUATION_MIN_PROGRESS
            slow_streak = slow_streak + 1 if slow else 0
        current, previous = after, D
        if threshold_streak >= CONTINUATION_THRESHOLD_CHECKS or slow_streak >= CONTINUATION_SLOW_CHECKS:
            break
    return current, True, steps
