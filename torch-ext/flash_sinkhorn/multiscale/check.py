"""Sampled marginal check.

Rows ``i ~ a`` and columns ``j ~ b`` are drawn once per solve and reused at every
check. For each sampled row the exact row sum against every target gives
``|r_i/a_i - 1| = |expm1((f_i - T(g)_i)/eps)|``, whose mean estimates
``||P1 - a||_1``; columns estimate ``||P^T 1 - b||_1`` the same way. A check
passes when both means plus three standard errors are below ``tol``. This is
an empirical stopping rule, not a bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from ..kernels._common import log_weights
from ..kernels.sinkhorn_splitn_sqeuclid import flashsinkhorn_lse_splitn, splitn_workspace
from .geometry import pad_columns

SAMPLE_Z = 3.0
STAGE_ONE = (1024, 0)       # (samples per side, seed)
STAGE_TWO = (8192, 159015)  # drawn only when stage one is inconclusive
SPLIT_N = 64


def sample_indices(weights: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """With-replacement draw ``i ~ weights`` from a seeded CPU generator."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if bool((weights.max() == weights.min()).item()):
        return torch.randint(int(weights.numel()), (count,), generator=generator)
    u = torch.rand(count, dtype=torch.float64, generator=generator)
    cdf = torch.cumsum(weights.detach().double(), 0)
    cdf = cdf / cdf[-1]
    return torch.searchsorted(cdf, u.to(weights.device)).cpu()


@dataclass
class _Target:
    """The opposite cloud of one direction, shared by both stages."""
    points: torch.Tensor      # padded to PAD_K columns
    shift: torch.Tensor       # ||y_j||^2
    log_weight: torch.Tensor  # log b_j, with the zero-weight sentinel


@dataclass
class _Direction:
    idx: torch.Tensor
    x_sample: torch.Tensor
    alpha_sample: torch.Tensor
    bias: torch.Tensor
    workspace: Tuple[torch.Tensor, torch.Tensor]


def _target(y: torch.Tensor, weight: torch.Tensor) -> _Target:
    return _Target(pad_columns(y), (y.float() * y.float()).sum(dim=1).contiguous(),
                   log_weights(weight).contiguous())


def _direction(x: torch.Tensor, source_weight: torch.Tensor, target: _Target, count: int,
               seed: int) -> _Direction:
    idx = sample_indices(source_weight, count, seed).to(x.device)
    xs = x.float()[idx]
    return _Direction(idx, pad_columns(xs), (xs * xs).sum(dim=1).contiguous(),
                      torch.empty_like(target.log_weight), splitn_workspace(SPLIT_N, count, x.device))


def _direction_stats(ws: _Direction, target: _Target, source_pot, target_pot, eps: float) -> torch.Tensor:
    """(mean, standard error, mean + z * SE) of |expm1((f_i - T(g)_i)/eps)| over the sample."""
    f_hat = (source_pot.float()[ws.idx] - ws.alpha_sample).contiguous()
    g_hat = (target_pot.float() - target.shift).contiguous()
    torch.add(target.log_weight, g_hat, alpha=1.0 / eps, out=ws.bias)
    transformed = flashsinkhorn_lse_splitn(
        ws.x_sample, target.points, ws.bias, eps, num_splits=SPLIT_N, cost_scale=1.0,
        damping=1.0, allow_tf32=False, use_exp2=True, workspace=ws.workspace)
    values = torch.expm1((f_hat.double() - transformed.double()) / eps).abs()
    mean = values.mean()
    se = values.std(correction=1) / math.sqrt(values.numel())
    return torch.stack((mean, se, mean + SAMPLE_Z * se))


def _stage_one_action(score: dict, tol: float) -> str:
    """accept, reject, or enlarge the sample when the two-sided interval straddles ``tol``."""
    if not score["finite"]:
        return "enlarge"
    sides = (score["row"], score["col"])
    lower = max(max(0.0, v[0] - SAMPLE_Z * v[1]) for v in sides)
    upper = max(v[0] + SAMPLE_Z * v[1] for v in sides)
    if upper <= tol:
        return "accept"
    if lower > tol:
        return "reject"
    return "enlarge"


class SampledCheck:
    """Two-stage sampled check; samples and workspaces are prepared once and reused."""

    def __init__(self, x, y, a, b, eps: float, tol: float):
        self.x, self.y, self.a, self.b = x, y, a, b
        self.eps, self.tol = float(eps), float(tol)
        self._targets = (_target(y, b), _target(x, a))  # row direction reads y, column direction x
        self._stages: Dict[int, Tuple[_Direction, _Direction]] = {}

    def _stage(self, stage: Tuple[int, int], f, g) -> dict:
        count, seed = stage
        if count not in self._stages:
            self._stages[count] = (_direction(self.x, self.a, self._targets[0], count, seed),
                                   _direction(self.y, self.b, self._targets[1], count, seed))
        rows, cols = self._stages[count]
        values = torch.stack((_direction_stats(rows, self._targets[0], f, g, self.eps),
                              _direction_stats(cols, self._targets[1], g, f, self.eps))).cpu()
        finite = bool(torch.isfinite(values).all())
        return dict(finite=finite, row=values[0].tolist() if finite else None,
                    col=values[1].tolist() if finite else None,
                    score=max(float(values[0, 2]), float(values[1, 2])) if finite else None)

    def evaluate(self, f: torch.Tensor, g: torch.Tensor) -> dict:
        """``accepted``; ``rank``, the stage-one score used for scheduling and for the fallback
        return (None unless the pair and the score are finite); and both stage scores."""
        pair_finite = bool(torch.isfinite(f).all()) and bool(torch.isfinite(g).all())
        first = self._stage(STAGE_ONE, f, g)
        action = _stage_one_action(first, self.tol)
        final = self._stage(STAGE_TWO, f, g) if action == "enlarge" else first
        accepted = pair_finite and final["finite"] and final["score"] <= self.tol
        rank = first["score"] if (pair_finite and first["finite"] and first["score"] >= 0) else None
        return dict(accepted=accepted, first=first, final=final, enlarged=action == "enlarge",
                    rank=rank, pair_finite=pair_finite)
