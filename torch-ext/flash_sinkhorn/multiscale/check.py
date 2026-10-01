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
FP64_ROWS, FP64_COLS = 64, 2**20  # tile of the fp64 re-check


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


@torch.no_grad()
def _fp64_deviation(x, source_pot, idx, y, target_pot, target_weight, eps: float) -> torch.Tensor:
    """|r_i/a_i - 1| of the sampled sources ``idx`` against every target, in fp64: the check's statistic without
    the fp32 shifted potentials, whose rounding reads high where ulp(|x|^2) approaches tol * eps."""
    log_b = torch.log(target_weight.double())       # zero weights drop out exactly
    out = []
    for r0 in range(0, len(idx), FP64_ROWS):
        rows = idx[r0:r0 + FP64_ROWS]
        xi, fi = x[rows].double(), source_pot[rows].double()
        top = torch.full((len(rows),), -math.inf, dtype=torch.float64, device=x.device)
        acc = torch.zeros_like(top)
        for c0 in range(0, len(y), FP64_COLS):
            yj = y[c0:c0 + FP64_COLS].double()
            cost = (xi * xi).sum(1)[:, None] + (yj * yj).sum(1)[None, :] - 2.0 * xi @ yj.T
            z = (fi[:, None] + target_pot[c0:c0 + FP64_COLS].double()[None, :] - cost) / eps \
                + log_b[c0:c0 + FP64_COLS][None, :]
            new_top = torch.maximum(top, z.max(1).values)
            shift = torch.where(new_top == -math.inf, 0.0, new_top)  # a tile of zero weights only stays at 0
            acc = acc * torch.exp(top - shift) + torch.exp(z - shift[:, None]).sum(1)
            top = new_top
        out.append(torch.expm1(top + torch.log(acc)).abs())
    return torch.cat(out)


def _stats(values: torch.Tensor) -> list:
    mean = values.mean()
    se = values.std(correction=1) / math.sqrt(values.numel())
    return [float(mean), float(se), float(mean + SAMPLE_Z * se)]


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

    def evaluate_fp64(self, f: torch.Tensor, g: torch.Tensor) -> dict:
        """The same two-stage decision on the same samples, with the statistic in fp64 (much slower: it is meant
        for one candidate, not for scheduling)."""
        def stage(spec):
            count, seed = spec
            if count not in self._stages:
                self._stages[count] = (_direction(self.x, self.a, self._targets[0], count, seed),
                                       _direction(self.y, self.b, self._targets[1], count, seed))
            rows, cols = self._stages[count]
            row = _stats(_fp64_deviation(self.x, f, rows.idx, self.y, g, self.b, self.eps))
            col = _stats(_fp64_deviation(self.y, g, cols.idx, self.x, f, self.a, self.eps))
            finite = all(math.isfinite(v) for v in row + col)
            return dict(finite=finite, row=row if finite else None, col=col if finite else None,
                        score=max(row[2], col[2]) if finite else None)

        pair_finite = bool(torch.isfinite(f).all()) and bool(torch.isfinite(g).all())
        first = stage(STAGE_ONE)
        action = _stage_one_action(first, self.tol)
        final = stage(STAGE_TWO) if action == "enlarge" else first
        accepted = bool(pair_finite and final["finite"] and final["score"] <= self.tol)
        rank = first["score"] if (pair_finite and first["finite"] and first["score"] >= 0) else None
        return dict(accepted=accepted, first=first, final=final, enlarged=action == "enlarge",
                    rank=rank, pair_finite=pair_finite)

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
