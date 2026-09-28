"""Fine stage: symmetric Sinkhorn updates on the points, restricted to a block mask.

The mask is rebuilt at every update from the potentials it serves: a box screen
over block bounding boxes, exact verification of the surviving tiles, and the
transpose for the column half-step. Each omitted tile contributes at most
``delta_rel`` of its block mass to any row or column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from ..kernels.sinkhorn_flashstyle_sqeuclid import flashsinkhorn_symmetric_step
from .block_mask import LOG_ZERO_SENTINEL, block_log_mass, build_block_mask_csr, csr_transpose
from .mask_refine import refine_mask_exact
from .sparse_step import sparse_symmetric_step

# Update densely when two points must be half the cloud's diameter apart for their kernel to fall below
# GEOMETRY_DELTA / B: a mask would then keep nearly every tile.
GEOMETRY_DELTA = 1e-10
# Omitted mass of the whole mask, summed over rows or columns, is at most tol / MASK_TOL_DIVISOR.
MASK_TOL_DIVISOR = 4
# A mask that admits no block for more than 1% of either side's block rows (of positive mass) is unstable.
EMPTY_ROW_LIMIT = 0.01


class UnstableMask(RuntimeError):
    """The fine masks have stopped describing the transport: the potentials left some rows so short of
    mass that whole block rows were pruned, and repairing them densely would cost more than the solve."""


@dataclass(frozen=True)
class FineState:
    f_hat: torch.Tensor
    g_hat: torch.Tensor
    ordinary_steps: int = 0
    kind: str = "ordinary"  # "terminal" for an unaveraged preview, which never continues


def mask_delta_rel(geo: dict, tol: float) -> float:
    return tol / (MASK_TOL_DIVISOR * max(geo["N_B"], geo["M_B"]))


@torch.no_grad()
def warm_state(geo: dict, f: torch.Tensor, g: torch.Tensor) -> FineState:
    """Sort and pad physical potentials (input order) and store them shifted."""
    fs = F.pad(f.float()[geo["perm_x"]], (0, len(geo["x_s"]) - len(f)), value=0.0)
    gs = F.pad(g.float()[geo["perm_y"]], (0, len(geo["y_s"]) - len(g)), value=0.0)
    return FineState((fs - geo["alpha"]).contiguous(), (gs - geo["beta"]).contiguous())


@torch.no_grad()
def physical_pair(geo: dict, state: FineState) -> Tuple[torch.Tensor, torch.Tensor]:
    return ((state.f_hat + geo["alpha"])[:geo["n"]][geo["inv_perm_x"]],
            (state.g_hat + geo["beta"])[:geo["m"]][geo["inv_perm_y"]])


def dense_by_geometry(geo: dict, eps: float) -> bool:
    return eps * (-math.log(GEOMETRY_DELTA) + math.log(geo["B"])) >= (geo["L"] / 2.0) ** 2


@torch.no_grad()
def fine_step(geo: dict, state: FineState, eps: float, delta_rel: float, *, alpha: float = 0.5):
    """One symmetric update; ``alpha`` 1/2 continues the solve, 1 gives a terminal preview.

    Returns the new state and ``(candidate tiles, admitted tiles)``, or ``None`` for a dense update.
    """
    if state.kind != "ordinary":
        raise ValueError("a terminal preview cannot be continued")
    B = geo["B"]
    x, y = geo["x_s"], geo["y_s"]
    if dense_by_geometry(geo, eps):
        f, g = flashsinkhorn_symmetric_step(
            x, y, state.f_hat, state.g_hat, geo["log_a"], geo["log_b"], eps, cost_scale=1.0,
            alpha=alpha, damping_f=1.0, damping_g=1.0, allow_tf32=True, use_exp2=True, autotune=True)
        tiles = None
    else:
        crow, col = build_block_mask_csr(
            geo["x_lo"], geo["x_hi"], geo["y_lo"], geo["y_hi"], state.f_hat, state.g_hat,
            geo["alpha"], geo["beta"], geo["log_a"], geo["log_b"], eps, B=B, delta_rel=delta_rel)
        candidates = int(col.numel())
        crow, col = refine_mask_exact(x, y, state.f_hat, state.g_hat, geo["log_a"], geo["log_b"], eps,
                                      crow, col, B=B, delta_rel=delta_rel)
        crow_t, col_t = csr_transpose(crow, col, geo["N_B"], geo["M_B"])
        empty = max(_empty_fraction(crow, geo["log_a"], B), _empty_fraction(crow_t, geo["log_b"], B))
        if empty > EMPTY_ROW_LIMIT:
            raise UnstableMask(f"the mask admits no block for {empty:.1%} of one side's block rows")
        f, g = sparse_symmetric_step(x, y, state.f_hat, state.g_hat, geo["log_a"], geo["log_b"], eps,
                                     crow, col, crow_t, col_t, block=B, alpha=alpha)
        tiles = (candidates, int(col.numel()))
    ordinary = alpha == 0.5
    new = FineState(f, g, state.ordinary_steps + int(ordinary), "ordinary" if ordinary else "terminal")
    return new, tiles


def _empty_fraction(crow: torch.Tensor, log_w: torch.Tensor, B: int) -> float:
    """Share of the block rows of positive mass that admit no block (zero-weight blocks never do)."""
    live = block_log_mass(log_w, len(crow) - 1, B) > LOG_ZERO_SENTINEL / 2
    return float(((crow[1:] == crow[:-1]) & live).sum()) / max(1, int(live.sum()))
