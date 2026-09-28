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
from . import block_mask
from .block_mask import LOG_ZERO_SENTINEL, block_log_mass, csr_transpose
from .mask_refine import refine_mask_exact
from .sparse_step import sparse_symmetric_step

# Update densely when two points must be half the cloud's diameter apart for their kernel to fall below
# GEOMETRY_DELTA / B: a mask would then keep nearly every tile.
GEOMETRY_DELTA = 1e-10
# Omitted mass of the whole mask, summed over rows or columns, is at most tol / MASK_TOL_DIVISOR.
MASK_TOL_DIVISOR = 4
# The dense repair of a mask's empty block rows (of positive mass, both sides together) may cost at most this
# multiple of one of the update's sparse half-steps; a mask that needs more is unstable.
REPAIR_COST_RATIO = 1


class UnstableMask(RuntimeError):
    """The fine masks have stopped describing the transport: the potentials left some rows so short of
    mass that whole block rows were pruned, and repairing them densely would cost more than the update."""


class MaskTooLarge(RuntimeError):
    """The mask an update needs exceeds the int32 indices or the device memory: the fine stage cannot run
    at this blur and size."""


class WorkBudget(RuntimeError):
    """The next launch would take the fine stage past its work budget (solver.WORK_PER_POINT)."""


class WorkMeter:
    """Point pairs the fine stage launches, charged before each launch: a charge past ``limit`` raises WorkBudget
    and launches nothing. Work of retried and interrupted attempts stays charged."""

    def __init__(self, limit: float = math.inf):
        self.limit, self.spent = limit, 0

    def require(self, work: float, what: str) -> None:
        if self.spent + work > self.limit:
            raise WorkBudget(f"{what} would evaluate {work:.3g} point pairs, more than the "
                             f"{self.limit - self.spent:.3g} left")

    def charge(self, work: float, what: str) -> None:
        self.require(work, what)
        self.spent += work


def _attempt(fn):
    """``(fn(), None)``, or ``(None, message)`` on a device out-of-memory error. A retry happens outside the
    handler, after the failed attempt's tensors are released."""
    try:
        return fn(), None
    except torch.cuda.OutOfMemoryError as ex:
        return None, str(ex).split("\n")[0][:200] or "out of device memory"


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
def fine_step(geo: dict, state: FineState, eps: float, delta_rel: float, *, alpha: float = 0.5,
              meter: WorkMeter = None):
    """One symmetric update; ``alpha`` 1/2 continues the solve, 1 gives a terminal preview.

    Returns the new state and ``(candidate tiles, admitted tiles)``, or ``None`` for a dense update.
    Charges each refinement, update and repair to ``meter`` before launching it, and raises WorkBudget
    before one it cannot afford, UnstableMask for non-finite potentials or an unaffordable repair, and
    MaskTooLarge when the update does not fit the int32 indices or the device memory.
    """
    if state.kind != "ordinary":
        raise ValueError("a terminal preview cannot be continued")
    B = geo["B"]
    x, y = geo["x_s"], geo["y_s"]
    if bool((~torch.isfinite(state.f_hat) & (geo["a_s"] > 0)).any() | (~torch.isfinite(state.g_hat) & (geo["b_s"] > 0)).any()):
        raise UnstableMask("the potentials are not finite on points of positive mass")

    meter = WorkMeter() if meter is None else meter
    if dense_by_geometry(geo, eps):
        meter.charge(2 * len(x) * len(y), "a dense update")
        result, err = _attempt(lambda: flashsinkhorn_symmetric_step(
            x, y, state.f_hat, state.g_hat, geo["log_a"], geo["log_b"], eps, cost_scale=1.0,
            alpha=alpha, damping_f=1.0, damping_g=1.0, allow_tf32=True, use_exp2=True, autotune=True))
        if err:
            raise MaskTooLarge(f"a dense update does not fit in device memory: {err}")
        f, g = result
        tiles = None
    else:
        candidates, crow, col = admitted_mask(geo, state, eps, delta_rel, meter)
        half = int(col.numel()) * B * B
        result, err = _attempt(lambda: csr_transpose(crow, col, geo["N_B"], geo["M_B"]))
        if err:
            raise MaskTooLarge(f"the transpose of a {int(col.numel()):,}-tile mask does not fit: {err}")
        crow_t, col_t = result
        empty_x, empty_y = _empty_rows(crow, geo["log_a"], B), _empty_rows(crow_t, geo["log_b"], B)
        repair = (empty_x * len(y) + empty_y * len(x)) * B
        if repair > REPAIR_COST_RATIO * half:
            raise UnstableMask(f"the mask admits no block for {empty_x} source and {empty_y} target block rows of "
                               f"positive mass; repairing them densely ({repair:.3g} pair evaluations) would cost "
                               f"more than a sparse half-step ({half:.3g})")
        meter.charge(2 * half + repair, "the half-steps and repairs")
        result, err = _attempt(lambda: sparse_symmetric_step(
            x, y, state.f_hat, state.g_hat, geo["log_a"], geo["log_b"], eps, crow, col, crow_t, col_t,
            block=B, alpha=alpha))
        if err:
            raise MaskTooLarge(f"the half-steps of a {int(col.numel()):,}-tile mask do not fit: {err}")
        f, g = result
        tiles = (candidates, int(col.numel()))
    ordinary = alpha == 0.5
    new = FineState(f, g, state.ordinary_steps + int(ordinary), "ordinary" if ordinary else "terminal")
    return new, tiles


@torch.no_grad()
def admitted_mask(geo: dict, state: FineState, eps: float, delta_rel: float, meter: WorkMeter = None):
    """``(candidates, crow, col)``: the box screen's candidates refined exactly into the admitted mask.

    When the candidates do not fit one CSR, or a refinement runs out of device memory, consecutive
    chunks of block rows (fewer after each failure) are screened and refined in turn and only their
    admitted tiles are kept; the admitted CSR is the same. Raises MaskTooLarge when a single block row
    or the admitted mask does not fit, and WorkBudget before a refinement ``meter`` cannot afford (every
    attempt is charged, retries included).
    """
    B = geo["B"]
    x, y = geo["x_s"], geo["y_s"]
    try:
        screen, err = _attempt(lambda: block_mask.prepare_screen(
            geo["x_lo"], geo["x_hi"], geo["y_lo"], geo["y_hi"], state.f_hat, state.g_hat,
            geo["alpha"], geo["beta"], geo["log_a"], geo["log_b"], eps, B=B, delta_rel=delta_rel))
    except block_mask.CSRCapacityError as ex:
        raise MaskTooLarge(f"the group screen exceeds the int32 indices: {ex}") from None
    if err:
        raise MaskTooLarge(f"the group screen does not fit in device memory: {err}")
    N_B, M_B = screen["N_B"], screen["M_B"]
    counts = block_mask.screen_counts(screen)
    counts64 = counts.to(torch.int64)
    candidates = int(counts64.sum())
    meter = WorkMeter() if meter is None else meter
    meter.require(candidates * B * B, f"refining {candidates:,} candidate tiles")

    def refine(r0: int, r1: int):
        crow_c, col_c = block_mask.screen_fill(screen, counts, r0, r1)
        return refine_mask_exact(x[r0 * B:r1 * B], y, state.f_hat[r0 * B:r1 * B], state.g_hat,
                                 geo["log_a"][r0 * B:r1 * B], geo["log_b"], eps, crow_c, col_c,
                                 B=B, delta_rel=delta_rel, la_block=screen["rows"][3][r0:r1],
                                 lb_block=screen["cols"][3])

    ends = torch.cumsum(counts64, 0).cpu()
    rows_admitted, cols, admitted, r0, most = [], [], 0, 0, N_B
    while r0 < N_B:
        room = block_mask.csr_capacity(N_B, M_B, screen["device"])
        base = int(ends[r0 - 1]) if r0 else 0
        r1 = min(N_B, r0 + most, max(r0 + 1, int(torch.searchsorted(ends, base + room, right=True))))
        if int(ends[r1 - 1]) - base > room:
            raise MaskTooLarge(f"block row {r0} alone has more candidates than a CSR can hold ({room:,})")
        meter.charge((int(ends[r1 - 1]) - base) * B * B, f"refining block rows {r0}-{r1 - 1}")
        result, err = _attempt(lambda: refine(r0, r1))
        if err:
            if r1 - r0 == 1:
                raise MaskTooLarge(f"block row {r0} alone does not fit in device memory: {err}")
            most = (r1 - r0) // 2                          # strictly fewer rows on the next attempt
            continue
        crow_r, col_r = result
        admitted += int(col_r.numel())
        if admitted > block_mask.CSR_INDEX_MAX:
            raise MaskTooLarge(f"the admitted mask exceeds {block_mask.CSR_INDEX_MAX:,} tiles "
                               f"({candidates:,} candidates)")
        rows_admitted.append(crow_r[1:] - crow_r[:-1])
        cols.append(col_r)
        r0 = r1
    if len(cols) == 1:
        return candidates, crow_r, cols[0]
    result, err = _attempt(lambda: (torch.cat(rows_admitted).cumsum(0), torch.cat(cols)))
    if err:
        raise MaskTooLarge(f"the admitted mask of {admitted:,} tiles does not fit in device memory: {err}")
    crow = torch.zeros(N_B + 1, dtype=torch.int32, device=screen["device"])
    crow[1:] = result[0]
    return candidates, crow, result[1]


def _empty_rows(crow: torch.Tensor, log_w: torch.Tensor, B: int) -> int:
    """Block rows of positive mass that admit no block (zero-weight blocks never do, and are not repaired)."""
    live = block_log_mass(log_w, len(crow) - 1, B) > LOG_ZERO_SENTINEL / 2
    return int(((crow[1:] == crow[:-1]) & live).sum())
