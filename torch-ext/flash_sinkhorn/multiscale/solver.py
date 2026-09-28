"""The multiscale solver: a coarse stage on Morton cells, then block-sparse fine updates.

``solve_multiscale`` anneals Sinkhorn on cell centroids, lifts the potentials to the
points and returns them as soon as a sampled check finds the marginal residual
``max(||P1 - a||_1, ||P^T 1 - b||_1)`` below ``tol``, for the full squared-Euclidean
cost with ``P_ij = a_i b_j exp((f_i + g_j - C_ij)/eps)``. If the fine stage stalls,
becomes unstable, reaches its ceiling or its work budget, or needs a mask too large to hold before a check passes, it
returns the candidate with the best stage-one score and reports ``accepted=False``. The coordinates must already be centered and TF32-rounded
by :func:`preprocess_coordinates` (the rounding is checked, the centering is not).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import check as C
from .coarse import (CONTINUATION_BATCH, CONTINUATION_MAX_UPDATES, CONTINUATION_MIN_PROGRESS,
                     CONTINUATION_SLOW_CHECKS, coarse_state, coarse_updates, continue_coarse, lift)
from .fine import (MASK_TOL_DIVISOR, FineState, MaskTooLarge, UnstableMask, WorkBudget, WorkMeter, fine_step,
                   mask_delta_rel, physical_pair, warm_state)
from .sparse_step import REPAIR_QUERY_CHUNK, dense_lse_rows
from .envelope import Envelope, envelope
from .geometry import (INT32_OFFSET_LIMIT, PAD_K, annealing_schedule, block_summaries, build_cells, choose_block_size,
                       choose_cell_level, execution_geometry, joint_sort, prefix_counts)
from ._preprocess import TF32_DROPPED_BITS, round_to_tf32

# The fine stage stops by the coarse continuation's own stall rule, applied to the stage-one score.
FINE_WINDOW = CONTINUATION_BATCH                # the score is compared every 64 ordinary updates;
FINE_MIN_PROGRESS = CONTINUATION_MIN_PROGRESS   # a window that lowers it by less than 1% is slow,
FINE_SLOW_WINDOWS = CONTINUATION_SLOW_CHECKS    # three slow windows in a row stop the stage,
FINE_CEILING = CONTINUATION_MAX_UPDATES         # which never runs more than 2048 updates.
FIXED_PREVIEW = 128  # an unaveraged candidate is always tried at update 128, whatever its score
UNSTABLE_NONFINITE_CHECKS = 2  # two fine-stage checks in a row without a finite score stop it as unstable,
                               # as does a mask whose empty block rows cost too much to repair (fine.REPAIR_COST_RATIO)
WORK_PER_POINT = 4e8  # the fine stage's charged work is at most 4e8 point pairs per point, over (n + m)/2 points

# Check scheduling.
CHECK_INTERVALS = (1, 2, 4, 8, 16)
CHECK_WORK_FRACTION = 0.1       # a check may cost at most this share of the updates between checks
NEAR_TARGET_FACTOR = 2.0        # check every update once the stage-one score is within 2 tol
MILESTONE = 16                  # a check (and, near the target, a terminal preview) every 16 updates
MAX_POINTS = INT32_OFFSET_LIMIT // PAD_K  # per side: offsets up to 16 * rows - 1 into the padded coordinates
FP32_MAX = torch.finfo(torch.float32).max
# The weights may miss a sum of one by at most min(1e-3, tol / MASK_TOL_DIVISOR): the mass defect
# takes no larger share of the residual budget than the mask omission does.
WEIGHT_SUM_TOLERANCE = 1e-3


@dataclass
class MultiscaleInfo:
    """What the solve did. ``returned`` is ``"lift"``, ``"ordinary"`` or ``"terminal"`` (an
    unaveraged preview), at fine update ``returned_step``; ``stop`` is ``"accepted"``, or why the
    fine stage ended without acceptance, ``"stalled"``, ``"unstable"``, ``"ceiling"``, ``"budget"`` (its
    next launch would pass WORK_PER_POINT point pairs per point) or ``"too large"`` (a mask or an update
    would exceed the int32 indices or the device memory)."""
    accepted: bool
    stop: str
    B: int
    t: int
    coarse_cells: Tuple[int, int]
    schedule_entries: int
    continuation_updates: int
    fine_updates: int
    terminal_previews: int
    returned: str
    returned_step: int
    checks: List[dict] = field(default_factory=list)
    tiles: List[Tuple[int, int, int]] = field(default_factory=list)  # (update, candidate, admitted); sparse updates only
    detail: str = ""  # why the fine stage stopped, when it stopped without acceptance
    fine_work: int = 0  # point pairs charged, each before its launch: refinement attempts, half-steps, repairs,
                        # dense updates (autotuning is not charged)
    envelope: Optional[Envelope] = None  # the problem against the warning thresholds (API.md, Limits)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return time.perf_counter() - start, out


def _interval(check_seconds, step_seconds, entry: dict, tol: float, cold: bool) -> int:
    """Check every update near the target, after a new autotune winner or with an unknown cost;
    otherwise as rarely as keeps the checks within CHECK_WORK_FRACTION of the update time."""
    rank = entry["rank"]
    valid = all(v is not None and math.isfinite(v) and v > 0 for v in (check_seconds, step_seconds))
    if rank is None or entry["enlarged"] or rank <= NEAR_TARGET_FACTOR * tol or cold or not valid:
        return 1
    return next((k for k in CHECK_INTERVALS if check_seconds <= CHECK_WORK_FRACTION * k * step_seconds),
                CHECK_INTERVALS[-1])


def _next_check(h: int, cap: int, interval: int) -> int:
    if h == cap:
        return cap
    if h < 2:
        return h + 1
    return min(h + interval, (h // MILESTONE + 1) * MILESTONE, cap)


def _preview_due(h: int, cap: int, entry: dict, tol: float) -> bool:
    """A terminal preview after the second update, at FIXED_PREVIEW, at the cap, and at milestones
    near the target."""
    if h in (2, FIXED_PREVIEW, cap):
        return True
    if h < 2 or h % MILESTONE:
        return False
    return entry["rank"] is None or entry["rank"] <= NEAR_TARGET_FACTOR * tol


def _window_slow(previous: Optional[float], current: Optional[float], first: bool) -> bool:
    """A window is slow when its stage-one score is not finite, or, after the first window, when it
    improved on the previous window's score by less than FINE_MIN_PROGRESS (no improvement can be
    measured from a nonfinite score)."""
    if current is None:
        return True
    if first:
        return False
    return previous is None or previous <= 0 or (previous - current) / previous < FINE_MIN_PROGRESS


def _autotune_entries() -> int:
    """Cached winners of the dense symmetric step, the one autotuned kernel a fine update can launch
    (a change marks a cold step)."""
    from ..kernels.sinkhorn_flashstyle_sqeuclid import _flashsinkhorn_symmetric_step_kernel_autotune as tuner
    return len(tuner.cache)


@torch.no_grad()
def _transform_zero_weights(geo: dict, a: torch.Tensor, b: torch.Tensor, pair, eps: float):
    """The pair with every zero-weight point's potential replaced by its dense c-transform against the other
    side: the fine updates hold such points (they carry no mass), so their values are set once, here."""
    f, g = pair
    out = []
    for pot, w, src, tgt, other, tgt_shift, lw, src_shift, src_inv, tgt_perm in (
            (f, a, geo["x_s"], geo["y_s"], g, geo["beta"], geo["log_b"], geo["alpha"], geo["inv_perm_x"], geo["perm_y"]),
            (g, b, geo["y_s"], geo["x_s"], f, geo["alpha"], geo["log_a"], geo["beta"], geo["inv_perm_y"], geo["perm_x"])):
        zero = torch.nonzero(w == 0).flatten()
        if zero.numel() == 0:
            out.append(pot)
            continue
        other_hat = F.pad(other.float()[tgt_perm], (0, len(tgt) - len(other))) - tgt_shift
        pot = pot.clone()
        for first in range(0, len(zero), REPAIR_QUERY_CHUNK):
            chunk = zero[first:first + REPAIR_QUERY_CHUNK]
            rows = src_inv[chunk]                          # sorted positions of these zero-weight points
            pot[chunk] = dense_lse_rows(src[rows].contiguous(), tgt, other_hat, lw, eps, 1.0) + src_shift[rows]
        out.append(pot)
    return out[0], out[1]


def _validate(x, y, a, b, tol: float):
    for name, t in (("x", x), ("y", y), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 3 or y.shape[1] != 3 or not len(x) or not len(y):
        raise ValueError("the multiscale solver needs nonempty three-dimensional point clouds")
    if max(len(x), len(y)) > MAX_POINTS:
        raise ValueError(f"the multiscale solver takes at most {MAX_POINTS:,} points per side (int32 offsets of the "
                         f"padded coordinates); got {len(x):,} and {len(y):,}")
    if y.device != x.device:
        raise ValueError("x and y must be on the same device")
    if a.shape != (len(x),) or b.shape != (len(y),) or a.device != x.device or b.device != x.device:
        raise ValueError("a and b must be 1-D weights matching x and y, on the same device")
    for name, t in (("x", x), ("y", y), ("a", a), ("b", b)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} must be finite")
    limit = min(WEIGHT_SUM_TOLERANCE, tol / MASK_TOL_DIVISOR)
    for name, w in (("a", a), ("b", b)):
        total = float(w.double().sum())
        if bool((w < 0).any()) or abs(total - 1.0) > limit:
            raise ValueError(f"{name} must be nonnegative and sum to one within {limit:.3g} (got {total:.9g})")
    # The kernels round every dot-product input to TF32 while the norms stay fp32; only on
    # TF32 values do both describe the same cloud.
    low_bits = (1 << TF32_DROPPED_BITS) - 1
    for name, p in (("x", x), ("y", y)):
        if bool((p.float().view(torch.int32) & low_bits).any()):
            raise ValueError(f"{name} is not TF32-rounded; pass both clouds through "
                             "flash_sinkhorn.multiscale.preprocess_coordinates first")


def _prepared(x, y, a, b, eps: float, tol: float):
    """The validated clouds and weights in fp32."""
    if not (math.isfinite(eps) and eps > 0 and math.isfinite(tol) and 0 < tol <= 1):
        raise ValueError("eps must be positive and tol in (0, 1]")
    _validate(x, y, a, b, tol)
    x, y, a, b = x.float().contiguous(), y.float().contiguous(), a.float().contiguous(), b.float().contiguous()
    for name, t in (("x", x), ("y", y), ("a", a), ("b", b)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} is not finite in fp32")
    return x, y, a, b


def _levels(x, y, a, b, eps: float, l2: int):
    """Joint sort, block summaries, block size, prefix counts, cell level and envelope."""
    joint = joint_sort(x, y, a, b)
    reach = joint.extent ** 2  # bounds every cost, and every squared norm of the centered clouds
    if not max(eps, reach, reach / eps, 1.0 / eps) < FP32_MAX:
        raise ValueError(f"eps={eps:g} and a cloud of extent {joint.extent:g} overflow fp32 potentials")
    sx, sy = block_summaries(joint.source), block_summaries(joint.target)
    B = choose_block_size(joint, sx, sy, l2)
    counts_x, counts_y = prefix_counts(joint.source), prefix_counts(joint.target)
    t = choose_cell_level(counts_x, counts_y, len(x), len(y), B, l2, joint.sides, eps)
    return joint, sx, sy, B, t, envelope(joint, counts_x, counts_y, B, t, l2, eps)


@torch.no_grad()
def check_limits(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float) -> Envelope:
    """The problem against the solver's warning thresholds, from its geometry alone (a sort, no solve): the same
    fp32 guards, cell level and numbers :func:`solve_multiscale` uses. The weights are checked as for tol=1, not
    against a later solve's tighter tolerance or its per-block omission budget."""
    x, y, a, b = _prepared(x, y, a, b, eps, 1.0)
    return _levels(x, y, a, b, eps, int(torch.cuda.get_device_properties(x.device).L2_cache_size))[-1]


@torch.no_grad()
def solve_multiscale(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     eps: float, tol: float) -> Tuple[torch.Tensor, torch.Tensor, MultiscaleInfo]:
    """Potentials ``(f, g)`` in input order for ``C = ||x - y||^2`` and probability weights ``a, b``."""
    x, y, a, b = _prepared(x, y, a, b, eps, tol)
    device = x.device
    n, m = len(x), len(y)

    # Geometry, block size, cell level, schedule.
    l2 = int(torch.cuda.get_device_properties(device).L2_cache_size)
    joint, sx, sy, B, t, env = _levels(x, y, a, b, eps, l2)
    if not env.inside:
        warnings.warn("the multiscale solver is above its measured warning thresholds for this problem, and "
                      f"solves it anyway: {'; '.join(env.reasons)}", RuntimeWarning, stacklevel=2)
    prefix, tail = annealing_schedule(joint.extent, eps)
    cells_x, cells_y = build_cells(joint.source, t, round_to_tf32), build_cells(joint.target, t, round_to_tf32)
    geo = execution_geometry(joint, B, sx, sy)
    delta_rel = mask_delta_rel(geo, tol)
    if not delta_rel > 0:
        raise ValueError(f"tol={tol:g} is too small for {max(geo['N_B'], geo['M_B']):,} blocks: the per-block "
                         "omission budget underflows")

    check = C.SampledCheck(x, y, a, b, eps, tol)
    records: List[dict] = []
    tiles: List[Tuple[int, int, int]] = []
    best: Optional[tuple] = None  # (rank, pair, kind, step)
    counts = dict(continuation=0, fine=0, previews=0)
    meter = WorkMeter(WORK_PER_POINT * (n + m) / 2)  # refinements, updates and repairs, retries and previews included

    def evaluate(kind: str, step: int, pair_fn: Callable):
        nonlocal best

        def run():
            pair = pair_fn()
            return pair, check.evaluate(*pair)

        seconds, (pair, entry) = _timed(device, run)
        entry = dict(entry, kind=kind, step=step, seconds=seconds)
        records.append(entry)
        if entry["rank"] is not None and (best is None or entry["rank"] < best[0]):
            best = (entry["rank"], pair, kind, step)
        return pair, entry

    def finish(pair, *, accepted: bool, kind: str, step: int, stop: str = "accepted", detail: str = ""):
        pair = _transform_zero_weights(geo, a, b, pair, eps)  # a lift's too: it transformed them against the cells
        if not accepted:                                     # the numbers that place the problem (API.md, Limits)
            detail = "; ".join([detail] + (env.reasons or [f"below the warning thresholds (coarse cells "
                                                            f"{env.cell_blurs:.3g} blurs wide, ulp(|x|^2)/eps "
                                                            f"{env.ulp_over_eps:.2g})"]))
        info = MultiscaleInfo(accepted=accepted, stop=stop, detail=detail, fine_work=meter.spent,
                              envelope=env, B=B, t=t,
                              coarse_cells=(len(cells_x.mass), len(cells_y.mass)),
                              schedule_entries=len(prefix) + len(tail),
                              continuation_updates=counts["continuation"], fine_updates=counts["fine"],
                              terminal_previews=counts["previews"], returned=kind, returned_step=step,
                              checks=records, tiles=tiles)
        return pair[0], pair[1], info

    # Anneal up to the first entry at eps; a lift that passes returns immediately.
    state = coarse_updates(coarse_state(cells_x, cells_y), prefix[:1], alpha=1.0)
    state = coarse_updates(state, prefix)
    pair, entry = evaluate("lift", 0, lambda: lift(geo, state, eps))
    if entry["accepted"]:
        return finish(pair, accepted=True, kind="lift", step=0)
    records.clear()  # the early check is discarded: it neither ranks nor schedules
    best = None

    # The remaining entries at eps, then lift and check again.
    state = coarse_updates(state, tail)
    tail_pair, entry = evaluate("lift", 0, lambda: lift(geo, state, eps))
    if entry["accepted"]:
        return finish(tail_pair, accepted=True, kind="lift", step=0)

    # Continue at eps until the coarse updates stall, then lift and check.
    continued, committed, extra = continue_coarse(state, eps, tol)
    fine_pair = tail_pair
    if committed:
        counts["continuation"] = extra
        fine_pair, entry = evaluate("lift", 0, lambda: lift(geo, continued, eps))
        if entry["accepted"]:
            return finish(fine_pair, accepted=True, kind="lift", step=0)
    last = entry

    # Block-sparse fine updates from the last lift.
    fs: FineState = warm_state(geo, *fine_pair)

    nonfinite = 0  # fine-stage checks in a row without a finite score

    def fine_check(kind: str, step: int, pair_fn):
        nonlocal nonfinite
        pair, entry = evaluate(kind, step, pair_fn)
        nonfinite = nonfinite + 1 if entry["rank"] is None else 0
        return pair, entry

    def preview(step: int):
        terminal, _ = fine_step(geo, fs, eps, delta_rel, alpha=1.0, meter=meter)
        counts["previews"] += 1
        return fine_check("terminal", step, lambda: physical_pair(geo, terminal))

    def advance(state: FineState, count: int) -> FineState:
        for _ in range(count):
            state, sizes = fine_step(geo, state, eps, delta_rel, alpha=0.5, meter=meter)
            counts["fine"] = state.ordinary_steps
            if sizes is not None:
                tiles.append((state.ordinary_steps, *sizes))
        return state

    stop, detail = "ceiling", f"{FINE_CEILING} fine updates without confirmation"
    found = None  # (pair, kind, step) of the accepted candidate, returned outside the handlers below
    try:
        if committed:
            pair, entry = preview(0)
            if entry["accepted"]:
                found = (pair, "terminal", 0)
        h, step_seconds, cold = 0, None, False
        window_score, slow = None, 0
        while found is None and h < FINE_CEILING:
            target = _next_check(h, FINE_CEILING, _interval(last["seconds"], step_seconds, last, tol, cold))
            before = _autotune_entries()
            seconds, fs = _timed(device, lambda: advance(fs, target - h))
            step_seconds, cold = seconds / (target - h), _autotune_entries() != before
            h = target
            pair, last = fine_check("ordinary", h, lambda: physical_pair(geo, fs))
            if last["accepted"]:
                found = (pair, "ordinary", h)
                break
            if nonfinite >= UNSTABLE_NONFINITE_CHECKS:
                stop, detail = "unstable", f"{UNSTABLE_NONFINITE_CHECKS} checks in a row gave no finite score"
                break
            stalled = False
            if h % FINE_WINDOW == 0:
                slow = slow + 1 if _window_slow(window_score, last["rank"], h == FINE_WINDOW) else 0
                window_score = last["rank"]
                stalled = slow >= FINE_SLOW_WINDOWS
            if _preview_due(h, FINE_CEILING, last, tol) or stalled:
                pair, entry = preview(h)
                if entry["accepted"]:
                    found = (pair, "terminal", h)
                    break
                if nonfinite >= UNSTABLE_NONFINITE_CHECKS:
                    stop, detail = "unstable", f"{UNSTABLE_NONFINITE_CHECKS} checks in a row gave no finite score"
                    break
            if stalled:
                stop = "stalled"
                detail = (f"the sampled score improved by less than {FINE_MIN_PROGRESS:.0%} over "
                          f"{FINE_SLOW_WINDOWS} windows of {FINE_WINDOW} updates")
                break
    except UnstableMask as ex:
        stop, detail = "unstable", str(ex)
    except MaskTooLarge as ex:
        stop, detail = "too large", str(ex)
    except WorkBudget as ex:
        stop, detail = "budget", f"{ex} (budget {WORK_PER_POINT:.0e} pairs per point)"
    except torch.cuda.OutOfMemoryError as ex:           # a check or a preview's workspace, not the mask
        stop, detail = "too large", "out of device memory in the fine stage: " + str(ex).split("\n")[0][:200]

    if found is not None:
        return finish(found[0], accepted=True, kind=found[1], step=found[2])
    # The fp32 check reads high where ulp(|x|^2) approaches tol * eps: before stopping unconfirmed, the best candidate
    # is checked once more on the same samples in fp64, and accepted if it passes there.
    if stop in ("stalled", "ceiling", "budget") and best is not None:
        try:
            seconds, entry = _timed(device, lambda: check.evaluate_fp64(*best[1]))
        except torch.cuda.OutOfMemoryError:
            detail += "; the fp64 re-check did not fit in device memory"
        else:
            records.append(dict(entry, kind="fp64", step=best[3], seconds=seconds))
            if entry["accepted"]:
                return finish(best[1], accepted=True, kind=best[2], step=best[3],
                              detail=f"confirmed by the fp64 check after the fine stage stopped as {stop}")
    if best is not None:
        return finish(best[1], accepted=False, kind=best[2], step=best[3], stop=stop, detail=detail)
    return finish(tail_pair, accepted=False, kind="lift", step=0, stop=stop, detail=detail)
