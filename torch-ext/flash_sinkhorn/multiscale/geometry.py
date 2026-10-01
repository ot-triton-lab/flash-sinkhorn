"""Morton geometry, block summaries, and the automatic block-size and cell-level rules.

Both clouds are sorted once along a shared 63-bit Morton curve. Blocks of ``B``
consecutive sorted points feed the fine stage; cells of points that share all
but the lowest ``t`` key bits feed the coarse stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from ..kernels._common import epsilon_schedule

MORTON_BITS = 21
MAX_QUANTIZED = (1 << MORTON_BITS) - 1
KEY_BITS = 3 * MORTON_BITS
BLOCK_SIZES = (64, 128, 256)
PAD_K = 16

# Capacity budgets (bytes are estimates for the L2 cache, not residency guarantees).
L2_FRACTION = 0.8
SUMMARY_BYTES_PER_BLOCK = 4 * (2 * 3 + 2)  # box lo/hi, potential maximum, log mass
COARSE_BYTES_PER_CELL = 112
CELLS_PER_BLOCK = 2  # at most twice as many cells as blocks on each side ...
CELL_FLOOR = 8192    # ... but at least 8192, so that small clouds are annealed finely enough for the fine stage
                     # to converge quickly (the limit above reaches 8192 at n = 2^19 with B = 128)
INT32_OFFSET_LIMIT = 2**31  # offsets up to 16 * rows - 1 must fit a signed 32-bit integer
CELL_EDGE_BLURS = 9.0  # the coarse cells may be at most 9 blurs wide, even beyond the L2 budget: wider
                       # cells can lift potentials that leave rows so short of mass that the fine masks empty them
CELL_L2_MULTIPLE = 2   # ... but never more than twice the cells the L2 budget holds: the dense coarse updates
                       # cost the square of the cell count

# Geometric block-doubling test (64 -> 128).
MERGE_INFLATION_Q90 = 1.5
MERGE_INFLATION_TAIL = 2.0
MERGE_INFLATION_TAIL_FRACTION = 0.01

# Coarse annealing schedule.
ANNEAL_SCALING = 0.95
TARGET_TAIL = 64

ZERO_WEIGHT_LOG = -1e5


@dataclass
class SortedCloud:
    points: torch.Tensor
    weights: torch.Tensor
    keys: torch.Tensor
    perm: torch.Tensor
    inv_perm: torch.Tensor
    n: int


@dataclass
class JointSort:
    source: SortedCloud
    target: SortedCloud
    extent: float                     # diagonal of the joint bounding box
    sides: Tuple[float, float, float]  # its side lengths along x, y, z


@dataclass
class BlockSummary:
    B: int
    counts: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor


@dataclass
class Cells:
    points: torch.Tensor
    mass: torch.Tensor


def pad_columns(x: torch.Tensor) -> torch.Tensor:
    """fp32 coordinates padded with zero columns to ``PAD_K`` (the kernels' tile width)."""
    return F.pad(x.float().contiguous(), (0, PAD_K - x.shape[1]))


def _spread(v: torch.Tensor) -> torch.Tensor:
    v = v & 0x1FFFFF
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF),
                        (8, 0x100F00F00F00F00F), (4, 0x10C30C30C30C30C3),
                        (2, 0x1249249249249249)):
        v = (v | (v << shift)) & mask
    return v


def _morton_keys(points: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    q = (points.float() - lo) / (hi - lo).clamp_min(1e-30) * MAX_QUANTIZED
    q = q.to(torch.int64).clamp_(0, MAX_QUANTIZED)
    return _spread(q[:, 0]) | (_spread(q[:, 1]) << 1) | (_spread(q[:, 2]) << 2)


@torch.no_grad()
def joint_sort(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> JointSort:
    """Sort each cloud along one Morton grid spanning the joint bounding box."""
    lo = torch.minimum(x.float().amin(0), y.float().amin(0))
    hi = torch.maximum(x.float().amax(0), y.float().amax(0))
    clouds = []
    for p, w in ((x, a), (y, b)):
        keys = _morton_keys(p, lo, hi)
        perm = torch.argsort(keys, stable=True)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm), device=perm.device)
        clouds.append(SortedCloud(p.float()[perm].contiguous(), w.float()[perm].contiguous(),
                                  keys[perm], perm, inv, len(p)))
    return JointSort(clouds[0], clouds[1], float((hi - lo).norm()), tuple((hi - lo).tolist()))


@torch.no_grad()
def prefix_counts(cloud: SortedCloud) -> Dict[int, int]:
    """Occupied cells for every dropped-bit count t in [0, 62], from one histogram."""
    diff = cloud.keys[1:] ^ cloud.keys[:-1]
    powers = torch.tensor([1 << k for k in range(KEY_BITS)], device=diff.device, dtype=torch.int64)
    lengths = torch.bucketize(diff, powers, right=True)
    histogram = torch.bincount(lengths, minlength=KEY_BITS + 1)
    greater = (histogram.sum() - histogram.cumsum(0)).cpu().tolist()
    return {t: 1 + int(greater[t]) for t in range(KEY_BITS)}


@torch.no_grad()
def build_cells(cloud: SortedCloud, t: int, round_fn: Callable[[torch.Tensor], torch.Tensor]) -> Cells:
    """Mass-weighted cell centroids (aggregated in FP64, then rounded by ``round_fn``)."""
    keys, inverse = torch.unique_consecutive(cloud.keys >> t, return_inverse=True)
    features = torch.cat((cloud.points.double() * cloud.weights.double()[:, None],
                          cloud.weights.double()[:, None]), dim=1)
    sums = torch.zeros((keys.numel(), 4), dtype=torch.float64, device=cloud.points.device)
    sums.index_add_(0, inverse, features)
    sums = sums[sums[:, 3] > 0]
    centers = round_fn((sums[:, :3] / sums[:, 3, None]).float().contiguous()).contiguous()
    return Cells(centers, sums[:, 3].float().contiguous())


def _base_summary(cloud: SortedCloud, B: int) -> BlockSummary:
    pad = (-cloud.n) % B
    p = cloud.points
    counts = F.pad(torch.ones(cloud.n, device=p.device, dtype=torch.int64), (0, pad)).view(-1, B).sum(1)
    lo = F.pad(p, (0, 0, 0, pad), value=float("inf")).view(-1, B, 3).amin(1)
    hi = F.pad(p, (0, 0, 0, pad), value=-float("inf")).view(-1, B, 3).amax(1)
    return BlockSummary(B, counts, lo, hi)


def _merge(child: BlockSummary) -> BlockSummary:
    pad = child.counts.numel() % 2
    counts = F.pad(child.counts, (0, pad)).view(-1, 2).sum(1)
    lo = F.pad(child.lo, (0, 0, 0, pad), value=float("inf")).view(-1, 2, 3).amin(1)
    hi = F.pad(child.hi, (0, 0, 0, pad), value=-float("inf")).view(-1, 2, 3).amax(1)
    return BlockSummary(2 * child.B, counts, lo, hi)


@torch.no_grad()
def block_summaries(cloud: SortedCloud) -> Dict[int, BlockSummary]:
    """Bounding boxes of blocks of 64 points, merged pairwise to 128 and 256."""
    summaries = {BLOCK_SIZES[0]: _base_summary(cloud, BLOCK_SIZES[0])}
    for B in BLOCK_SIZES[1:]:
        summaries[B] = _merge(summaries[B // 2])
    return summaries


def _doubling_allowed(summaries: Dict[int, BlockSummary], extent: float) -> bool:
    """Merging pairs of 64-blocks into 128-blocks must not inflate their boxes much."""
    floor = max(abs(extent) * 1e-12, 1e-30)
    child, parent = summaries[64], summaries[128]
    child_diag = (child.hi - child.lo).double().norm(dim=1)
    parent_diag = (parent.hi - parent.lo).double().norm(dim=1)
    pairs = child_diag.numel() // 2
    if not pairs:
        return False
    scale = child_diag[:2 * pairs].view(-1, 2).square().mean(1).sqrt()
    diagonal = parent_diag[:pairs]
    ratio = diagonal / (2 ** (1 / 3) * scale.clamp_min(floor))
    ratio = torch.where((scale <= floor) & (diagonal <= floor), torch.ones_like(ratio), ratio)
    q90 = float(torch.quantile(ratio, torch.tensor(0.9, device=ratio.device, dtype=torch.float64)))
    tail = float((ratio > MERGE_INFLATION_TAIL).double().mean())
    return q90 <= MERGE_INFLATION_Q90 and tail <= MERGE_INFLATION_TAIL_FRACTION


def summary_bytes(n: int, m: int, B: int) -> int:
    return SUMMARY_BYTES_PER_BLOCK * (math.ceil(n / B) + math.ceil(m / B))


def choose_block_size(joint: JointSort, sx: Dict[int, BlockSummary], sy: Dict[int, BlockSummary],
                      l2_bytes: int) -> int:
    """64 or 128 by the doubling test, then larger while the block summaries exceed the L2 budget."""
    n, m = joint.source.n, joint.target.n
    B = 128 if _doubling_allowed(sx, joint.extent) and _doubling_allowed(sy, joint.extent) else 64
    while summary_bytes(n, m, B) > L2_FRACTION * l2_bytes:
        if B == BLOCK_SIZES[-1]:
            raise RuntimeError("block summaries exceed the L2 budget even at the largest block size")
        B *= 2
    return B


def _index_supported(kx: int, ky: int, n: int, m: int, B: int) -> bool:
    rows = max(kx, ky, math.ceil(n / B) * B, math.ceil(m / B) * B)
    return 0 < kx <= n and 0 < ky <= m and rows * PAD_K <= INT32_OFFSET_LIMIT


def cell_edge(sides: Tuple[float, float, float], t: int) -> float:
    """Longest side of the Morton cells after dropping t key bits (x holds the lowest bit of each triple)."""
    return max(side * (1 << math.ceil(max(t - axis, 0) / 3)) / MAX_QUANTIZED for axis, side in enumerate(sides))


def _feasible_levels(counts_x: Dict[int, int], counts_y: Dict[int, int], n: int, m: int, B: int):
    limit_x = max(CELLS_PER_BLOCK * math.ceil(n / B), CELL_FLOOR)
    limit_y = max(CELLS_PER_BLOCK * math.ceil(m / B), CELL_FLOOR)
    feasible = [t for t in range(KEY_BITS) if _index_supported(counts_x[t], counts_y[t], n, m, B)
                and counts_x[t] <= limit_x and counts_y[t] <= limit_y]
    if not feasible:
        raise RuntimeError("no cell level satisfies the indexing and cell-count limits")
    return feasible


def narrowest_cell_edge(counts_x: Dict[int, int], counts_y: Dict[int, int], n: int, m: int, B: int,
                        l2_bytes: int, sides: Tuple[float, float, float]) -> float:
    """Longest cell edge of the finest level within CELL_L2_MULTIPLE times the L2 budget: the narrowest cells
    the coarse stage can use on this GPU."""
    cap = CELL_L2_MULTIPLE * L2_FRACTION * l2_bytes
    levels = [t for t in _feasible_levels(counts_x, counts_y, n, m, B)
              if COARSE_BYTES_PER_CELL * (counts_x[t] + counts_y[t]) <= cap]
    return cell_edge(sides, levels[0]) if levels else math.inf


def choose_cell_level(counts_x: Dict[int, int], counts_y: Dict[int, int], n: int, m: int, B: int,
                      l2_bytes: int, sides: Tuple[float, float, float], eps: float) -> int:
    """Finest t whose cells fit the L2 budget, unless its cells are more than CELL_EDGE_BLURS blurs wide:
    then the coarsest t whose cells are narrow enough, within CELL_L2_MULTIPLE times the L2 budget, and if
    none is, the finest t within the L2 budget again (the dense coarse updates cost the square of the cell
    count). A cloud with no level within the L2 budget is refused. Every choice respects the int32 indexing
    and the per-side cell limit ``max(CELLS_PER_BLOCK * ceil(n / B), CELL_FLOOR)``; small clouds are annealed
    on the points themselves."""
    feasible = _feasible_levels(counts_x, counts_y, n, m, B)
    widest = CELL_EDGE_BLURS * math.sqrt(eps)
    size = lambda t: COARSE_BYTES_PER_CELL * (counts_x[t] + counts_y[t])
    within_l2 = [t for t in feasible if size(t) <= L2_FRACTION * l2_bytes]
    if not within_l2:
        raise RuntimeError(f"no cell level fits the L2 budget ({L2_FRACTION:.0%} of {l2_bytes:,} bytes at "
                           f"{COARSE_BYTES_PER_CELL} bytes per cell)")
    if cell_edge(sides, within_l2[0]) <= widest:
        return within_l2[0]
    narrow = [t for t in feasible if size(t) <= CELL_L2_MULTIPLE * L2_FRACTION * l2_bytes
              and cell_edge(sides, t) <= widest]
    return narrow[-1] if narrow else within_l2[0]


def annealing_schedule(extent: float, eps: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """``(prefix, tail)``: GeomLoss epsilon-scaling from the cloud diameter down to eps, whose last
    entry is the first one equal to eps, then ``TARGET_TAIL`` further entries at eps.

    A cloud no wider than the blur has nothing to anneal, and its prefix is the single entry eps.
    """
    blur = math.sqrt(eps)
    if extent > blur:
        prefix = [float(e) for e in epsilon_schedule(extent, blur, ANNEAL_SCALING, p=2.0)]
        prefix[-1] = eps
        if eps in prefix[:-1]:
            raise ValueError("the annealing schedule reaches eps before its last entry")
    else:
        prefix = [eps]
    return tuple(prefix), (eps,) * TARGET_TAIL


@torch.no_grad()
def execution_geometry(joint: JointSort, B: int, sx: Dict[int, BlockSummary],
                       sy: Dict[int, BlockSummary]) -> dict:
    """Solver-ready sorted arrays padded to whole blocks and to 16 coordinate columns."""
    geo = dict(B=B, L=joint.extent, n=joint.source.n, m=joint.target.n)
    for side, cloud, sums, w_key, shift_key, blocks_key in (
            ("x", joint.source, sx, "a", "alpha", "N_B"), ("y", joint.target, sy, "b", "beta", "M_B")):
        pad = (-cloud.n) % B
        points = F.pad(cloud.points, (0, PAD_K - 3, 0, pad)).contiguous()
        weights = F.pad(cloud.weights, (0, pad)).contiguous()
        geo[f"{side}_s"] = points
        geo[f"{side}_lo"], geo[f"{side}_hi"] = sums[B].lo, sums[B].hi
        geo[f"perm_{side}"], geo[f"inv_perm_{side}"] = cloud.perm, cloud.inv_perm
        geo[f"{w_key}_s"] = weights
        geo[f"log_{w_key}"] = torch.where(weights > 0, weights.log(),
                                         torch.full_like(weights, ZERO_WEIGHT_LOG))
        geo[shift_key] = points.square().sum(1)
        geo[blocks_key] = len(sums[B].counts)
    return geo
