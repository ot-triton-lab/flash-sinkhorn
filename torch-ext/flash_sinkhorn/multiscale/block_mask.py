"""Block-pair mask of the fine stage, built by a box screen into CSR form.

The clouds are sorted and cut into blocks of ``B`` consecutive points. A block
pair ``(I, J)`` is kept when the axis-aligned boxes of the two blocks do not rule
out a non-negligible plan entry:

    cost_scale * d_min^2(I, J) < F_I + G_J + eps * (-log delta_rel + log max(a_I, b_J)),

with ``F_I = max_{i in I} (f_hat_i + alpha_i)`` and ``G_J`` likewise for the
targets (the potentials in the convention ``P_ij = a_i b_j exp((f_i + g_j -
C_ij)/eps)``), ``a_I`` and ``b_J`` the block masses, and ``d_min^2`` the squared
gap between the two boxes. Omitting a pair bounds its mass in every row of ``I``
by ``delta_rel * a_i`` and in every column of ``J`` by ``delta_rel * b_j``, so one
mask and its transpose serve both half-steps.

The predicate runs in registers on the GPU in two passes, one counting and one
writing, and only the CSR arrays reach memory. Columns come out ascending within
each row. The target blocks come in groups of ``BLOCK_J``; a first, flat screen
tests each group as one box with the group's largest potential and block mass,
and the block tests run only in the groups it keeps. Both screens evaluate the same
predicate code, and every quantity of a group bounds its blocks', so by the
monotonicity of rounding a group that fails holds no block that passes: the kept
pairs are exactly those of the pair-by-pair test.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

#: ``kernels._common.log_weights`` maps a zero weight to this finite value.
LOG_ZERO_SENTINEL = -100000.0
#: int32 CSR capacity for row pointers, column indices and block counts.
CSR_INDEX_MAX = 2 ** 31 - 1
#: A CSR is built only if its estimated size stays within this share of the free device memory.
CSR_MEMORY_FRACTION = 0.8
CSR_BYTES_PER_ENTRY = 24
CSR_BYTES_PER_ROW = 8
#: Target blocks tested per program iteration.
BLOCK_J = 256


class CSRCapacityError(RuntimeError):
    """A CSR would exceed the int32 indices."""


def csr_preflight(nnz: int, N_B: int, M_B: int) -> None:
    """Refuse a CSR whose dimensions or entry count exceed int32 before allocating it."""
    if N_B > CSR_INDEX_MAX or M_B > CSR_INDEX_MAX:
        raise CSRCapacityError(
            f"block-sparse CSR dimensions exceed int32 index capacity: "
            f"N_B={N_B:,}, M_B={M_B:,}, capacity={CSR_INDEX_MAX:,}")
    if nnz > CSR_INDEX_MAX:
        raise CSRCapacityError(
            f"block-sparse CSR index overflow: nnz={nnz:,} exceeds the int32 "
            f"capacity {CSR_INDEX_MAX:,} (N_B={N_B:,}, M_B={M_B:,}, "
            f"density={nnz / (N_B * M_B):.4%})")


def csr_memory_admission(nnz: int, N_B: int, M_B: int, device: torch.device,
                         bytes_per_entry: int = CSR_BYTES_PER_ENTRY) -> None:
    """Refuse a CSR whose estimated size exceeds the free (and reusable cached) device memory share."""
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    estimate = bytes_per_entry * nnz + CSR_BYTES_PER_ROW * (N_B + M_B + 2)
    if estimate > CSR_MEMORY_FRACTION * available:
        raise torch.cuda.OutOfMemoryError(
            f"block-sparse mask needs about {estimate / 2**30:.2f} GiB for {nnz:,} tiles, more than "
            f"{CSR_MEMORY_FRACTION:.0%} of the {available / 2**30:.2f} GiB available")


@triton.jit
def _row_keep(
    x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound,
    i, j0, M_B, cost_scale, eps, neg_log_drel,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK: tl.constexpr,
):
    """Keep flags of block row ``i`` against target blocks ``[j0, j0 + BLOCK)``."""
    offs_d = tl.arange(0, D_PAD)
    md = offs_d < D
    xlo = tl.load(x_lo + i * D + offs_d, mask=md, other=0.0)
    xhi = tl.load(x_hi + i * D + offs_d, mask=md, other=0.0)

    offs_j = j0 + tl.arange(0, BLOCK)
    mj = offs_j < M_B
    ptr = offs_j[:, None] * D + offs_d[None, :]
    m2 = mj[:, None] & md[None, :]
    ylo = tl.load(y_lo + ptr, mask=m2, other=0.0)
    yhi = tl.load(y_hi + ptr, mask=m2, other=0.0)

    gap = tl.maximum(tl.maximum(xlo[None, :] - yhi, ylo - xhi[None, :]), 0.0)
    gap = tl.where(m2, gap, 0.0)
    lhs = cost_scale * tl.sum(gap * gap, axis=1)

    gb = tl.load(g_bound + offs_j, mask=mj, other=float("-inf"))
    la = tl.load(la_bound + i)
    lb = tl.load(lb_bound + offs_j, mask=mj, other=float("-inf"))
    rhs = tl.load(f_bound + i) + gb + eps * (neg_log_drel + tl.maximum(la, lb))
    return (lhs < rhs) & mj, offs_j


@triton.jit
def _count_flat_kernel(
    x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound, counts,
    M_B, cost_scale, eps, neg_log_drel,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    cnt = tl.zeros((), dtype=tl.int32)
    for j0 in range(0, M_B, BLOCK):
        keep, _ = _row_keep(
            x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound,
            i, j0, M_B, cost_scale, eps, neg_log_drel, D, D_PAD, BLOCK)
        cnt += tl.sum(keep.to(tl.int32), axis=0)
    tl.store(counts + i, cnt)


@triton.jit
def _fill_flat_kernel(
    x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound, crow, col,
    M_B, cost_scale, eps, neg_log_drel,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    base = tl.load(crow + i)
    written = tl.zeros((), dtype=tl.int32)
    for j0 in range(0, M_B, BLOCK):
        keep, offs_j = _row_keep(
            x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound,
            i, j0, M_B, cost_scale, eps, neg_log_drel, D, D_PAD, BLOCK)
        ki = keep.to(tl.int32)
        # Exclusive prefix sum: each kept column's slot inside this tile.
        pos = tl.cumsum(ki, axis=0) - ki
        tl.store(col + base + written + pos, offs_j.to(tl.int32), mask=keep)
        written += tl.sum(ki, axis=0)


@triton.jit
def _count_kernel(
    x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound, counts, gcrow, gcol,
    M_B, cost_scale, eps, neg_log_drel,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK: tl.constexpr,
):
    """Block-level count over the target groups the flat group screen kept for this row."""
    i = tl.program_id(0)
    cnt = tl.zeros((), dtype=tl.int32)
    g0 = tl.load(gcrow + i)
    for gg in range(tl.load(gcrow + i + 1) - g0):
        q = tl.load(gcol + g0 + gg)
        keep, _ = _row_keep(
            x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound,
            i, q * BLOCK, M_B, cost_scale, eps, neg_log_drel, D, D_PAD, BLOCK)
        cnt += tl.sum(keep.to(tl.int32), axis=0)
    tl.store(counts + i, cnt)


@triton.jit
def _fill_kernel(
    x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound, crow, col, gcrow, gcol,
    M_B, cost_scale, eps, neg_log_drel,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK: tl.constexpr,
):
    """Block-level fill over the kept target groups; groups ascend, so columns ascend."""
    i = tl.program_id(0)
    base = tl.load(crow + i)
    written = tl.zeros((), dtype=tl.int32)
    g0 = tl.load(gcrow + i)
    for gg in range(tl.load(gcrow + i + 1) - g0):
        q = tl.load(gcol + g0 + gg)
        keep, offs_j = _row_keep(
            x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound,
            i, q * BLOCK, M_B, cost_scale, eps, neg_log_drel, D, D_PAD, BLOCK)
        ki = keep.to(tl.int32)
        pos = tl.cumsum(ki, axis=0) - ki
        tl.store(col + base + written + pos, offs_j.to(tl.int32), mask=keep)
        written += tl.sum(ki, axis=0)


def block_log_mass(log_w: Tensor, n_blocks: int, B: int) -> Tensor:
    """``log sum_{i in I} w_i`` per block; the sentinel contributes ``exp(-1e5) = 0``."""
    return torch.logsumexp(log_w.view(n_blocks, B).float(), dim=1).contiguous()


def _block_max(v: Tensor, zero: Tensor, n_blocks: int, B: int) -> Tensor:
    """Per-block maximum with zero-weight slots excluded."""
    return v.masked_fill(zero, -torch.inf).view(n_blocks, B).max(dim=1).values


def _target_groups(y_lo: Tensor, y_hi: Tensor, g_bound: Tensor, lb_bound: Tensor) -> Tuple[Tensor, ...]:
    """Box, largest potential bound and largest block log-mass of each group of ``BLOCK_J`` target blocks."""
    M_B, d = y_lo.shape
    pad = (-M_B) % BLOCK_J
    lo = F.pad(y_lo, (0, 0, 0, pad), value=float("inf")).view(-1, BLOCK_J, d).amin(1)
    hi = F.pad(y_hi, (0, 0, 0, pad), value=-float("inf")).view(-1, BLOCK_J, d).amax(1)
    g = F.pad(g_bound, (0, pad), value=-float("inf")).view(-1, BLOCK_J).amax(1)
    lb = F.pad(lb_bound, (0, pad), value=-float("inf")).view(-1, BLOCK_J).amax(1)
    return lo.contiguous(), hi.contiguous(), g.contiguous(), lb.contiguous()


def prepare_screen(
    x_lo: Tensor, x_hi: Tensor, y_lo: Tensor, y_hi: Tensor,
    f_hat: Tensor, g_hat: Tensor, alpha: Tensor, beta: Tensor,
    log_a: Tensor, log_b: Tensor, eps: float, *,
    B: int, delta_rel: float, cost_scale: float = 1.0,
) -> dict:
    """Block bounds, target groups and launch arguments of one box screen (see :func:`build_block_mask_csr`)."""
    if x_lo.device.type != "cuda":
        raise ValueError("the mask builder runs on CUDA tensors")
    N_B, d = x_lo.shape
    M_B = y_lo.shape[0]
    csr_preflight(0, N_B, M_B)
    zero_f = log_a <= LOG_ZERO_SENTINEL
    zero_g = log_b <= LOG_ZERO_SENTINEL
    f_bound = _block_max(f_hat + alpha, zero_f, N_B, B)
    g_bound = _block_max(g_hat + beta, zero_g, M_B, B)
    la_bound = block_log_mass(log_a, N_B, B)
    lb_bound = block_log_mass(log_b, M_B, B)
    rows = (x_lo.contiguous(), x_hi.contiguous(), f_bound.contiguous(), la_bound.contiguous())
    cols = (y_lo.contiguous(), y_hi.contiguous(), g_bound.contiguous(), lb_bound.contiguous())
    groups = _target_groups(y_lo, y_hi, g_bound, lb_bound)
    kw = dict(D=d, D_PAD=max(4, triton.next_power_of_2(d)), BLOCK=BLOCK_J)
    tail = (cost_scale, eps, -math.log(delta_rel))
    # The flat screen over groups: the same predicate, with each group taken as one block.
    flat = (rows[0], rows[1], groups[0], groups[1], rows[2], groups[2], rows[3], groups[3])
    G = groups[0].shape[0]
    gcounts = torch.empty(N_B, dtype=torch.int32, device=x_lo.device)
    _count_flat_kernel[(N_B,)](*flat, gcounts, G, *tail, **kw)
    g_nnz = int(gcounts.to(torch.int64).sum())
    csr_preflight(g_nnz, N_B, G)
    csr_memory_admission(g_nnz, N_B, G, x_lo.device, bytes_per_entry=4)  # the group columns only
    gcrow = torch.zeros(N_B + 1, dtype=torch.int32, device=x_lo.device)
    gcrow[1:] = gcounts.cumsum(0)
    gcol = torch.empty(g_nnz, dtype=torch.int32, device=x_lo.device)
    if g_nnz:
        _fill_flat_kernel[(N_B,)](*flat, gcrow, gcol, G, *tail, **kw)
    return dict(rows=rows, cols=cols, gcrow=gcrow, gcol=gcol, scalars=(M_B, *tail), kw=kw,
                N_B=N_B, M_B=M_B, device=x_lo.device)


def _screen_args(screen: dict, r0: int, r1: int):
    x_lo, x_hi, f_bound, la_bound = (t[r0:r1] for t in screen["rows"])
    y_lo, y_hi, g_bound, lb_bound = screen["cols"]
    return (x_lo, x_hi, y_lo, y_hi, f_bound, g_bound, la_bound, lb_bound)


def screen_counts(screen: dict) -> Tensor:
    """int32 number of candidate target blocks of every block row."""
    counts = torch.empty(screen["N_B"], dtype=torch.int32, device=screen["device"])
    _count_kernel[(screen["N_B"],)](*_screen_args(screen, 0, screen["N_B"]), counts, screen["gcrow"],
                                    screen["gcol"], *screen["scalars"], **screen["kw"])
    return counts


def screen_fill(screen: dict, counts: Tensor, r0: int, r1: int) -> Tuple[Tensor, Tensor]:
    """CSR ``(crow, col)`` of the candidates of block rows ``[r0, r1)`` (rows renumbered from 0)."""
    crow = torch.zeros(r1 - r0 + 1, dtype=torch.int32, device=screen["device"])
    crow[1:] = counts[r0:r1].cumsum(0)
    col = torch.empty(int(crow[-1]), dtype=torch.int32, device=screen["device"])
    if col.numel():
        _fill_kernel[(r1 - r0,)](*_screen_args(screen, r0, r1), crow, col, screen["gcrow"][r0:r1 + 1],
                                 screen["gcol"], *screen["scalars"], **screen["kw"])
    return crow, col


def csr_capacity(N_B: int, M_B: int, device: torch.device) -> int:
    """Largest entry count one CSR may have: int32 indices and the memory admission of :func:`csr_memory_admission`."""
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    by_memory = (CSR_MEMORY_FRACTION * available - CSR_BYTES_PER_ROW * (N_B + M_B + 2)) // CSR_BYTES_PER_ENTRY
    return max(0, min(CSR_INDEX_MAX, int(by_memory)))


def build_block_mask_csr(
    x_lo: Tensor, x_hi: Tensor, y_lo: Tensor, y_hi: Tensor,
    f_hat: Tensor, g_hat: Tensor, alpha: Tensor, beta: Tensor,
    log_a: Tensor, log_b: Tensor, eps: float, *,
    B: int, delta_rel: float, cost_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """CSR ``(crow, col)`` of the block pairs that pass the box screen.

    Args:
        x_lo, x_hi: ``[N_B, d]`` source block boxes.
        y_lo, y_hi: ``[M_B, d]`` target block boxes.
        f_hat, g_hat: Shifted potentials ``f - alpha``, ``g - beta`` on the padded clouds.
        alpha, beta: Shifts ``cost_scale * ||x_i||^2`` and ``cost_scale * ||y_j||^2``.
        log_a, log_b: Log weights, ``LOG_ZERO_SENTINEL`` on zero-weight padding.
        eps: Regularization of the update the mask serves.
        B: Block size of the boxes.
        delta_rel: Per-block omitted-mass budget.
        cost_scale: 1.0 for ``||x - y||^2``, 0.5 for ``||x - y||^2 / 2``.

    Returns:
        int32 ``crow [N_B + 1]`` and ``col [nnz]``.
    """
    screen = prepare_screen(x_lo, x_hi, y_lo, y_hi, f_hat, g_hat, alpha, beta, log_a, log_b, eps,
                            B=B, delta_rel=delta_rel, cost_scale=cost_scale)
    counts = screen_counts(screen)
    # The total is taken in int64 before it is narrowed into the int32 row pointers.
    nnz = int(counts.to(torch.int64).sum())
    csr_preflight(nnz, screen["N_B"], screen["M_B"])
    csr_memory_admission(nnz, screen["N_B"], screen["M_B"], screen["device"])
    return screen_fill(screen, counts, 0, screen["N_B"])


def csr_transpose(crow: Tensor, col: Tensor, N_B: int, M_B: int) -> Tuple[Tensor, Tensor]:
    """CSR of the transposed block-pair matrix; row indices stay ascending within each column."""
    device = crow.device
    nnz = col.shape[0]
    csr_preflight(nnz, M_B, N_B)
    csr_memory_admission(nnz, M_B, N_B, device)
    if nnz == 0:
        return (torch.zeros(M_B + 1, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device))

    col_counts = torch.zeros(M_B, dtype=torch.int32, device=device)
    col_counts.scatter_add_(0, col.long(), torch.ones(nnz, dtype=torch.int32, device=device))
    crow_t = torch.zeros(M_B + 1, dtype=torch.int32, device=device)
    crow_t[1:] = col_counts.cumsum(dim=0)

    row_ids = torch.repeat_interleave(
        torch.arange(N_B, device=device, dtype=torch.int32), (crow[1:] - crow[:-1]).long())
    # A stable sort by column keeps each column's rows in ascending order.
    order = torch.argsort(col.long(), stable=True)
    sorted_cols = col[order]
    sorted_rows = row_ids[order]
    positions = (crow_t[sorted_cols.long()]
                 + (torch.arange(nnz, device=device, dtype=torch.int32)
                    - _group_start_indices(sorted_cols, device)))
    col_t = torch.empty(nnz, dtype=torch.int32, device=device)
    col_t[positions.long()] = sorted_rows
    return crow_t, col_t


def _group_start_indices(sorted_keys: Tensor, device: torch.device) -> Tensor:
    """For each element of a sorted array, the index where its run of equal keys starts."""
    n = sorted_keys.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    changes = torch.ones(n, dtype=torch.bool, device=device)
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    index = torch.arange(n, dtype=torch.int32, device=device)
    return torch.where(changes, index, torch.zeros_like(index)).cummax(dim=0).values
