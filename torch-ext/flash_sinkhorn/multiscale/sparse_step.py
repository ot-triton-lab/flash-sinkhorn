"""Symmetric block-sparse fine update, with dense repair of empty block rows.

Both half-steps start from the same incoming pair ``(f_hat, g_hat)``: the source
update reads the mask, the target update reads its transpose, and each result is
averaged with its input. A block row with no admitted tile would have an empty
sum; those rows alone are recomputed against every point of the other cloud, by
a streaming kernel that splits the target dimension. No ``n x m`` matrix is formed.
A block row of zero weight carries no mass: it keeps its incoming potential instead
(its points get their dense c-transform once, for the returned pair).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..kernels._triton_helpers import _final_lse, _tiled_dot
from ..kernels.sinkhorn_blocksparse_sqeuclid import blocksparse_lse_fused
from .block_mask import LOG_ZERO_SENTINEL, block_log_mass

#: Tile of the dense repair kernel (rows, columns, dot K-slice).
REPAIR_TILE = 64
REPAIR_BLOCK_K = 16
#: Rows per program in the repair merge kernel.
REPAIR_MERGE_BLOCK = 256
#: Targets per repair split, and the maximum number of splits.
REPAIR_SPLIT_TARGETS = 4096
REPAIR_MAX_SPLITS = 128
#: Query points repaired per launch.
REPAIR_QUERY_CHUNK = 4096


@triton.jit
def _repair_partial_kernel(X, Y, G, LW, PM, PS, N, M, SPLIT_SIZE,
                           SX0, SX1, SY0, SY1, COORD_SCALE, EPS,
                           D: tl.constexpr, TF32: tl.constexpr, EXP2: tl.constexpr,
                           TILE: tl.constexpr, BLOCK_K: tl.constexpr):
    """Online LSE of ``TILE`` query rows over one target slice; raw accumulators out."""
    im = tl.program_id(0) * TILE + tl.arange(0, TILE)
    split = tl.program_id(1)
    valid_m = im < N
    begin = split * SPLIT_SIZE
    end = tl.minimum(begin + SPLIT_SIZE, M)
    maximum = tl.full((TILE,), -float("inf"), tl.float32)
    total = tl.zeros((TILE,), tl.float32)
    inv_eps = 1. / EPS
    log2e = 1.4426950408889634
    scaled = COORD_SCALE * inv_eps
    for j0 in range(begin, end, TILE):
        jn = j0 + tl.arange(0, TILE)
        valid_n = jn < end
        g = tl.load(G + jn, valid_n, other=0.).to(tl.float32)
        lw = tl.load(LW + jn, valid_n, other=-float("inf")).to(tl.float32)
        dot = _tiled_dot(X, Y, im, jn, SX0, SX1, SY0, SY1,
                         D, valid_m, valid_n, TILE, TILE, BLOCK_K, TF32)
        if EXP2:
            bias = g * (inv_eps * log2e) + lw * log2e
            values = tl.fma(dot, scaled * log2e, bias[None, :])
        else:
            values = dot * scaled + (g * inv_eps + lw)[None, :]
        values = tl.where(valid_n[None, :], values, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(values, 1))
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        if EXP2:
            rescale = tl.exp2(maximum - safe_max)
            weights = tl.exp2(values - safe_max[:, None])
        else:
            rescale = tl.exp(maximum - safe_max)
            weights = tl.exp(values - safe_max[:, None])
        total = total * rescale + tl.sum(weights, 1)
        maximum = next_max
    offset = split * N + im
    tl.store(PM + offset, maximum, valid_m)
    tl.store(PS + offset, total, valid_m)


@triton.jit
def _repair_merge_kernel(PM, PS, OUT, N, SPLITS, EPS, DAMPING, EXP2: tl.constexpr,
                         BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
    total = tl.zeros((BLOCK,), tl.float32)
    for k in range(SPLITS):
        m = tl.load(PM + k * N + i, valid, other=-float("inf"))
        s = tl.load(PS + k * N + i, valid, other=0.)
        nxt = tl.maximum(maximum, m)
        safe = tl.where(nxt == -float("inf"), 0., nxt)
        if EXP2:
            total = total * tl.exp2(maximum - safe) + s * tl.exp2(m - safe)
        else:
            total = total * tl.exp(maximum - safe) + s * tl.exp(m - safe)
        maximum = nxt
    value = -EPS * DAMPING * _final_lse(maximum, total, EXP2)
    tl.store(OUT + i, value, valid)


def dense_lse_rows(x, y, g_hat, log_w, eps, damping, *, cost_scale=1.0,
                   allow_tf32=True, use_exp2=True):
    """Full-target shifted-potential LSE for the query rows ``x``."""
    n, m = len(x), len(y)
    splits = min(REPAIR_MAX_SPLITS, max(1, triton.cdiv(m, REPAIR_SPLIT_TARGETS)))
    split_size = triton.cdiv(m, splits)
    partial_m = torch.empty((splits, n), device=x.device, dtype=torch.float32)
    partial_s = torch.empty_like(partial_m)
    out = torch.empty(n, device=x.device, dtype=torch.float32)
    _repair_partial_kernel[(triton.cdiv(n, REPAIR_TILE), splits)](
        x, y, g_hat, log_w, partial_m, partial_s, n, m, split_size,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        2. * cost_scale, float(eps), D=x.shape[1], TF32=allow_tf32, EXP2=use_exp2,
        TILE=REPAIR_TILE, BLOCK_K=REPAIR_BLOCK_K, num_warps=4, num_stages=1)
    _repair_merge_kernel[(triton.cdiv(n, REPAIR_MERGE_BLOCK),)](
        partial_m, partial_s, out, n, splits, float(eps), float(damping),
        EXP2=use_exp2, BLOCK=REPAIR_MERGE_BLOCK)
    return out


def _half_step(x, y, g_hat, log_w, eps, damping, crow, col, *, block, cost_scale,
               allow_tf32, use_exp2, live=None, held=None):
    """Sparse half-step; rows of empty block rows are replaced by the dense result, except those of
    empty block rows that are not ``live`` (zero weight), which take ``held`` (their incoming values)."""
    out = blocksparse_lse_fused(
        x, y, g_hat, log_w, eps, damping, crow, col,
        block_m=block, block_n=block, cost_scale=cost_scale,
        allow_tf32=allow_tf32, use_exp2=use_exp2)
    empty_rows = crow[1:] == crow[:-1]
    if live is not None:
        out = torch.where((empty_rows & ~live).repeat_interleave(block), held, out)
        empty_rows = empty_rows & live
    empty = torch.nonzero(empty_rows, as_tuple=False).flatten()
    blocks_per_chunk = max(1, REPAIR_QUERY_CHUNK // block)
    offsets = torch.arange(block, device=x.device)
    for first in range(0, len(empty), blocks_per_chunk):
        idx = (empty[first:first + blocks_per_chunk][:, None] * block + offsets[None, :]).flatten()
        out[idx] = dense_lse_rows(
            x[idx].contiguous(), y, g_hat, log_w, eps, damping,
            cost_scale=cost_scale, allow_tf32=allow_tf32, use_exp2=use_exp2)
    return out


@torch.no_grad()
def sparse_symmetric_step(
    x, y, f_hat, g_hat, log_a, log_b, eps, crow, col, crow_t, col_t, *,
    block, alpha=.5, cost_scale=1.0, allow_tf32=True, use_exp2=True,
):
    """One symmetric update of both shifted potentials on a block mask.

    Args:
        x, y: Sorted, padded fp32 coordinates.
        f_hat, g_hat: Incoming shifted potentials.
        log_a, log_b: Log weights, with the zero-weight sentinel on padding.
        eps: Regularization.
        crow, col: Admitted mask, source blocks to target blocks.
        crow_t, col_t: Its transpose.
        block: Block size of the mask.
        alpha: Averaging weight; 0.5 for an ordinary update, 1.0 for a full step.
        cost_scale, allow_tf32, use_exp2: As in :func:`blocksparse_lse_fused`.

    Returns:
        ``((1 - alpha) f_hat + alpha f_new, (1 - alpha) g_hat + alpha g_new)``.
    """
    common = dict(block=block, cost_scale=cost_scale, allow_tf32=allow_tf32, use_exp2=use_exp2)
    live_x = block_log_mass(log_a, len(crow) - 1, block) > LOG_ZERO_SENTINEL / 2
    live_y = block_log_mass(log_b, len(crow_t) - 1, block) > LOG_ZERO_SENTINEL / 2
    f_new = _half_step(x, y, g_hat, log_b, eps, 1., crow, col, live=live_x, held=f_hat, **common)
    g_new = _half_step(y, x, f_hat, log_a, eps, 1., crow_t, col_t, live=live_y, held=g_hat, **common)
    return (1. - alpha) * f_hat + alpha * f_new, (1. - alpha) * g_hat + alpha * g_new
