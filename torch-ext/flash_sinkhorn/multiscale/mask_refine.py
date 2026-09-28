"""Exact verification of the block pairs the box screen admitted.

For each candidate ``(I, J)`` of :func:`block_mask.build_block_mask_csr` the
kernel evaluates the exact tile maximum the screen upper-bounds,

    m_IJ = max_{i in I, j in J} [ f_hat_i + g_hat_j + 2 * cost_scale * x_i.y_j ]
         = max_{i in I, j in J} [ (f_i + g_j - cost_scale * C_ij) ],

and keeps the pair when ``m_IJ > eps * log(delta_rel / max(a_I, b_J))``, the same
per-pair threshold the screen uses. Every pair the exact test admits passes the
screen, so the refined mask equals the exact-test set at block granularity, and
each dropped pair still carries the per-block omitted-mass bound. Zero-weight
padding is excluded from the maximum and from the block masses.

The statistic is computed in fp32 with the same tiled dot as the sparse LSE, so
the decision matches the arithmetic of the half-step that consumes the mask.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..kernels._triton_helpers import _tiled_dot
from ..kernels.sinkhorn_blocksparse_sqeuclid import _launch_for
from .block_mask import LOG_ZERO_SENTINEL, block_log_mass

#: ``(num_warps, num_stages)`` for each supported block size.
LAUNCH = {64: (2, 3), 128: (4, 2), 256: (8, 3)}


@triton.jit
def _tile_max_kernel(
    x_ptr,              # source coordinates [n, d]
    y_ptr,              # target coordinates [m, d]
    f_row_ptr,          # f_hat, -inf on zero-weight slots [n]
    g_col_ptr,          # g_hat, -inf on zero-weight slots [m]
    keep_ptr,           # keep flag per candidate [nnz], int8
    score_ptr,          # m_IJ per candidate [nnz], fp32 (written if WRITE_SCORES)
    crow_ptr,           # CSR row pointers [N_B + 1], int32
    col_ptr,            # CSR column indices [nnz], int32
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    coord_scale,        # 2 * cost_scale
    la_ptr,             # log a_I [N_B]
    lb_ptr,             # log b_J [M_B]
    eps,
    log_drel,           # log(delta_rel)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per source block row, one keep flag per candidate tile."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n
    f_row = tl.load(f_row_ptr + offs_m, mask=mask_m,
                    other=-float("inf")).to(tl.float32)

    row_start = tl.load(crow_ptr + pid).to(tl.int32)
    row_end = tl.load(crow_ptr + pid + 1).to(tl.int32)
    la_I = tl.load(la_ptr + pid).to(tl.float32)

    for jj in range(row_end - row_start):
        col_idx = tl.load(col_ptr + row_start + jj).to(tl.int32)
        lb_J = tl.load(lb_ptr + col_idx).to(tl.float32)
        offs_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m
        g_col = tl.load(g_col_ptr + offs_n, mask=mask_n,
                        other=-float("inf")).to(tl.float32)

        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        vals = dot * coord_scale + f_row[:, None] + g_col[None, :]
        pmax = tl.max(vals)
        t_IJ = eps * (log_drel - tl.maximum(la_I, lb_J))
        tl.store(keep_ptr + row_start + jj, (pmax > t_IJ).to(tl.int8))
        if WRITE_SCORES:
            tl.store(score_ptr + row_start + jj, pmax)


def refine_mask_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    crow: torch.Tensor,
    col: torch.Tensor,
    *,
    B: int,
    delta_rel: float,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    return_scores: bool = False,
    la_block: torch.Tensor = None,
    lb_block: torch.Tensor = None,
):
    """Keep the candidates whose exact tile maximum exceeds the per-pair threshold.

    Args:
        x, y: Sorted, padded coordinates the sparse half-steps use.
        f_hat, g_hat: Shifted potentials on the padded clouds.
        log_a, log_b: Log weights, ``LOG_ZERO_SENTINEL`` on padding.
        eps: Regularization of the update the mask serves.
        crow, col: Candidate CSR from the box screen.
        B: Block size, a key of ``LAUNCH``.
        delta_rel: Per-block omitted-mass budget, as in the screen.
        cost_scale: 1.0 for ``||x - y||^2``, 0.5 for ``||x - y||^2 / 2``.
        allow_tf32: TF32 tensor-core dot products.
        return_scores: Also return ``m_IJ`` for every candidate.
        la_block, lb_block: The block log-masses of ``log_a`` and ``log_b``, when the caller has them
            (the screen does); computed here otherwise.

    Returns:
        ``(crow, col)`` of the admitted pairs, a subset of the candidates, and,
        with ``return_scores``, the ``[nnz_candidates]`` fp32 tile maxima.
    """
    num_warps, num_stages = _launch_for(LAUNCH, B, B)
    device = x.device
    n, d = x.shape
    m = y.shape[0]
    nnz = int(col.numel())
    if nnz == 0:
        return (crow, col, None) if return_scores else (crow, col)

    N_B = crow.shape[0] - 1
    M_B = m // B
    la_block = block_log_mass(log_a, N_B, B) if la_block is None else la_block.contiguous()
    lb_block = block_log_mass(log_b, M_B, B) if lb_block is None else lb_block.contiguous()
    log_drel = math.log(delta_rel)

    f_row = f_hat.float().masked_fill(log_a <= LOG_ZERO_SENTINEL, -torch.inf).contiguous()
    g_col = g_hat.float().masked_fill(log_b <= LOG_ZERO_SENTINEL, -torch.inf).contiguous()

    keep = torch.empty(nnz, dtype=torch.int8, device=device)
    scores = (torch.empty(nnz, dtype=torch.float32, device=device)
              if return_scores else keep)
    x = x.contiguous()
    y = y.contiguous()
    crow = crow.contiguous().to(torch.int32)
    col = col.contiguous().to(torch.int32)
    args = (x, y, f_row, g_col, keep, scores, crow, col, n, m,
            x.stride(0), x.stride(1), y.stride(0), y.stride(1),
            2.0 * cost_scale, la_block, lb_block, float(eps), log_drel)

    _tile_max_kernel[(N_B,)](
        *args, D=d, ALLOW_TF32=allow_tf32, WRITE_SCORES=return_scores,
        BLOCK_M=B, BLOCK_N=B, BLOCK_K=32 if d >= 32 else 16,
        num_warps=num_warps, num_stages=num_stages,
    )

    keep_b = keep.bool()
    col2 = col[keep_b]
    # crow2[i] = number of kept candidates among the first crow[i]; it fits int32
    # because the kept set is a subset of a CSR that passed the preflight.
    cs = torch.zeros(nnz + 1, dtype=torch.int64, device=device)
    cs[1:] = torch.cumsum(keep_b.to(torch.int64), dim=0)
    crow2 = cs[crow.long()].to(torch.int32)
    return (crow2, col2, scores) if return_scores else (crow2, col2)
