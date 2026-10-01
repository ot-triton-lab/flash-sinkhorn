"""Split-N FlashSinkhorn LSE for few query rows against many targets.

    out_i = -eps * damping * LSE_j [ coord_scale * x_i.y_j / eps + bias_j ],

with ``coord_scale = 2 * cost_scale``. When the rows are few (a sampled marginal
check queries about a thousand points against the whole opposite cloud), one
program per row block leaves most SMs idle. This path splits the target
dimension into ``num_splits`` slices: each program reduces one row block over one
slice to a partial ``(max, sum)`` pair, and a second kernel merges the partials.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._triton_helpers import _tiled_dot
from .sinkhorn_flashstyle_sqeuclid import _default_block_sizes_alternating

#: Fixed launch of the partial kernel.
PARTIAL_NUM_WARPS = 4
PARTIAL_NUM_STAGES = 3
#: Rows per program in the merge kernel.
REDUCE_BLOCK = 256


@triton.jit
def _lse_splitn_partial_kernel(
    x_ptr,           # query coordinates [n, d]
    y_ptr,           # target coordinates [m, d]
    bias_ptr,        # pre-scaled bias [m]
    partial_m_ptr,   # partial maxima [num_splits, n]
    partial_s_ptr,   # partial sums [num_splits, n]
    n,
    m,
    split_size,      # targets per split
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_bias,
    coord_scale,     # 2 * cost_scale
    eps,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Online LSE of one row block over one target slice; writes raw accumulators."""
    row_block = tl.program_id(0)
    split_id = tl.program_id(1)

    offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    col_start = split_id * split_size
    col_end = tl.minimum(col_start + split_size, m)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    inv_eps = 1.0 / eps
    scaled_inv_eps = coord_scale * inv_eps
    log2e = 1.4426950408889634
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    for j0 in range(col_start, col_end, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < col_end

        bias = tl.load(
            bias_ptr + offs_n * stride_bias,
            mask=mask_n, other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)

        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        if USE_EXP2:
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :] * log2e)
        else:
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # A row whose values so far are all -inf keeps a zero sum instead of exp(-inf + inf) = NaN.
        new_m = tl.maximum(m_i, tl.max(vals, axis=1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        if USE_EXP2:
            alpha = tl.exp2(m_i - safe_m)
            w = tl.exp2(vals - safe_m[:, None])
        else:
            alpha = tl.exp(m_i - safe_m)
            w = tl.exp(vals - safe_m[:, None])
        s_i = s_i * alpha + tl.sum(w, axis=1)
        m_i = new_m

    ws_offset = split_id * n + offs_m
    tl.store(partial_m_ptr + ws_offset, m_i, mask=mask_m)
    tl.store(partial_s_ptr + ws_offset, s_i, mask=mask_m)


@triton.jit
def _lse_splitn_reduce_kernel(
    partial_m_ptr,   # [num_splits, n]
    partial_s_ptr,   # [num_splits, n]
    out_ptr,         # [n]: -eps * damping * LSE
    n,
    num_splits,
    eps,
    damping,
    USE_EXP2: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Merge the partial accumulators in the same base (exp2 or exp) they were built in."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = rows < n

    m_acc = tl.full([BLOCK_B], -float("inf"), tl.float32)
    s_acc = tl.zeros([BLOCK_B], tl.float32)

    for k in range(num_splits):
        m_k = tl.load(partial_m_ptr + k * n + rows, mask=mask, other=-float("inf"))
        s_k = tl.load(partial_s_ptr + k * n + rows, mask=mask, other=0.0)
        new_m = tl.maximum(m_acc, m_k)
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        if USE_EXP2:
            s_acc = s_acc * tl.exp2(m_acc - safe_m) + s_k * tl.exp2(m_k - safe_m)
        else:
            s_acc = s_acc * tl.exp(m_acc - safe_m) + s_k * tl.exp(m_k - safe_m)
        m_acc = new_m

    s_safe = tl.maximum(s_acc, 1e-40)
    if USE_EXP2:
        ln2: tl.constexpr = 0.6931471805599453
        lse = (m_acc + tl.log2(s_safe)) * ln2
    else:
        lse = m_acc + tl.log(s_safe)

    out = -eps * damping * lse
    tl.store(out_ptr + rows, out, mask=mask)


def splitn_workspace(num_splits: int, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the ``(partial_max, partial_sum)`` buffers, each ``[num_splits, n]`` fp32."""
    return (torch.empty((num_splits, n), device=device, dtype=torch.float32),
            torch.empty((num_splits, n), device=device, dtype=torch.float32))


def flashsinkhorn_lse_splitn(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    num_splits: int,
    cost_scale: float = 1.0,
    damping: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    workspace: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Shifted-potential LSE with the target dimension split into ``num_splits`` slices.

    Args:
        x: Query coordinates ``[n, d]``.
        y: Target coordinates ``[m, d]``.
        bias: Pre-scaled bias ``g_hat / eps + log w`` ``[m]``.
        eps: Regularization.
        num_splits: Number of target slices.
        cost_scale: 1.0 for ``||x - y||^2``, 0.5 for ``||x - y||^2 / 2``.
        damping: Unbalanced damping factor, 1.0 for balanced OT.
        allow_tf32: TF32 tensor-core dot products.
        use_exp2: exp2/log2 arithmetic.
        workspace: Optional buffers from :func:`splitn_workspace`, reused across calls.

    Returns:
        ``out [n]`` in fp32.
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2-D with the same number of columns")
    if bias.ndim != 1 or bias.shape[0] != y.shape[0]:
        raise ValueError("bias must be 1-D with one entry per target")
    if num_splits < 1:
        raise ValueError("num_splits must be positive")

    n, d = x.shape
    m = y.shape[0]
    x = x.contiguous().float()
    y = y.contiguous().float()
    bias = bias.contiguous().float()
    out = torch.empty((n,), device=x.device, dtype=torch.float32)

    if workspace is None:
        workspace = splitn_workspace(num_splits, n, x.device)
    partial_m, partial_s = workspace
    if partial_m.shape != (num_splits, n) or partial_s.shape != (num_splits, n):
        raise ValueError("workspace shape does not match (num_splits, n)")

    split_size = triton.cdiv(m, num_splits)
    bm, bn, bk, _ = _default_block_sizes_alternating(n, m, d)
    _lse_splitn_partial_kernel[(triton.cdiv(n, bm), num_splits)](
        x, y, bias, partial_m, partial_s,
        n, m, split_size,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1), bias.stride(0),
        float(2.0 * cost_scale), float(eps),
        D=d, ALLOW_TF32=allow_tf32, USE_EXP2=use_exp2,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=PARTIAL_NUM_WARPS, num_stages=PARTIAL_NUM_STAGES,
    )
    _lse_splitn_reduce_kernel[(triton.cdiv(n, REDUCE_BLOCK),)](
        partial_m, partial_s, out, n, num_splits, float(eps), float(damping),
        USE_EXP2=use_exp2, BLOCK_B=REDUCE_BLOCK,
    )
    return out
