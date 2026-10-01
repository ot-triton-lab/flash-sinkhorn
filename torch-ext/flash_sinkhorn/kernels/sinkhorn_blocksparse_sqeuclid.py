"""Block-sparse FlashSinkhorn LSE half-step over a CSR mask of block pairs.

For a source block row ``I`` the kernel computes, over the target blocks ``J``
listed for ``I`` in the CSR mask,

    out_i = -eps * damping * LSE_{j in J in mask(I)} [ coord_scale * x_i.y_j / eps
                                                     + g_hat_j / eps + log w_j ],

with ``coord_scale = 2 * cost_scale``. Inside the loop the arithmetic is the
dense FlashSinkhorn kernel's (tiled dot, bias, online log-sum-exp); only the
inner loop differs, running over the row's admitted blocks through the CSR
indirection instead of over all target blocks.

``BLOCK_M`` and ``BLOCK_N`` are structural: they must equal the block size the
mask was built with, and both clouds must be padded to a multiple of it. The
loop bound is each row's own number of admitted blocks, so a row with none
exits with an empty sum; the multiscale solver repairs such rows separately.

The launch configuration is fixed per block size, which avoids autotuning every
new problem size (one warp per 32 rows of a block).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from ._triton_helpers import _final_lse, _online_softmax_rescale, _tiled_dot

#: ``(num_warps, num_stages)`` for each supported block size, chosen for 16 padded columns (wider inputs
#: run correctly with BLOCK_K 32).
LAUNCH = {64: (2, 2), 128: (4, 3), 256: (8, 2)}


def _launch_for(launch: dict, block_m: int, block_n: int) -> Tuple[int, int]:
    """The entry of ``launch`` for the larger of the two block sizes."""
    if block_m not in launch or block_n not in launch:
        raise ValueError(f"block sizes must be among {sorted(launch)}, got {block_m} and {block_n}")
    return launch[max(block_m, block_n)]


@triton.jit
def _blocksparse_lse_kernel(
    x_ptr,              # source coordinates [n, d]
    y_ptr,              # target coordinates [m, d]
    g_hat_ptr,          # shifted target potential [m]
    log_w_ptr,          # log target weights [m]
    out_ptr,            # shifted source potential [n]
    crow_ptr,           # CSR row pointers [N_B + 1], int32
    col_ptr,            # CSR column indices [nnz], int32
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_out,
    coord_scale,        # 2 * cost_scale
    eps,
    damping,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per source block row, iterating over its admitted blocks."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    log2e = 1.4426950408889634
    inv_eps = 1.0 / eps
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    row_start = tl.load(crow_ptr + pid).to(tl.int32)
    row_end = tl.load(crow_ptr + pid + 1).to(tl.int32)
    for jj in range(row_end - row_start):
        col_idx = tl.load(col_ptr + row_start + jj).to(tl.int32)
        offs_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g_hat = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)
        log_w = tl.load(log_w_ptr + offs_n, mask=mask_n, other=-float("inf"),
                        eviction_policy="evict_first").to(tl.float32)

        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = g_hat * inv_eps_log2 + log_w * log2e
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :])
        else:
            bias = g_hat * inv_eps + log_w
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        new_m, alpha, w = _online_softmax_rescale(vals, m_i, USE_EXP2)
        s_i = s_i * alpha + tl.sum(w, axis=1)
        m_i = new_m

    lse = _final_lse(m_i, s_i, USE_EXP2)
    out = -eps * damping * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


def blocksparse_lse_fused(
    x: torch.Tensor,
    y: torch.Tensor,
    g_hat: torch.Tensor,
    log_w: torch.Tensor,
    eps: float,
    damping: float,
    crow: torch.Tensor,
    col: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
) -> torch.Tensor:
    """Block-sparse shifted-potential half-step.

    Args:
        x: Source coordinates ``[n, d]``, fp32, padded to a multiple of ``block_m``.
        y: Target coordinates ``[m, d]``, fp32, padded to a multiple of ``block_n``.
        g_hat: Shifted target potential ``g - cost_scale * ||y||^2`` ``[m]``.
        log_w: Log target weights ``[m]`` (not scaled by ``eps``).
        eps: Regularization.
        damping: Unbalanced damping factor, 1.0 for balanced OT.
        crow, col: int32 CSR of admitted ``(I, J)`` block pairs.
        block_m, block_n: Block sizes the mask was built with, each a key of ``LAUNCH``.
        cost_scale: 1.0 for ``||x - y||^2``, 0.5 for ``||x - y||^2 / 2``.
        allow_tf32: TF32 tensor-core dot products.
        use_exp2: exp2/log2 arithmetic.

    Returns:
        Shifted source potential ``[n]`` in fp32.
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2-D with the same number of columns")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("x and y must be float32")

    num_warps, num_stages = _launch_for(LAUNCH, block_m, block_n)
    n, d = x.shape
    m = y.shape[0]
    x = x.contiguous()
    y = y.contiguous()
    g_hat = g_hat.contiguous().float()
    log_w = log_w.contiguous().float()
    crow = crow.contiguous().to(torch.int32)
    col = col.contiguous().to(torch.int32)
    out = torch.empty(n, device=x.device, dtype=torch.float32)
    _blocksparse_lse_kernel[(crow.shape[0] - 1,)](
        x, y, g_hat, log_w, out, crow, col, n, m,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1), out.stride(0),
        2.0 * cost_scale, eps, damping,
        D=d, ALLOW_TF32=allow_tf32, USE_EXP2=use_exp2,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=32 if d >= 32 else 16,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
