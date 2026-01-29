from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


def _validate_device(
    x: torch.Tensor,
    tensors: Sequence[Tuple[str, Optional[torch.Tensor]]],
) -> None:
    """Validate that all tensors are on the same CUDA device as x.

    Args:
        x: Reference tensor (must be on CUDA)
        tensors: List of (name, tensor) pairs to validate

    Raises:
        ValueError: If any tensor is not on the same CUDA device as x
    """
    ref_device = x.device
    for name, tensor in tensors:
        if tensor is not None:
            if not tensor.is_cuda or tensor.device != ref_device:
                raise ValueError(
                    f"{name} must be on the same CUDA device as x ({ref_device}), "
                    f"got {tensor.device if tensor.is_cuda else 'CPU'}"
                )


def _apply_vec_autotune_configs() -> Sequence[triton.Config]:
    """Autotune configs for apply_plan_vec kernels.

    Curated configs based on tuning experiments for d=64.
    Reduced from 108 to 12 configs for faster autotuning.
    """
    # Top performers from tuning at n=4096, 10000, 20000
    return [
        # Best overall: BLOCK_M=64, BLOCK_N=64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        # Good for small n
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        # Good for large n
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # Fallback options
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ]


def _apply_mat_autotune_configs() -> Sequence[triton.Config]:
    """Autotune configs for apply_plan_mat and mat5 kernels.

    Curated configs based on tuning experiments. Includes:
    - Small BLOCK_D (16-64): For typical d <= 64 dimensions
    - Medium BLOCK_D (128-256): For d up to 256
    - Large BLOCK_D (512-2048): For high-dimensional features (d > 256)

    Config pruning functions filter these at runtime based on actual D
    to avoid compiling wasteful or invalid configs.
    """
    # Top performers - similar pattern to vec but with BLOCK_D
    return [
        # Best overall
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        # Good for small n
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        # Good for large n
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        # Fallback options
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=64 for d >= 64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=128 for d >= 128
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=256 for d >= 256 (reduced BLOCK_M/N for register pressure)
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=512 for d >= 512
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_D": 512}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=1024 for d >= 1024
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_D": 1024}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=2048 for d >= 2048
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_D": 2048}, num_warps=4, num_stages=2),
    ]


def _default_block_sizes(n: int, m: int, d: int) -> Tuple[int, int, int]:
    """Default block sizes for apply kernels (vec, mat, mat5).

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.

    The key constraint is: BLOCK_K < D (must have at least 2 k iterations).
    - d >= 64: block_k = 32 → at least 2 iterations
    - d >= 32: block_k = 16 → at least 2 iterations
    - d < 32:  block_k = 16 → minimum for tl.dot
    """
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    elif n >= 32:
        block_m = 32
    else:
        block_m = 16

    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    elif m >= 32:
        block_n = 32
    else:
        block_n = 16

    # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

    return block_m, block_n, block_k


@triton.jit
def _apply_plan_axis1_vec_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    eps,
    cost_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)
        vec = tl.load(vec_ptr + offs_n * stride_vec, mask=mask_n, other=0.0).to(
            tl.float32
        )

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp2(logits - new_m[:, None])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None])
            )

        s_i = s_i * alpha + tl.sum(w * vec[None, :], axis=1)
        m_i = new_m

    if USE_EXP2:
        out = tl.exp2(m_i) * s_i
    else:
        out = tl.exp(m_i) * s_i
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


@triton.jit
def _apply_plan_axis0_vec_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    eps,
    cost_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    inv_eps = 1.0 / eps

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)
        vec = tl.load(vec_ptr + offs_m * stride_vec, mask=mask_m, other=0.0).to(
            tl.float32
        )

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_m[:, None], logits, -float("inf"))

        block_max = tl.max(logits, axis=0)
        new_m = tl.maximum(m_j, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp2(logits - new_m[None, :])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp(logits - new_m[None, :])
            )

        s_j = s_j * alpha + tl.sum(w * vec[:, None], axis=0)
        m_j = new_m

    if USE_EXP2:
        out = tl.exp2(m_j) * s_j
    else:
        out = tl.exp(m_j) * s_j
    tl.store(out_ptr + offs_n * stride_out, out, mask=mask_n)


# Create autotuned versions of vec kernels
# Note: key uses n, m (runtime args at positions 8, 9) for autotune caching
_apply_plan_axis1_vec_kernel_autotune = triton.autotune(
    configs=_apply_vec_autotune_configs(),
    key=["n", "m"],
)(_apply_plan_axis1_vec_kernel)

_apply_plan_axis0_vec_kernel_autotune = triton.autotune(
    configs=_apply_vec_autotune_configs(),
    key=["n", "m"],
)(_apply_plan_axis0_vec_kernel)


@triton.jit
def _apply_plan_axis1_mat_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    mat_ptr,
    scale_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_mat0,
    stride_mat1,
    stride_scale,
    stride_out0,
    stride_out1,
    eps,
    cost_scale,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp2(logits - new_m[:, None])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None])
            )

        mat_tile = tl.load(
            mat_ptr + offs_n[:, None] * stride_mat0 + offs_d[None, :] * stride_mat1,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(scale_ptr + offs_n * stride_scale, mask=mask_n, other=0.0).to(
                tl.float32
            )
            mat_tile = mat_tile * scale[:, None]

        o = o * alpha[:, None] + tl.dot(w, mat_tile, allow_tf32=ALLOW_TF32)
        m_i = new_m

    if USE_EXP2:
        out = tl.exp2(m_i)[:, None] * o
    else:
        out = tl.exp(m_i)[:, None] * o

    tl.store(
        out_ptr + offs_m[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _apply_plan_axis0_mat_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    mat_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_mat0,
    stride_mat1,
    stride_out0,
    stride_out1,
    eps,
    cost_scale,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    o = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_m[:, None], logits, -float("inf"))

        block_max = tl.max(logits, axis=0)
        new_m = tl.maximum(m_j, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp2(logits - new_m[None, :])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp(logits - new_m[None, :])
            )

        mat_tile = tl.load(
            mat_ptr + offs_m[:, None] * stride_mat0 + offs_d[None, :] * stride_mat1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        o = o * alpha[:, None] + tl.dot(tl.trans(w), mat_tile, allow_tf32=ALLOW_TF32)
        m_j = new_m

    if USE_EXP2:
        out = tl.exp2(m_j)[:, None] * o
    else:
        out = tl.exp(m_j)[:, None] * o

    tl.store(
        out_ptr + offs_n[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_n[:, None] & mask_d[None, :],
    )


@triton.jit
def _mat5_axis1_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    a_ptr,
    x2_ptr,
    y2_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_a0,
    stride_a1,
    stride_x2,
    stride_y2,
    stride_out0,
    stride_out1,
    eps,
    scale,
    cost_scale,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(
            y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
        ).to(tl.float32)

        dot_xy = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        dot_ay = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D

            xk = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            ak = tl.load(
                a_ptr + offs_m[:, None] * stride_a0 + offs_k[None, :] * stride_a1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            yk = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )

            dot_xy += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)
            dot_ay += tl.dot(ak, yk, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot_xy)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        if USE_EXP2:
            alpha = tl.exp2(m_i - new_m)
            w = tl.exp2(logits - new_m[:, None])
        else:
            alpha = tl.exp(m_i - new_m)
            w = tl.exp(logits - new_m[:, None])

        tmp = w * dot_ay

        o = o * alpha[:, None]
        yv_t = tl.load(
            y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
            mask=mask_d[:, None]
            & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
            other=0.0,
        ).to(tl.float32)
        yv = tl.trans(yv_t)  # (BLOCK_N, BLOCK_D)
        o += tl.dot(tmp, yv, allow_tf32=ALLOW_TF32)
        m_i = new_m

    if USE_EXP2:
        out = tl.exp2(m_i)[:, None] * o
    else:
        out = tl.exp(m_i)[:, None] * o
    out = out * scale

    tl.store(
        out_ptr + offs_m[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _apply_mat_prune_configs(configs, named_args, **kwargs):
    """Prune apply_plan_mat autotune configs for given D.

    Unlike mat5, these kernels DO tile over D (2D grid), so BLOCK_D can be < D.
    We only prune excessively large BLOCK_D configs to reduce compile time.
    """
    D = named_args.get("D", 64)
    # Upper bound: BLOCK_D should be <= max(D, 64) to avoid wasteful configs
    # For small D, allow up to 64; for large D, allow up to D itself
    max_block_d = max(D, 64)

    pruned = [
        cfg for cfg in configs
        if cfg.kwargs.get("BLOCK_D", 16) <= max_block_d
    ]

    # Safety: if all configs pruned, keep smallest ones
    if not pruned:
        min_block_d = min(cfg.kwargs.get("BLOCK_D", 16) for cfg in configs)
        pruned = [cfg for cfg in configs if cfg.kwargs.get("BLOCK_D", 16) == min_block_d]

    return pruned


# Create autotuned versions of mat kernels
# Note: key uses n, m, D (runtime args) for autotune caching
_apply_plan_axis1_mat_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["n", "m", "D"],
    prune_configs_by={"early_config_prune": _apply_mat_prune_configs},
)(_apply_plan_axis1_mat_kernel)

_apply_plan_axis0_mat_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["n", "m", "D"],
    prune_configs_by={"early_config_prune": _apply_mat_prune_configs},
)(_apply_plan_axis0_mat_kernel)

def _mat5_prune_configs(configs, named_args, **kwargs):
    """Prune mat5 autotune configs to valid range for given D.

    The mat5 kernel does NOT tile along the D dimension (only 1D grid),
    so it requires BLOCK_D >= D to process all feature dimensions.

    Also prunes excessively large BLOCK_D configs (BLOCK_D > 4*D) to:
    - Reduce compile time on first run
    - Avoid OutOfResources on smaller GPUs with limited shared memory
    """
    D = named_args.get("D", 64)
    # Lower bound: BLOCK_D must be >= D (kernel doesn't tile over D)
    # Upper bound: BLOCK_D should be <= 4*D (avoid wasteful configs)
    #   Exception: always allow at least one config (minimum BLOCK_D that fits)
    min_valid_block_d = D
    max_block_d = max(D * 4, 128)  # At least allow BLOCK_D=128 for small D

    pruned = [
        cfg for cfg in configs
        if min_valid_block_d <= cfg.kwargs.get("BLOCK_D", 16) <= max_block_d
    ]

    # Safety: if all configs pruned (shouldn't happen), keep smallest valid ones
    if not pruned:
        pruned = [cfg for cfg in configs if cfg.kwargs.get("BLOCK_D", 16) >= D]

    return pruned


# Create autotuned version of mat5 kernel with config pruning
_mat5_axis1_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["n", "m", "D"],  # Include D in key since BLOCK_D must >= D
    prune_configs_by={"early_config_prune": _mat5_prune_configs},
)(_mat5_axis1_kernel)


def apply_plan_vec_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    vec: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Apply P or Pᵀ to a vector without materializing P (streaming, stable).

    Computes:
      axis=1: out[i] = sum_j exp((f_i + g_j - cost_scale*C_ij)/eps) * vec[j]
      axis=0: out[j] = sum_i exp((f_i + g_j - cost_scale*C_ij)/eps) * vec[i]

    Args:
        cost_scale: Scaling for cost function. 1.0 for full ||x-y||², 0.5 for half.
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (n,d) and (m,d).")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D.")
    if vec.ndim != 1:
        raise ValueError("vec must be 1D.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("vec", vec), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")

    if axis == 1 and vec.shape[0] != m:
        raise ValueError("vec must have shape (m,) for axis=1.")
    if axis == 0 and vec.shape[0] != n:
        raise ValueError("vec must have shape (n,) for axis=0.")

    eps_f = float(eps)
    x2 = (x.float() * x.float()).sum(dim=1).contiguous() if x2 is None else x2.float().contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous() if y2 is None else y2.float().contiguous()

    f = f.float().contiguous()
    g = g.float().contiguous()
    vec = vec.float().contiguous()

    # Determine whether to use autotuning
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
    )
    use_autotune = autotune and not manual_blocks

    if not use_autotune:
        # Manual block sizes
        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16

    out = torch.empty((n,) if axis == 1 else (m,), device=x.device, dtype=torch.float32)

    if axis == 1:
        if use_autotune:
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)
            _apply_plan_axis1_vec_kernel_autotune[grid](
                x, y, f, g, x2, y2, vec, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                vec.stride(0), out.stride(0),
                eps_f,
                float(cost_scale),
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
            )
        else:
            grid = (triton.cdiv(n, block_m),)
            _apply_plan_axis1_vec_kernel[grid](
                x, y, f, g, x2, y2, vec, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                vec.stride(0), out.stride(0),
                eps_f,
                float(cost_scale),
                D=d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        if use_autotune:
            def grid(meta):
                return (triton.cdiv(m, meta["BLOCK_N"]),)
            _apply_plan_axis0_vec_kernel_autotune[grid](
                x, y, f, g, x2, y2, vec, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                vec.stride(0), out.stride(0),
                eps_f,
                float(cost_scale),
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
            )
        else:
            grid = (triton.cdiv(m, block_n),)
            _apply_plan_axis0_vec_kernel[grid](
                x, y, f, g, x2, y2, vec, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                vec.stride(0), out.stride(0),
                eps_f,
                float(cost_scale),
                D=d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return out


def apply_plan_mat_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    mat: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    scale: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    block_d: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Apply P or Pᵀ to a matrix without materializing P (streaming, stable).

    Computes:
      axis=1: out[i, :] = sum_j exp((f_i + g_j - cost_scale*C_ij)/eps) * mat[j, :]
      axis=0: out[j, :] = sum_i exp((f_i + g_j - cost_scale*C_ij)/eps) * mat[i, :]

    If ``scale`` is provided (axis=1 only), the kernel uses `mat[j,:] * scale[j]`
    on the fly (avoids allocating `mat * scale[:,None]`).

    Args:
        cost_scale: Scaling for cost function. 1.0 for full ||x-y||², 0.5 for half.
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or mat.ndim != 2:
        raise ValueError("x,y,mat must be 2D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("mat", mat), ("scale", scale), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")

    if axis == 1 and mat.shape != (m, d):
        raise ValueError("For axis=1, mat must have shape (m, d).")
    if axis == 0 and mat.shape != (n, d):
        raise ValueError("For axis=0, mat must have shape (n, d).")

    if scale is not None:
        if axis != 1:
            raise ValueError("scale is only supported for axis=1.")
        if scale.shape != (m,):
            raise ValueError("scale must have shape (m,).")

    eps_f = float(eps)
    x2 = (x.float() * x.float()).sum(dim=1).contiguous() if x2 is None else x2.float().contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous() if y2 is None else y2.float().contiguous()

    f = f.float().contiguous()
    g = g.float().contiguous()
    mat = mat.float().contiguous()
    if scale is not None:
        scale = scale.float().contiguous()

    # Determine whether to use autotuning
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or block_d is not None
    )
    use_autotune = autotune and not manual_blocks

    if not use_autotune:
        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16
        if block_d is None:
            block_d = 16

    scale_ptr = scale if scale is not None else mat
    stride_scale = int(scale.stride(0)) if scale is not None else 0
    has_scale = scale is not None

    if axis == 1:
        out = torch.empty((n, d), device=x.device, dtype=torch.float32)
        if use_autotune:
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]), triton.cdiv(d, meta["BLOCK_D"]))
            _apply_plan_axis1_mat_kernel_autotune[grid](
                x, y, f, g, x2, y2, mat, scale_ptr, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                mat.stride(0), mat.stride(1),
                stride_scale,
                out.stride(0), out.stride(1),
                eps_f,
                float(cost_scale),
                D=d,
                HAS_SCALE=has_scale,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
            )
        else:
            grid = (triton.cdiv(n, block_m), triton.cdiv(d, block_d))
            _apply_plan_axis1_mat_kernel[grid](
                x, y, f, g, x2, y2, mat, scale_ptr, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                f.stride(0), g.stride(0),
                x2.stride(0), y2.stride(0),
                mat.stride(0), mat.stride(1),
                stride_scale,
                out.stride(0), out.stride(1),
                eps_f,
                float(cost_scale),
                D=d,
                BLOCK_D=block_d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                HAS_SCALE=has_scale,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return out

    out = torch.empty((m, d), device=x.device, dtype=torch.float32)
    if use_autotune:
        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_N"]), triton.cdiv(d, meta["BLOCK_D"]))
        _apply_plan_axis0_mat_kernel_autotune[grid](
            x, y, f, g, x2, y2, mat, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            x2.stride(0), y2.stride(0),
            mat.stride(0), mat.stride(1),
            out.stride(0), out.stride(1),
            eps_f,
            float(cost_scale),
            D=d,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
        )
    else:
        grid = (triton.cdiv(m, block_n), triton.cdiv(d, block_d))
        _apply_plan_axis0_mat_kernel[grid](
            x, y, f, g, x2, y2, mat, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            x2.stride(0), y2.stride(0),
            mat.stride(0), mat.stride(1),
            out.stride(0), out.stride(1),
            eps_f,
            float(cost_scale),
            D=d,
            BLOCK_D=block_d,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def mat5_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    cost_scale: float = 1.0,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Compute Mat5 term for HVP: Mat5 = (-4*cost_scale/eps) * sum_j P_ij (A_i·y_j) y_j.

    Args:
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """

    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("A", A), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)
    x2 = (x.float() * x.float()).sum(dim=1).contiguous() if x2 is None else x2.float().contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous() if y2 is None else y2.float().contiguous()

    f = f.float().contiguous()
    g = g.float().contiguous()
    A = A.float().contiguous()

    # Determine whether to use autotuning
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
    )
    use_autotune = autotune and not manual_blocks

    if not use_autotune:
        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16

    block_d = max(16, 1 << (int(d) - 1).bit_length())

    out = torch.empty((n, d), device=x.device, dtype=torch.float32)
    scale = -4.0 * cost_scale / eps_f

    if use_autotune:
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)
        _mat5_axis1_kernel_autotune[grid](
            x, y, f, g, A, x2, y2, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            A.stride(0), A.stride(1),
            x2.stride(0), y2.stride(0),
            out.stride(0), out.stride(1),
            eps_f,
            float(scale),
            float(cost_scale),
            D=d,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
        )
    else:
        grid = (triton.cdiv(n, block_m),)
        _mat5_axis1_kernel[grid](
            x, y, f, g, A, x2, y2, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            A.stride(0), A.stride(1),
            x2.stride(0), y2.stride(0),
            out.stride(0), out.stride(1),
            eps_f,
            float(scale),
            float(cost_scale),
            D=d,
            BLOCK_D=block_d,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
