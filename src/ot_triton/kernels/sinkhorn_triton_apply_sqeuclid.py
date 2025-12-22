from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _default_block_sizes(n: int, m: int, d: int) -> Tuple[int, int, int]:
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

    if d >= 64:
        block_k = 64
    elif d >= 32:
        block_k = 32
    else:
        block_k = 16

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

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
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

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
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

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
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

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
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

        cost = x2[:, None] + y2[None, :] - 2.0 * dot_xy
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


def apply_plan_vec_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    vec: torch.Tensor,
    *,
    eps: float,
    axis: int,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Apply P or Pᵀ to a vector without materializing P (streaming, stable).

    Computes:
      axis=1: out[i] = sum_j exp((f_i + g_j - C_ij)/eps) * vec[j]
      axis=0: out[j] = sum_i exp((f_i + g_j - C_ij)/eps) * vec[i]
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

    if block_m is None or block_n is None or block_k is None:
        block_m, block_n, block_k = _default_block_sizes(n, m, d)
    if block_k < 16:
        block_k = 16

    if axis == 1:
        out = torch.empty((n,), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(n, block_m),)
        _apply_plan_axis1_vec_kernel[grid](
            x,
            y,
            f,
            g,
            x2,
            y2,
            vec,
            out,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            x2.stride(0),
            y2.stride(0),
            vec.stride(0),
            out.stride(0),
            eps_f,
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

    out = torch.empty((m,), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_n),)
    _apply_plan_axis0_vec_kernel[grid](
        x,
        y,
        f,
        g,
        x2,
        y2,
        vec,
        out,
        n,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        f.stride(0),
        g.stride(0),
        x2.stride(0),
        y2.stride(0),
        vec.stride(0),
        out.stride(0),
        eps_f,
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
) -> torch.Tensor:
    """Apply P or Pᵀ to a matrix without materializing P (streaming, stable).

    Computes:
      axis=1: out[i, :] = sum_j exp((f_i + g_j - C_ij)/eps) * mat[j, :]
      axis=0: out[j, :] = sum_i exp((f_i + g_j - C_ij)/eps) * mat[i, :]

    If ``scale`` is provided (axis=1 only), the kernel uses `mat[j,:] * scale[j]`
    on the fly (avoids allocating `mat * scale[:,None]`).
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or mat.ndim != 2:
        raise ValueError("x,y,mat must be 2D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D.")

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

    if block_m is None or block_n is None or block_k is None:
        block_m, block_n, block_k = _default_block_sizes(n, m, d)
    if block_k < 16:
        block_k = 16

    if block_d is None:
        # Conservative default to avoid shared-memory overflows for larger configs.
        block_d = 16

    if axis == 1:
        out = torch.empty((n, d), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(n, block_m), triton.cdiv(d, block_d))
        scale_ptr = scale if scale is not None else mat
        stride_scale = int(scale.stride(0)) if scale is not None else 0
        _apply_plan_axis1_mat_kernel[grid](
            x,
            y,
            f,
            g,
            x2,
            y2,
            mat,
            scale_ptr,
            out,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            x2.stride(0),
            y2.stride(0),
            mat.stride(0),
            mat.stride(1),
            stride_scale,
            out.stride(0),
            out.stride(1),
            eps_f,
            D=d,
            BLOCK_D=block_d,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            HAS_SCALE=scale is not None,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    out = torch.empty((m, d), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_n), triton.cdiv(d, block_d))
    _apply_plan_axis0_mat_kernel[grid](
        x,
        y,
        f,
        g,
        x2,
        y2,
        mat,
        out,
        n,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        f.stride(0),
        g.stride(0),
        x2.stride(0),
        y2.stride(0),
        mat.stride(0),
        mat.stride(1),
        out.stride(0),
        out.stride(1),
        eps_f,
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
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Compute Mat5 term for HVP: Mat5 = (-4/eps) * sum_j P_ij (A_i·y_j) y_j."""

    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")

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

    if block_m is None or block_n is None or block_k is None:
        block_m, block_n, block_k = _default_block_sizes(n, m, d)
    if block_k < 16:
        block_k = 16

    block_d = max(16, 1 << (int(d) - 1).bit_length())

    out = torch.empty((n, d), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_m),)
    scale = -4.0 / eps_f
    _mat5_axis1_kernel[grid](
        x,
        y,
        f,
        g,
        A,
        x2,
        y2,
        out,
        n,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        f.stride(0),
        g.stride(0),
        A.stride(0),
        A.stride(1),
        x2.stride(0),
        y2.stride(0),
        out.stride(0),
        out.stride(1),
        eps_f,
        float(scale),
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
