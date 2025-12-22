from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    BlockSparseTaskCSRBuckets,
    BlockSparseTaskCSR,
)


@triton.jit
def _apply_plan_axis1_vec_taskcsr_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    vec_ptr,
    out_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    prog_x_cluster_ptr,
    prog_x_block_ptr,
    row_ptr_x_ptr,
    nbr_y_cluster_ptr,
    nbr_y_block_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_vec,
    stride_out,
    eps,
    MAX_TASKS_X: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    cluster = tl.load(prog_x_cluster_ptr + pid).to(tl.int32)
    blk = tl.load(prog_x_block_ptr + pid).to(tl.int32)

    start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
    end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

    offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = (offs_m < end_x) & (offs_m < n)

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    if D == 1:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = x0 * x0
    elif D == 2:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x1v, x1v, x0 * x0)
    else:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2v = tl.load(
            x_ptr + offs_m * stride_x0 + 2 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x2v, x2v, tl.fma(x1v, x1v, x0 * x0))

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    row_start = tl.load(row_ptr_x_ptr + pid).to(tl.int32)
    row_end = tl.load(row_ptr_x_ptr + pid + 1).to(tl.int32)

    for t in range(0, MAX_TASKS_X):
        idx = row_start + t
        has = idx < row_end
        y_cluster = tl.load(nbr_y_cluster_ptr + idx, mask=has, other=0).to(tl.int32)
        y_block = tl.load(nbr_y_block_ptr + idx, mask=has, other=0).to(tl.int32)

        start_y = tl.load(offsets_y_ptr + y_cluster, mask=has, other=0).to(tl.int32)
        end_y = tl.load(offsets_y_ptr + y_cluster + 1, mask=has, other=0).to(tl.int32)

        offs_n = start_y + y_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = has & (offs_n < end_y) & (offs_n < m)
        mask_mn = mask_m[:, None] & mask_n[None, :]

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        vec = tl.load(vec_ptr + offs_n * stride_vec, mask=mask_n, other=0.0).to(
            tl.float32
        )

        if D == 1:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = y0 * y0
            dot = x0[:, None] * y0[None, :]
        elif D == 2:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y1w, y1w, y0 * y0)
            dot = x0[:, None] * y0[None, :] + x1v[:, None] * y1w[None, :]
        else:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2w = tl.load(
                y_ptr + offs_n * stride_y0 + 2 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y2w, y2w, tl.fma(y1w, y1w, y0 * y0))
            dot = (
                x0[:, None] * y0[None, :]
                + x1v[:, None] * y1w[None, :]
                + x2v[:, None] * y2w[None, :]
            )

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
            w = tl.where(new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None]))

        w = tl.where(mask_mn, w, 0.0)
        s_i = s_i * alpha + tl.sum(w * vec[None, :], axis=1)
        m_i = new_m

    if USE_EXP2:
        out = tl.exp2(m_i) * s_i
    else:
        out = tl.exp(m_i) * s_i
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


@triton.jit
def _apply_plan_axis0_vec_taskcsr_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    vec_ptr,
    out_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    prog_y_cluster_ptr,
    prog_y_block_ptr,
    row_ptr_y_ptr,
    nbr_x_cluster_ptr,
    nbr_x_block_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_vec,
    stride_out,
    eps,
    MAX_TASKS_Y: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    cluster = tl.load(prog_y_cluster_ptr + pid).to(tl.int32)
    blk = tl.load(prog_y_block_ptr + pid).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cluster + 1).to(tl.int32)

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < end_y) & (offs_n < m)

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    if D == 1:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = y0 * y0
    elif D == 2:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y1v = tl.load(
            y_ptr + offs_n * stride_y0 + 1 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = tl.fma(y1v, y1v, y0 * y0)
    else:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y1v = tl.load(
            y_ptr + offs_n * stride_y0 + 1 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2v = tl.load(
            y_ptr + offs_n * stride_y0 + 2 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = tl.fma(y2v, y2v, tl.fma(y1v, y1v, y0 * y0))

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    row_start = tl.load(row_ptr_y_ptr + pid).to(tl.int32)
    row_end = tl.load(row_ptr_y_ptr + pid + 1).to(tl.int32)

    for t in range(0, MAX_TASKS_Y):
        idx = row_start + t
        has = idx < row_end
        x_cluster = tl.load(nbr_x_cluster_ptr + idx, mask=has, other=0).to(tl.int32)
        x_block = tl.load(nbr_x_block_ptr + idx, mask=has, other=0).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + x_cluster, mask=has, other=0).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + x_cluster + 1, mask=has, other=0).to(tl.int32)

        offs_m = start_x + x_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = has & (offs_m < end_x) & (offs_m < n)
        mask_mn = mask_m[:, None] & mask_n[None, :]

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        vec = tl.load(vec_ptr + offs_m * stride_vec, mask=mask_m, other=0.0).to(
            tl.float32
        )

        if D == 1:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = x0 * x0
            dot = x0[:, None] * y0[None, :]
        elif D == 2:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x1w = tl.load(
                x_ptr + offs_m * stride_x0 + 1 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.fma(x1w, x1w, x0 * x0)
            dot = x0[:, None] * y0[None, :] + x1w[:, None] * y1v[None, :]
        else:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x1w = tl.load(
                x_ptr + offs_m * stride_x0 + 1 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2w = tl.load(
                x_ptr + offs_m * stride_x0 + 2 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.fma(x2w, x2w, tl.fma(x1w, x1w, x0 * x0))
            dot = (
                x0[:, None] * y0[None, :]
                + x1w[:, None] * y1v[None, :]
                + x2w[:, None] * y2v[None, :]
            )

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
            w = tl.where(new_m_neg_inf[None, :], 0.0, tl.exp(logits - new_m[None, :]))

        w = tl.where(mask_mn, w, 0.0)
        s_j = s_j * alpha + tl.sum(w * vec[:, None], axis=0)
        m_j = new_m

    if USE_EXP2:
        out = tl.exp2(m_j) * s_j
    else:
        out = tl.exp(m_j) * s_j
    tl.store(out_ptr + offs_n * stride_out, out, mask=mask_n)


@triton.jit
def _apply_plan_axis1_mat_taskcsr_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    mat_ptr,
    scale_ptr,
    out_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    prog_x_cluster_ptr,
    prog_x_block_ptr,
    row_ptr_x_ptr,
    nbr_y_cluster_ptr,
    nbr_y_block_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_mat0,
    stride_mat1,
    stride_scale,
    stride_out0,
    stride_out1,
    eps,
    MAX_TASKS_X: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    cluster = tl.load(prog_x_cluster_ptr + pid).to(tl.int32)
    blk = tl.load(prog_x_block_ptr + pid).to(tl.int32)

    start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
    end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

    offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = (offs_m < end_x) & (offs_m < n)

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    if D == 1:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = x0 * x0
    elif D == 2:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x1v, x1v, x0 * x0)
    else:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2v = tl.load(
            x_ptr + offs_m * stride_x0 + 2 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x2v, x2v, tl.fma(x1v, x1v, x0 * x0))

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o0 = tl.zeros([BLOCK_M], tl.float32)
    o1 = tl.zeros([BLOCK_M], tl.float32)
    o2 = tl.zeros([BLOCK_M], tl.float32)

    row_start = tl.load(row_ptr_x_ptr + pid).to(tl.int32)
    row_end = tl.load(row_ptr_x_ptr + pid + 1).to(tl.int32)

    for t in range(0, MAX_TASKS_X):
        idx = row_start + t
        has = idx < row_end
        y_cluster = tl.load(nbr_y_cluster_ptr + idx, mask=has, other=0).to(tl.int32)
        y_block = tl.load(nbr_y_block_ptr + idx, mask=has, other=0).to(tl.int32)

        start_y = tl.load(offsets_y_ptr + y_cluster, mask=has, other=0).to(tl.int32)
        end_y = tl.load(offsets_y_ptr + y_cluster + 1, mask=has, other=0).to(tl.int32)

        offs_n = start_y + y_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = has & (offs_n < end_y) & (offs_n < m)
        mask_mn = mask_m[:, None] & mask_n[None, :]

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)

        if D == 1:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = y0 * y0
            dot = x0[:, None] * y0[None, :]
        elif D == 2:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y1w, y1w, y0 * y0)
            dot = x0[:, None] * y0[None, :] + x1v[:, None] * y1w[None, :]
        else:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2w = tl.load(
                y_ptr + offs_n * stride_y0 + 2 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y2w, y2w, tl.fma(y1w, y1w, y0 * y0))
            dot = (
                x0[:, None] * y0[None, :]
                + x1v[:, None] * y1w[None, :]
                + x2v[:, None] * y2w[None, :]
            )

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
            w = tl.where(new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None]))

        w = tl.where(mask_mn, w, 0.0)

        if D >= 1:
            mat0 = tl.load(
                mat_ptr + offs_n * stride_mat0 + 0 * stride_mat1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            if HAS_SCALE:
                sc = tl.load(scale_ptr + offs_n * stride_scale, mask=mask_n, other=0.0)
                mat0 = mat0 * sc.to(tl.float32)
            tmp0 = tl.sum(w * mat0[None, :], axis=1)
            o0 = o0 * alpha + tmp0
        if D >= 2:
            mat1 = tl.load(
                mat_ptr + offs_n * stride_mat0 + 1 * stride_mat1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            if HAS_SCALE:
                sc = tl.load(scale_ptr + offs_n * stride_scale, mask=mask_n, other=0.0)
                mat1 = mat1 * sc.to(tl.float32)
            tmp1 = tl.sum(w * mat1[None, :], axis=1)
            o1 = o1 * alpha + tmp1
        if D == 3:
            mat2 = tl.load(
                mat_ptr + offs_n * stride_mat0 + 2 * stride_mat1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            if HAS_SCALE:
                sc = tl.load(scale_ptr + offs_n * stride_scale, mask=mask_n, other=0.0)
                mat2 = mat2 * sc.to(tl.float32)
            tmp2 = tl.sum(w * mat2[None, :], axis=1)
            o2 = o2 * alpha + tmp2

        m_i = new_m

    if USE_EXP2:
        scale_m = tl.exp2(m_i)
    else:
        scale_m = tl.exp(m_i)
    if D >= 1:
        tl.store(
            out_ptr + offs_m * stride_out0 + 0 * stride_out1,
            scale_m * o0,
            mask=mask_m,
        )
    if D >= 2:
        tl.store(
            out_ptr + offs_m * stride_out0 + 1 * stride_out1,
            scale_m * o1,
            mask=mask_m,
        )
    if D == 3:
        tl.store(
            out_ptr + offs_m * stride_out0 + 2 * stride_out1,
            scale_m * o2,
            mask=mask_m,
        )


@triton.jit
def _apply_plan_axis0_mat_taskcsr_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    mat_ptr,
    out_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    prog_y_cluster_ptr,
    prog_y_block_ptr,
    row_ptr_y_ptr,
    nbr_x_cluster_ptr,
    nbr_x_block_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_mat0,
    stride_mat1,
    stride_out0,
    stride_out1,
    eps,
    MAX_TASKS_Y: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    cluster = tl.load(prog_y_cluster_ptr + pid).to(tl.int32)
    blk = tl.load(prog_y_block_ptr + pid).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cluster + 1).to(tl.int32)

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < end_y) & (offs_n < m)

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    if D == 1:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = y0 * y0
    elif D == 2:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y1v = tl.load(
            y_ptr + offs_n * stride_y0 + 1 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = tl.fma(y1v, y1v, y0 * y0)
    else:
        y0 = tl.load(
            y_ptr + offs_n * stride_y0 + 0 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y1v = tl.load(
            y_ptr + offs_n * stride_y0 + 1 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2v = tl.load(
            y_ptr + offs_n * stride_y0 + 2 * stride_y1,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        y2 = tl.fma(y2v, y2v, tl.fma(y1v, y1v, y0 * y0))

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    o0 = tl.zeros([BLOCK_N], tl.float32)
    o1 = tl.zeros([BLOCK_N], tl.float32)
    o2 = tl.zeros([BLOCK_N], tl.float32)

    row_start = tl.load(row_ptr_y_ptr + pid).to(tl.int32)
    row_end = tl.load(row_ptr_y_ptr + pid + 1).to(tl.int32)

    for t in range(0, MAX_TASKS_Y):
        idx = row_start + t
        has = idx < row_end
        x_cluster = tl.load(nbr_x_cluster_ptr + idx, mask=has, other=0).to(tl.int32)
        x_block = tl.load(nbr_x_block_ptr + idx, mask=has, other=0).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + x_cluster, mask=has, other=0).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + x_cluster + 1, mask=has, other=0).to(tl.int32)

        offs_m = start_x + x_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = has & (offs_m < end_x) & (offs_m < n)
        mask_mn = mask_m[:, None] & mask_n[None, :]

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)

        if D == 1:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = x0 * x0
            dot = x0[:, None] * y0[None, :]
        elif D == 2:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x1w = tl.load(
                x_ptr + offs_m * stride_x0 + 1 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.fma(x1w, x1w, x0 * x0)
            dot = x0[:, None] * y0[None, :] + x1w[:, None] * y1v[None, :]
        else:
            x0 = tl.load(
                x_ptr + offs_m * stride_x0 + 0 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x1w = tl.load(
                x_ptr + offs_m * stride_x0 + 1 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2w = tl.load(
                x_ptr + offs_m * stride_x0 + 2 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.fma(x2w, x2w, tl.fma(x1w, x1w, x0 * x0))
            dot = (
                x0[:, None] * y0[None, :]
                + x1w[:, None] * y1v[None, :]
                + x2w[:, None] * y2v[None, :]
            )

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
            w = tl.where(new_m_neg_inf[None, :], 0.0, tl.exp(logits - new_m[None, :]))

        w = tl.where(mask_mn, w, 0.0)

        if D >= 1:
            mat0 = tl.load(
                mat_ptr + offs_m * stride_mat0 + 0 * stride_mat1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            tmp0 = tl.sum(w * mat0[:, None], axis=0)
            o0 = o0 * alpha + tmp0
        if D >= 2:
            mat1 = tl.load(
                mat_ptr + offs_m * stride_mat0 + 1 * stride_mat1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            tmp1 = tl.sum(w * mat1[:, None], axis=0)
            o1 = o1 * alpha + tmp1
        if D == 3:
            mat2 = tl.load(
                mat_ptr + offs_m * stride_mat0 + 2 * stride_mat1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            tmp2 = tl.sum(w * mat2[:, None], axis=0)
            o2 = o2 * alpha + tmp2

        m_j = new_m

    if USE_EXP2:
        scale_m = tl.exp2(m_j)
    else:
        scale_m = tl.exp(m_j)

    if D >= 1:
        tl.store(
            out_ptr + offs_n * stride_out0 + 0 * stride_out1,
            scale_m * o0,
            mask=mask_n,
        )
    if D >= 2:
        tl.store(
            out_ptr + offs_n * stride_out0 + 1 * stride_out1,
            scale_m * o1,
            mask=mask_n,
        )
    if D == 3:
        tl.store(
            out_ptr + offs_n * stride_out0 + 2 * stride_out1,
            scale_m * o2,
            mask=mask_n,
        )


@triton.jit
def _mat5_axis1_taskcsr_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    A_ptr,
    out_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    prog_x_cluster_ptr,
    prog_x_block_ptr,
    row_ptr_x_ptr,
    nbr_y_cluster_ptr,
    nbr_y_block_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_A0,
    stride_A1,
    stride_out0,
    stride_out1,
    eps,
    scale,
    MAX_TASKS_X: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    cluster = tl.load(prog_x_cluster_ptr + pid).to(tl.int32)
    blk = tl.load(prog_x_block_ptr + pid).to(tl.int32)

    start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
    end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

    offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = (offs_m < end_x) & (offs_m < n)

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)

    if D == 1:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = x0 * x0
        A0 = tl.load(
            A_ptr + offs_m * stride_A0 + 0 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
    elif D == 2:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x1v, x1v, x0 * x0)
        A0 = tl.load(
            A_ptr + offs_m * stride_A0 + 0 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        A1 = tl.load(
            A_ptr + offs_m * stride_A0 + 1 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
    else:
        x0 = tl.load(
            x_ptr + offs_m * stride_x0 + 0 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x1v = tl.load(
            x_ptr + offs_m * stride_x0 + 1 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2v = tl.load(
            x_ptr + offs_m * stride_x0 + 2 * stride_x1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.fma(x2v, x2v, tl.fma(x1v, x1v, x0 * x0))
        A0 = tl.load(
            A_ptr + offs_m * stride_A0 + 0 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        A1 = tl.load(
            A_ptr + offs_m * stride_A0 + 1 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        A2 = tl.load(
            A_ptr + offs_m * stride_A0 + 2 * stride_A1,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o0 = tl.zeros([BLOCK_M], tl.float32)
    o1 = tl.zeros([BLOCK_M], tl.float32)
    o2 = tl.zeros([BLOCK_M], tl.float32)

    row_start = tl.load(row_ptr_x_ptr + pid).to(tl.int32)
    row_end = tl.load(row_ptr_x_ptr + pid + 1).to(tl.int32)

    for t in range(0, MAX_TASKS_X):
        idx = row_start + t
        has = idx < row_end
        y_cluster = tl.load(nbr_y_cluster_ptr + idx, mask=has, other=0).to(tl.int32)
        y_block = tl.load(nbr_y_block_ptr + idx, mask=has, other=0).to(tl.int32)

        start_y = tl.load(offsets_y_ptr + y_cluster, mask=has, other=0).to(tl.int32)
        end_y = tl.load(offsets_y_ptr + y_cluster + 1, mask=has, other=0).to(tl.int32)

        offs_n = start_y + y_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = has & (offs_n < end_y) & (offs_n < m)
        mask_mn = mask_m[:, None] & mask_n[None, :]

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)

        if D == 1:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = y0 * y0
            dot_xy = x0[:, None] * y0[None, :]
            dot_ay = A0[:, None] * y0[None, :]
        elif D == 2:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y1w, y1w, y0 * y0)
            dot_xy = x0[:, None] * y0[None, :] + x1v[:, None] * y1w[None, :]
            dot_ay = A0[:, None] * y0[None, :] + A1[:, None] * y1w[None, :]
        else:
            y0 = tl.load(
                y_ptr + offs_n * stride_y0 + 0 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y1w = tl.load(
                y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2w = tl.load(
                y_ptr + offs_n * stride_y0 + 2 * stride_y1,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            y2 = tl.fma(y2w, y2w, tl.fma(y1w, y1w, y0 * y0))
            dot_xy = (
                x0[:, None] * y0[None, :]
                + x1v[:, None] * y1w[None, :]
                + x2v[:, None] * y2w[None, :]
            )
            dot_ay = (
                A0[:, None] * y0[None, :]
                + A1[:, None] * y1w[None, :]
                + A2[:, None] * y2w[None, :]
            )

        cost = x2[:, None] + y2[None, :] - 2.0 * dot_xy
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
            w = tl.where(new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None]))

        w = tl.where(mask_mn, w, 0.0)
        w_dot = w * dot_ay

        if D >= 1:
            tmp0 = tl.sum(w_dot * y0[None, :], axis=1)
            o0 = o0 * alpha + tmp0
        if D >= 2:
            tmp1 = tl.sum(w_dot * y1w[None, :], axis=1)
            o1 = o1 * alpha + tmp1
        if D == 3:
            tmp2 = tl.sum(w_dot * y2w[None, :], axis=1)
            o2 = o2 * alpha + tmp2

        m_i = new_m

    if USE_EXP2:
        scale_m = tl.exp2(m_i) * scale
    else:
        scale_m = tl.exp(m_i) * scale

    if D >= 1:
        tl.store(
            out_ptr + offs_m * stride_out0 + 0 * stride_out1,
            scale_m * o0,
            mask=mask_m,
        )
    if D >= 2:
        tl.store(
            out_ptr + offs_m * stride_out0 + 1 * stride_out1,
            scale_m * o1,
            mask=mask_m,
        )
    if D == 3:
        tl.store(
            out_ptr + offs_m * stride_out0 + 2 * stride_out1,
            scale_m * o2,
            mask=mask_m,
        )


def apply_plan_vec_sqeuclid_taskcsr(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    vec: torch.Tensor,
    *,
    eps: float,
    axis: int,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    taskcsr_x: BlockSparseTaskCSR,
    taskcsr_y: BlockSparseTaskCSR,
    buckets: Optional[BlockSparseTaskCSRBuckets] = None,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    use_exp2: bool,
) -> torch.Tensor:
    if axis == 1:
        out = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float32)
        if buckets is None:
            n_prog = int(taskcsr_x.prog_cluster.numel())
            if n_prog == 0:
                return out
            grid = (n_prog,)
            _apply_plan_axis1_vec_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                vec,
                out,
                offsets_x,
                offsets_y,
                taskcsr_x.prog_cluster,
                taskcsr_x.prog_block,
                taskcsr_x.row_ptr,
                taskcsr_x.nbr_cluster,
                taskcsr_x.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                vec.stride(0),
                out.stride(0),
                float(eps),
                MAX_TASKS_X=int(taskcsr_x.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
            return out

        for bx, _ in buckets.buckets:
            n_prog = int(bx.prog_cluster.numel())
            if n_prog == 0:
                continue
            grid = (n_prog,)
            _apply_plan_axis1_vec_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                vec,
                out,
                offsets_x,
                offsets_y,
                bx.prog_cluster,
                bx.prog_block,
                bx.row_ptr,
                bx.nbr_cluster,
                bx.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                vec.stride(0),
                out.stride(0),
                float(eps),
                MAX_TASKS_X=int(bx.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
        return out

    if axis == 0:
        out = torch.zeros((y.shape[0],), device=x.device, dtype=torch.float32)
        if buckets is None:
            n_prog = int(taskcsr_y.prog_cluster.numel())
            if n_prog == 0:
                return out
            grid = (n_prog,)
            _apply_plan_axis0_vec_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                vec,
                out,
                offsets_x,
                offsets_y,
                taskcsr_y.prog_cluster,
                taskcsr_y.prog_block,
                taskcsr_y.row_ptr,
                taskcsr_y.nbr_cluster,
                taskcsr_y.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                vec.stride(0),
                out.stride(0),
                float(eps),
                MAX_TASKS_Y=int(taskcsr_y.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
            return out

        for _, by in buckets.buckets:
            n_prog = int(by.prog_cluster.numel())
            if n_prog == 0:
                continue
            grid = (n_prog,)
            _apply_plan_axis0_vec_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                vec,
                out,
                offsets_x,
                offsets_y,
                by.prog_cluster,
                by.prog_block,
                by.row_ptr,
                by.nbr_cluster,
                by.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                vec.stride(0),
                out.stride(0),
                float(eps),
                MAX_TASKS_Y=int(by.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
        return out

    raise ValueError("axis must be 0 or 1.")


def apply_plan_mat_sqeuclid_taskcsr(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    mat: torch.Tensor,
    *,
    eps: float,
    axis: int,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    taskcsr_x: BlockSparseTaskCSR,
    taskcsr_y: BlockSparseTaskCSR,
    buckets: Optional[BlockSparseTaskCSRBuckets] = None,
    scale: Optional[torch.Tensor] = None,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    use_exp2: bool,
) -> torch.Tensor:
    if int(x.shape[1]) not in (1, 2, 3):
        raise ValueError("Blocksparse apply only supports D in {1,2,3}.")
    if mat.ndim != 2:
        raise ValueError("mat must be 2D.")

    if axis == 1:
        if mat.shape != y.shape:
            raise ValueError("axis=1 expects mat shaped (m,d) matching y.")
        if scale is not None and scale.shape != (y.shape[0],):
            raise ValueError("scale must have shape (m,).")

        out = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
        has_scale = scale is not None
        scale_t = scale if scale is not None else y

        if buckets is None:
            n_prog = int(taskcsr_x.prog_cluster.numel())
            if n_prog == 0:
                return out
            grid = (n_prog,)
            _apply_plan_axis1_mat_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                mat,
                scale_t,
                out,
                offsets_x,
                offsets_y,
                taskcsr_x.prog_cluster,
                taskcsr_x.prog_block,
                taskcsr_x.row_ptr,
                taskcsr_x.nbr_cluster,
                taskcsr_x.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                mat.stride(0),
                mat.stride(1),
                scale_t.stride(0),
                out.stride(0),
                out.stride(1),
                float(eps),
                MAX_TASKS_X=int(taskcsr_x.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                HAS_SCALE=bool(has_scale),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
            return out

        for bx, _ in buckets.buckets:
            n_prog = int(bx.prog_cluster.numel())
            if n_prog == 0:
                continue
            grid = (n_prog,)
            _apply_plan_axis1_mat_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                mat,
                scale_t,
                out,
                offsets_x,
                offsets_y,
                bx.prog_cluster,
                bx.prog_block,
                bx.row_ptr,
                bx.nbr_cluster,
                bx.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                mat.stride(0),
                mat.stride(1),
                scale_t.stride(0),
                out.stride(0),
                out.stride(1),
                float(eps),
                MAX_TASKS_X=int(bx.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                HAS_SCALE=bool(has_scale),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
        return out

    if axis == 0:
        if mat.shape != x.shape:
            raise ValueError("axis=0 expects mat shaped (n,d) matching x.")
        out = torch.zeros((y.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
        if buckets is None:
            n_prog = int(taskcsr_y.prog_cluster.numel())
            if n_prog == 0:
                return out
            grid = (n_prog,)
            _apply_plan_axis0_mat_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                mat,
                out,
                offsets_x,
                offsets_y,
                taskcsr_y.prog_cluster,
                taskcsr_y.prog_block,
                taskcsr_y.row_ptr,
                taskcsr_y.nbr_cluster,
                taskcsr_y.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                mat.stride(0),
                mat.stride(1),
                out.stride(0),
                out.stride(1),
                float(eps),
                MAX_TASKS_Y=int(taskcsr_y.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
            return out

        for _, by in buckets.buckets:
            n_prog = int(by.prog_cluster.numel())
            if n_prog == 0:
                continue
            grid = (n_prog,)
            _apply_plan_axis0_mat_taskcsr_kernel[grid](
                x,
                y,
                f,
                g,
                mat,
                out,
                offsets_x,
                offsets_y,
                by.prog_cluster,
                by.prog_block,
                by.row_ptr,
                by.nbr_cluster,
                by.nbr_block,
                x.shape[0],
                y.shape[0],
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                mat.stride(0),
                mat.stride(1),
                out.stride(0),
                out.stride(1),
                float(eps),
                MAX_TASKS_Y=int(by.max_tasks),
                D=int(x.shape[1]),
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                USE_EXP2=bool(use_exp2),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )
        return out

    raise ValueError("axis must be 0 or 1.")


def mat5_sqeuclid_taskcsr(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    taskcsr_x: BlockSparseTaskCSR,
    buckets: Optional[BlockSparseTaskCSRBuckets] = None,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    use_exp2: bool,
) -> torch.Tensor:
    if int(x.shape[1]) not in (1, 2, 3):
        raise ValueError("Blocksparse Mat5 only supports D in {1,2,3}.")
    if A.shape != x.shape:
        raise ValueError("A must have the same shape as x.")

    out = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
    n_prog = int(taskcsr_x.prog_cluster.numel())
    if n_prog == 0:
        return out

    scale = -4.0 / float(eps)
    if buckets is None:
        grid = (n_prog,)
        _mat5_axis1_taskcsr_kernel[grid](
            x,
            y,
            f,
            g,
            A,
            out,
            offsets_x,
            offsets_y,
            taskcsr_x.prog_cluster,
            taskcsr_x.prog_block,
            taskcsr_x.row_ptr,
            taskcsr_x.nbr_cluster,
            taskcsr_x.nbr_block,
            x.shape[0],
            y.shape[0],
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            A.stride(0),
            A.stride(1),
            out.stride(0),
            out.stride(1),
            float(eps),
            float(scale),
            MAX_TASKS_X=int(taskcsr_x.max_tasks),
            D=int(x.shape[1]),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            USE_EXP2=bool(use_exp2),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
        return out

    for bx, _ in buckets.buckets:
        n_prog = int(bx.prog_cluster.numel())
        if n_prog == 0:
            continue
        grid = (n_prog,)
        _mat5_axis1_taskcsr_kernel[grid](
            x,
            y,
            f,
            g,
            A,
            out,
            offsets_x,
            offsets_y,
            bx.prog_cluster,
            bx.prog_block,
            bx.row_ptr,
            bx.nbr_cluster,
            bx.nbr_block,
            x.shape[0],
            y.shape[0],
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            A.stride(0),
            A.stride(1),
            out.stride(0),
            out.stride(1),
            float(eps),
            float(scale),
            MAX_TASKS_X=int(bx.max_tasks),
            D=int(x.shape[1]),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            USE_EXP2=bool(use_exp2),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    return out
