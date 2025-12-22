from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


def _pid_maps_from_offsets(
    offsets: torch.Tensor, *, block: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = offsets.device
    num_clusters = int(offsets.numel() - 1)
    if num_clusters <= 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty, 0

    counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    blocks = (counts + int(block) - 1) // int(block)
    blocks = torch.clamp(blocks, min=0)
    total = int(blocks.sum().item())
    if total <= 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty, int(blocks.max().item()) if blocks.numel() else 0

    max_blocks = int(blocks.max().item())

    clusters = torch.arange(num_clusters, device=device, dtype=torch.int32)
    pid_cluster = torch.repeat_interleave(clusters, blocks.to(torch.int64))

    starts = torch.cumsum(blocks, dim=0) - blocks
    starts_rep = torch.repeat_interleave(starts, blocks.to(torch.int64))
    pid_block = (torch.arange(total, device=device, dtype=torch.int64) - starts_rep).to(
        torch.int32
    )
    return pid_cluster.to(torch.int32), pid_block, max_blocks


def blocksparse_prepare_metadata(
    *,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    row_ptr_y: torch.Tensor,
    block_m: int,
    block_n: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
]:
    pid_x_cluster, pid_x_block, max_blocks_x = _pid_maps_from_offsets(
        offsets_x, block=int(block_m)
    )
    pid_y_cluster, pid_y_block, max_blocks_y = _pid_maps_from_offsets(
        offsets_y, block=int(block_n)
    )

    # Keep everything on the same device (GPU in the common case).
    pid_x_cluster = pid_x_cluster.to(device=offsets_x.device)
    pid_x_block = pid_x_block.to(device=offsets_x.device)
    pid_y_cluster = pid_y_cluster.to(device=offsets_x.device)
    pid_y_block = pid_y_block.to(device=offsets_x.device)

    deg_x = (row_ptr_x[1:] - row_ptr_x[:-1]).to(torch.int64)
    deg_y = (row_ptr_y[1:] - row_ptr_y[:-1]).to(torch.int64)
    max_deg_x = int(deg_x.max().item()) if deg_x.numel() else 0
    max_deg_y = int(deg_y.max().item()) if deg_y.numel() else 0

    return (
        pid_x_cluster,
        pid_x_block,
        pid_y_cluster,
        pid_y_block,
        int(max_blocks_x),
        int(max_blocks_y),
        int(max_deg_x),
        int(max_deg_y),
    )


@triton.jit
def _geomloss_symmetric_step_sqeuclid_blocksparse_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    x2_ptr,
    y2_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    pid_x_cluster_ptr,
    pid_x_block_ptr,
    pid_y_cluster_ptr,
    pid_y_block_ptr,
    row_ptr_x_ptr,
    col_idx_x_ptr,
    row_ptr_y_ptr,
    col_idx_y_ptr,
    f_out_ptr,
    g_out_ptr,
    n,
    m,
    n_prog_x,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_loga,
    stride_logb,
    stride_x2,
    stride_y2,
    stride_f_out,
    stride_g_out,
    eps,
    alpha,
    max_deg_x,
    max_deg_y,
    max_blocks_x,
    max_blocks_y,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps_log2 = inv_eps * log2e

    # ------------------------------------------------------------
    # f update: one program per (x_cluster, block_in_cluster)
    # ------------------------------------------------------------
    if pid < n_prog_x:
        cluster = tl.load(pid_x_cluster_ptr + pid).to(tl.int32)
        blk = tl.load(pid_x_block_ptr + pid).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

        offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_x
        mask_m = mask_m & (offs_m < n)

        f_old = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        row_start = tl.load(row_ptr_x_ptr + cluster).to(tl.int32)
        row_end = tl.load(row_ptr_x_ptr + cluster + 1).to(tl.int32)

        for t in range(0, max_deg_x):
            idx = row_start + t
            has_nbr = idx < row_end
            nbr = tl.load(col_idx_x_ptr + idx, mask=has_nbr, other=0).to(tl.int32)

            start_y = tl.load(offsets_y_ptr + nbr, mask=has_nbr, other=0).to(tl.int32)
            end_y = tl.load(offsets_y_ptr + nbr + 1, mask=has_nbr, other=0).to(tl.int32)

            for by in range(0, max_blocks_y):
                offs_n = start_y + by * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < end_y
                mask_n = mask_n & (offs_n < m)
                mask_n = mask_n & has_nbr

                g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
                logb = tl.load(
                    logb_ptr + offs_n * stride_logb,
                    mask=mask_n,
                    other=-float("inf"),
                ).to(tl.float32)
                if USE_EXP2:
                    logb = logb * log2e
                y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

                dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for k0 in range(0, D, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < D
                    xk = tl.load(
                        x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    yk = tl.load(
                        y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                        mask=mask_n[None, :] & mask_k[:, None],
                        other=0.0,
                    )
                    dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

                cost = x2[:, None] + y2[None, :] - 2.0 * dot
                if USE_EXP2:
                    vals = tl.fma(g[None, :] - cost, inv_eps_log2, logb[None, :])
                else:
                    vals = (g[None, :] - cost) * inv_eps + logb[None, :]
                vals = tl.where(mask_m[:, None] & mask_n[None, :], vals, -float("inf"))

                block_max = tl.max(vals, axis=1)
                new_m = tl.maximum(m_i, block_max)
                new_m_neg_inf = new_m == -float("inf")
                if USE_EXP2:
                    alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_i - new_m))
                    w = tl.where(
                        new_m_neg_inf[:, None], 0.0, tl.exp2(vals - new_m[:, None])
                    )
                else:
                    alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp(m_i - new_m))
                    w = tl.where(
                        new_m_neg_inf[:, None], 0.0, tl.exp(vals - new_m[:, None])
                    )

                s_i = s_i * alpha_m + tl.sum(w, axis=1)
                m_i = new_m

        valid = (m_i != -float("inf")) & (s_i > 0)
        lse = (m_i + tl.log2(s_i)) * ln2 if USE_EXP2 else (m_i + tl.log(s_i))
        cand = -eps * lse
        cand = tl.where(valid, cand, f_old)
        f_new = (1.0 - alpha) * f_old + alpha * cand
        tl.store(f_out_ptr + offs_m * stride_f_out, f_new, mask=mask_m)
        return

    # ------------------------------------------------------------
    # g update: one program per (y_cluster, block_in_cluster)
    # ------------------------------------------------------------
    pid_y = pid - n_prog_x
    cluster = tl.load(pid_y_cluster_ptr + pid_y).to(tl.int32)
    blk = tl.load(pid_y_block_ptr + pid_y).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cluster + 1).to(tl.int32)

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < end_y
    mask_n = mask_n & (offs_n < m)

    g_old = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    row_start = tl.load(row_ptr_y_ptr + cluster).to(tl.int32)
    row_end = tl.load(row_ptr_y_ptr + cluster + 1).to(tl.int32)

    for t in range(0, max_deg_y):
        idx = row_start + t
        has_nbr = idx < row_end
        nbr = tl.load(col_idx_y_ptr + idx, mask=has_nbr, other=0).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + nbr, mask=has_nbr, other=0).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + nbr + 1, mask=has_nbr, other=0).to(tl.int32)

        for bx in range(0, max_blocks_x):
            offs_m = start_x + bx * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < end_x
            mask_m = mask_m & (offs_m < n)
            mask_m = mask_m & has_nbr

            f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
            loga = tl.load(
                loga_ptr + offs_m * stride_loga,
                mask=mask_m,
                other=-float("inf"),
            ).to(tl.float32)
            if USE_EXP2:
                loga = loga * log2e
            x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                xk = tl.load(
                    x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                yk = tl.load(
                    y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                    mask=mask_n[None, :] & mask_k[:, None],
                    other=0.0,
                )
                dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

            cost = x2[:, None] + y2[None, :] - 2.0 * dot
            if USE_EXP2:
                vals = tl.fma(f[:, None] - cost, inv_eps_log2, loga[:, None])
            else:
                vals = (f[:, None] - cost) * inv_eps + loga[:, None]
            vals = tl.where(mask_m[:, None], vals, -float("inf"))

            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            new_m_neg_inf = new_m == -float("inf")
            if USE_EXP2:
                alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
                w = tl.where(
                    new_m_neg_inf[None, :], 0.0, tl.exp2(vals - new_m[None, :])
                )
            else:
                alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
                w = tl.where(
                    new_m_neg_inf[None, :], 0.0, tl.exp(vals - new_m[None, :])
                )

            s_j = s_j * alpha_m + tl.sum(w, axis=0)
            m_j = new_m

    valid = (m_j != -float("inf")) & (s_j > 0)
    lse = (m_j + tl.log2(s_j)) * ln2 if USE_EXP2 else (m_j + tl.log(s_j))
    cand = -eps * lse
    cand = tl.where(valid, cand, g_old)
    g_new = (1.0 - alpha) * g_old + alpha * cand
    tl.store(g_out_ptr + offs_n * stride_g_out, g_new, mask=mask_n)


@triton.jit
def _geomloss_grad_sqeuclid_blocksparse_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    a_ptr,
    b_ptr,
    x2_ptr,
    y2_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    pid_x_cluster_ptr,
    pid_x_block_ptr,
    pid_y_cluster_ptr,
    pid_y_block_ptr,
    row_ptr_x_ptr,
    col_idx_x_ptr,
    row_ptr_y_ptr,
    col_idx_y_ptr,
    grad_scale_ptr,
    grad_x_ptr,
    grad_y_ptr,
    n,
    m,
    n_prog_x,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_loga,
    stride_logb,
    stride_a,
    stride_b,
    stride_x2,
    stride_y2,
    stride_grad_x0,
    stride_grad_x1,
    stride_grad_y0,
    stride_grad_y1,
    eps,
    max_deg_x,
    max_deg_y,
    max_blocks_x,
    max_blocks_y,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps
    grad_scale = tl.load(grad_scale_ptr).to(tl.float32)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # ------------------------------------------------------------
    # grad_x programs
    # ------------------------------------------------------------
    if pid < n_prog_x:
        cluster = tl.load(pid_x_cluster_ptr + pid).to(tl.int32)
        blk = tl.load(pid_x_block_ptr + pid).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

        offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_x
        mask_m = mask_m & (offs_m < n)

        a = tl.load(a_ptr + offs_m * stride_a, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        row_start = tl.load(row_ptr_x_ptr + cluster).to(tl.int32)
        row_end = tl.load(row_ptr_x_ptr + cluster + 1).to(tl.int32)

        for t in range(0, max_deg_x):
            idx = row_start + t
            has_nbr = idx < row_end
            nbr = tl.load(col_idx_x_ptr + idx, mask=has_nbr, other=0).to(tl.int32)

            start_y = tl.load(offsets_y_ptr + nbr, mask=has_nbr, other=0).to(tl.int32)
            end_y = tl.load(offsets_y_ptr + nbr + 1, mask=has_nbr, other=0).to(tl.int32)

            for by in range(0, max_blocks_y):
                offs_n = start_y + by * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < end_y
                mask_n = mask_n & (offs_n < m)
                mask_n = mask_n & has_nbr

                mask_mn = mask_m[:, None] & mask_n[None, :]

                g = tl.load(
                    g_ptr + offs_n[None, :] * stride_g + offs_m[:, None] * 0,
                    mask=mask_mn,
                    other=0.0,
                ).to(tl.float32)
                logb = tl.load(
                    logb_ptr + offs_n[None, :] * stride_logb + offs_m[:, None] * 0,
                    mask=mask_mn,
                    other=-float("inf"),
                ).to(tl.float32)
                if USE_EXP2:
                    logb = logb * log2e
                y2 = tl.load(
                    y2_ptr + offs_n[None, :] * stride_y2 + offs_m[:, None] * 0,
                    mask=mask_mn,
                    other=0.0,
                ).to(tl.float32)

                dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for k0 in range(0, D, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < D
                    xk = tl.load(
                        x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    yk = tl.load(
                        y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                        mask=mask_n[None, :] & mask_k[:, None],
                        other=0.0,
                    )
                    dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

                cost = x2[:, None] + y2 - 2.0 * dot
                if USE_EXP2:
                    vals = tl.fma(g - cost, inv_eps_log2, logb)
                else:
                    vals = (g - cost) * inv_eps + logb
                vals = tl.where(mask_mn, vals, -float("inf"))

                block_max = tl.max(vals, axis=1)
                new_m = tl.maximum(m_i, block_max)
                if USE_EXP2:
                    alpha_m = tl.exp2(m_i - new_m)
                    w = tl.exp2(vals - new_m[:, None])
                else:
                    alpha_m = tl.exp(m_i - new_m)
                    w = tl.exp(vals - new_m[:, None])

                l_i = l_i * alpha_m + tl.sum(w, axis=1)
                o = o * alpha_m[:, None]

                yv_t = tl.load(
                    y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
                    mask=mask_d[:, None]
                    & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
                    other=0.0,
                ).to(tl.float32)
                yv = tl.trans(yv_t)
                o += tl.dot(w, yv, allow_tf32=ALLOW_TF32)
                m_i = new_m

        l_i_safe = tl.where(l_i > 1e-30, l_i, 1e-30)
        y_bar = o / l_i_safe[:, None]
        scale = (2.0 * grad_scale) * a
        grad = (x - y_bar) * scale[:, None]
        tl.store(
            grad_x_ptr
            + offs_m[:, None] * stride_grad_x0
            + offs_d[None, :] * stride_grad_x1,
            grad,
            mask=mask_m[:, None] & mask_d[None, :],
        )
        return

    # ------------------------------------------------------------
    # grad_y programs
    # ------------------------------------------------------------
    pid_y = pid - n_prog_x
    cluster = tl.load(pid_y_cluster_ptr + pid_y).to(tl.int32)
    blk = tl.load(pid_y_block_ptr + pid_y).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cluster + 1).to(tl.int32)

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < end_y
    mask_n = mask_n & (offs_n < m)

    b = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    y_t = tl.load(
        y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
        mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
        other=0.0,
    ).to(tl.float32)
    y = tl.trans(y_t)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    l_j = tl.zeros([BLOCK_N], tl.float32)
    o = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    row_start = tl.load(row_ptr_y_ptr + cluster).to(tl.int32)
    row_end = tl.load(row_ptr_y_ptr + cluster + 1).to(tl.int32)

    for t in range(0, max_deg_y):
        idx = row_start + t
        has_nbr = idx < row_end
        nbr = tl.load(col_idx_y_ptr + idx, mask=has_nbr, other=0).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + nbr, mask=has_nbr, other=0).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + nbr + 1, mask=has_nbr, other=0).to(tl.int32)

        for bx in range(0, max_blocks_x):
            offs_m = start_x + bx * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < end_x
            mask_m = mask_m & (offs_m < n)
            mask_m = mask_m & has_nbr

            mask_mn = mask_m[:, None] & mask_n[None, :]

            f = tl.load(
                f_ptr + offs_m[:, None] * stride_f + offs_n[None, :] * 0,
                mask=mask_mn,
                other=0.0,
            ).to(tl.float32)
            loga = tl.load(
                loga_ptr + offs_m[:, None] * stride_loga + offs_n[None, :] * 0,
                mask=mask_mn,
                other=-float("inf"),
            ).to(tl.float32)
            if USE_EXP2:
                loga = loga * log2e
            x2 = tl.load(
                x2_ptr + offs_m[:, None] * stride_x2 + offs_n[None, :] * 0,
                mask=mask_mn,
                other=0.0,
            ).to(tl.float32)

            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                xk = tl.load(
                    x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                yk = tl.load(
                    y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                    mask=mask_n[None, :] & mask_k[:, None],
                    other=0.0,
                )
                dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

            cost = x2 + y2[None, :] - 2.0 * dot
            if USE_EXP2:
                vals = tl.fma(f - cost, inv_eps_log2, loga)
            else:
                vals = (f - cost) * inv_eps + loga
            vals = tl.where(mask_mn, vals, -float("inf"))

            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            if USE_EXP2:
                alpha_m = tl.exp2(m_j - new_m)
                w = tl.exp2(vals - new_m[None, :])
            else:
                alpha_m = tl.exp(m_j - new_m)
                w = tl.exp(vals - new_m[None, :])

            l_j = l_j * alpha_m + tl.sum(w, axis=0)
            o = o * alpha_m[:, None]

            xv = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            o += tl.dot(tl.trans(w), xv, allow_tf32=ALLOW_TF32)
            m_j = new_m

    l_j_safe = tl.where(l_j > 1e-30, l_j, 1e-30)
    x_bar = o / l_j_safe[:, None]
    scale = (2.0 * grad_scale) * b
    grad = (y - x_bar) * scale[:, None]
    tl.store(
        grad_y_ptr + offs_n[:, None] * stride_grad_y0 + offs_d[None, :] * stride_grad_y1,
        grad,
        mask=mask_n[:, None] & mask_d[None, :],
    )


def geomloss_blocksparse_symmetric_step_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    *,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    col_idx_x: torch.Tensor,
    row_ptr_y: torch.Tensor,
    col_idx_y: torch.Tensor,
    pid_x_cluster: Optional[torch.Tensor] = None,
    pid_x_block: Optional[torch.Tensor] = None,
    pid_y_cluster: Optional[torch.Tensor] = None,
    pid_y_block: Optional[torch.Tensor] = None,
    max_blocks_x: Optional[int] = None,
    max_blocks_y: Optional[int] = None,
    max_deg_x: Optional[int] = None,
    max_deg_y: Optional[int] = None,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    eps: float,
    alpha: float,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    allow_tf32: bool,
    use_exp2: bool,
) -> None:
    if (
        pid_x_cluster is None
        or pid_x_block is None
        or pid_y_cluster is None
        or pid_y_block is None
        or max_blocks_x is None
        or max_blocks_y is None
        or max_deg_x is None
        or max_deg_y is None
    ):
        (
            pid_x_cluster,
            pid_x_block,
            pid_y_cluster,
            pid_y_block,
            max_blocks_x,
            max_blocks_y,
            max_deg_x,
            max_deg_y,
        ) = blocksparse_prepare_metadata(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            row_ptr_y=row_ptr_y,
            block_m=int(block_m),
            block_n=int(block_n),
        )

    n_prog_x = int(pid_x_cluster.numel())
    n_prog_y = int(pid_y_cluster.numel())
    grid = (n_prog_x + n_prog_y,)

    _geomloss_symmetric_step_sqeuclid_blocksparse_impl[grid](
        x,
        y,
        f,
        g,
        loga,
        logb,
        x2,
        y2,
        offsets_x,
        offsets_y,
        pid_x_cluster,
        pid_x_block,
        pid_y_cluster,
        pid_y_block,
        row_ptr_x,
        col_idx_x,
        row_ptr_y,
        col_idx_y,
        f_out,
        g_out,
        x.shape[0],
        y.shape[0],
        n_prog_x,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        f.stride(0),
        g.stride(0),
        loga.stride(0),
        logb.stride(0),
        x2.stride(0),
        y2.stride(0),
        f_out.stride(0),
        g_out.stride(0),
        float(eps),
        float(alpha),
        max_deg_x=int(max_deg_x),
        max_deg_y=int(max_deg_y),
        max_blocks_x=int(max_blocks_x),
        max_blocks_y=int(max_blocks_y),
        D=int(x.shape[1]),
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        USE_EXP2=bool(use_exp2),
        ALLOW_TF32=bool(allow_tf32),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )


def geomloss_blocksparse_grad_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    *,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    col_idx_x: torch.Tensor,
    row_ptr_y: torch.Tensor,
    col_idx_y: torch.Tensor,
    eps: float,
    grad_scale: torch.Tensor,
    pid_x_cluster: Optional[torch.Tensor] = None,
    pid_x_block: Optional[torch.Tensor] = None,
    pid_y_cluster: Optional[torch.Tensor] = None,
    pid_y_block: Optional[torch.Tensor] = None,
    max_blocks_x: Optional[int] = None,
    max_blocks_y: Optional[int] = None,
    max_deg_x: Optional[int] = None,
    max_deg_y: Optional[int] = None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    allow_tf32: bool,
    use_exp2: bool,
    compute_grad_x: bool,
    compute_grad_y: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if compute_grad_x:
        grad_x = torch.empty_like(x, dtype=torch.float32)
    else:
        grad_x = None
    if compute_grad_y:
        grad_y = torch.empty_like(y, dtype=torch.float32)
    else:
        grad_y = None

    if (
        pid_x_cluster is None
        or pid_x_block is None
        or pid_y_cluster is None
        or pid_y_block is None
        or max_blocks_x is None
        or max_blocks_y is None
        or max_deg_x is None
        or max_deg_y is None
    ):
        (
            pid_x_cluster,
            pid_x_block,
            pid_y_cluster,
            pid_y_block,
            max_blocks_x,
            max_blocks_y,
            max_deg_x,
            max_deg_y,
        ) = blocksparse_prepare_metadata(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            row_ptr_y=row_ptr_y,
            block_m=int(block_m),
            block_n=int(block_n),
        )

    n_prog_x = int(pid_x_cluster.numel()) if compute_grad_x else 0
    n_prog_y = int(pid_y_cluster.numel()) if compute_grad_y else 0
    grid = (n_prog_x + n_prog_y,)

    if n_prog_x + n_prog_y == 0:
        return None, None

    # Triton tl.dot requires non-batch dims >= 16. Pad feature dims accordingly.
    d = int(x.shape[1])
    block_d = max(16, 1 << (d - 1).bit_length())
    _geomloss_grad_sqeuclid_blocksparse_impl[grid](
        x,
        y,
        f,
        g,
        loga,
        logb,
        a,
        b,
        x2,
        y2,
        offsets_x,
        offsets_y,
        pid_x_cluster,
        pid_x_block,
        pid_y_cluster,
        pid_y_block,
        row_ptr_x,
        col_idx_x,
        row_ptr_y,
        col_idx_y,
        grad_scale,
        grad_x if grad_x is not None else x,
        grad_y if grad_y is not None else y,
        x.shape[0],
        y.shape[0],
        int(n_prog_x),
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        f.stride(0),
        g.stride(0),
        loga.stride(0),
        logb.stride(0),
        a.stride(0),
        b.stride(0),
        x2.stride(0),
        y2.stride(0),
        grad_x.stride(0) if grad_x is not None else x.stride(0),
        grad_x.stride(1) if grad_x is not None else x.stride(1),
        grad_y.stride(0) if grad_y is not None else y.stride(0),
        grad_y.stride(1) if grad_y is not None else y.stride(1),
        float(eps),
        max_deg_x=int(max_deg_x),
        max_deg_y=int(max_deg_y),
        max_blocks_x=int(max_blocks_x),
        max_blocks_y=int(max_blocks_y),
        D=int(x.shape[1]),
        BLOCK_D=int(block_d),
        ALLOW_TF32=bool(allow_tf32),
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        USE_EXP2=bool(use_exp2),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )

    return grad_x, grad_y
