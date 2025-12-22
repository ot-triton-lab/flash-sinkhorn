from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    BlockSparseTasks,
    blocksparse_build_tasks_from_csr,
)


@dataclass(frozen=True)
class BlockSparseTaskCSR:
    prog_cluster: torch.Tensor
    prog_block: torch.Tensor
    row_ptr: torch.Tensor
    nbr_cluster: torch.Tensor
    nbr_block: torch.Tensor
    max_tasks: int


@dataclass(frozen=True)
class BlockSparseTaskCSRBuckets:
    """Degree-bucketed task-CSR schedule to reduce MAX_TASKS padding.

    Each element is a pair (x_bucket, y_bucket) that can be launched with the
    standard task-CSR kernel.
    """

    buckets: Tuple[Tuple["BlockSparseTaskCSR", "BlockSparseTaskCSR"], ...]


def blocksparse_build_taskcsr(
    tasks: BlockSparseTasks, *, by: str
) -> BlockSparseTaskCSR:
    """Convert a flat (x_cluster,x_block,y_cluster,y_block) list into per-program CSR.

    `by="x"` builds CSR for x-programs and stores (y_cluster,y_block) as neighbors.
    `by="y"` builds CSR for y-programs and stores (x_cluster,x_block) as neighbors.
    """
    if by not in ("x", "y"):
        raise ValueError("by must be 'x' or 'y'.")
    device = tasks.x_cluster.device
    if tasks.x_cluster.numel() == 0:
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        empty_ptr = torch.empty((1,), device=device, dtype=torch.int32)
        empty_ptr[0] = 0
        return BlockSparseTaskCSR(empty_i32, empty_i32, empty_ptr, empty_i32, empty_i32, 0)

    if by == "x":
        key_hi = tasks.x_cluster.to(torch.int64)
        key_lo = tasks.x_block.to(torch.int64) & 0xFFFFFFFF
        nbr_cluster = tasks.y_cluster
        nbr_block = tasks.y_block
    else:
        key_hi = tasks.y_cluster.to(torch.int64)
        key_lo = tasks.y_block.to(torch.int64) & 0xFFFFFFFF
        nbr_cluster = tasks.x_cluster
        nbr_block = tasks.x_block

    keys = (key_hi << 32) + key_lo
    order = torch.argsort(keys)
    keys_s = keys[order]
    nbr_cluster_s = nbr_cluster[order].to(torch.int32)
    nbr_block_s = nbr_block[order].to(torch.int32)

    uniq, counts = torch.unique_consecutive(keys_s, return_counts=True)
    prog_cluster = (uniq >> 32).to(torch.int32)
    prog_block = (uniq & 0xFFFFFFFF).to(torch.int32)

    counts_i32 = counts.to(torch.int32)
    row_ptr = torch.empty((counts_i32.numel() + 1,), device=device, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(counts_i32, dim=0)
    max_tasks = int(counts.max().item()) if counts.numel() else 0
    return BlockSparseTaskCSR(
        prog_cluster=prog_cluster,
        prog_block=prog_block,
        row_ptr=row_ptr,
        nbr_cluster=nbr_cluster_s,
        nbr_block=nbr_block_s,
        max_tasks=max_tasks,
    )


def _empty_taskcsr(device: torch.device) -> BlockSparseTaskCSR:
    empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
    empty_ptr = torch.empty((1,), device=device, dtype=torch.int32)
    empty_ptr[0] = 0
    return BlockSparseTaskCSR(empty_i32, empty_i32, empty_ptr, empty_i32, empty_i32, 0)


def _taskcsr_select(
    taskcsr: BlockSparseTaskCSR, prog_idx: torch.Tensor
) -> BlockSparseTaskCSR:
    """Select programs and pack their neighbor lists contiguously (GPU-only)."""
    if prog_idx.ndim != 1:
        raise ValueError("prog_idx must be 1D.")
    device = taskcsr.prog_cluster.device
    if prog_idx.numel() == 0:
        return _empty_taskcsr(device)

    prog_idx64 = prog_idx.to(torch.int64)
    prog_cluster = taskcsr.prog_cluster[prog_idx64].contiguous()
    prog_block = taskcsr.prog_block[prog_idx64].contiguous()

    starts = taskcsr.row_ptr[prog_idx64].to(torch.int64)
    ends = taskcsr.row_ptr[prog_idx64 + 1].to(torch.int64)
    counts = (ends - starts).clamp_min(0)

    row_ptr = torch.empty((counts.numel() + 1,), device=device, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(counts.to(torch.int32), dim=0)

    total = int(row_ptr[-1].item())
    if total == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return BlockSparseTaskCSR(prog_cluster, prog_block, row_ptr, empty, empty, 0)

    counts_i64 = counts.to(torch.int64)
    starts_rep = torch.repeat_interleave(starts, counts_i64)
    prefix = torch.cumsum(counts_i64, dim=0) - counts_i64
    prefix_rep = torch.repeat_interleave(prefix, counts_i64)
    offs = torch.arange(total, device=device, dtype=torch.int64) - prefix_rep
    src_idx = starts_rep + offs

    nbr_cluster = taskcsr.nbr_cluster[src_idx].to(torch.int32).contiguous()
    nbr_block = taskcsr.nbr_block[src_idx].to(torch.int32).contiguous()
    max_tasks = int(counts.max().item()) if counts.numel() else 0
    return BlockSparseTaskCSR(
        prog_cluster=prog_cluster.to(torch.int32),
        prog_block=prog_block.to(torch.int32),
        row_ptr=row_ptr,
        nbr_cluster=nbr_cluster,
        nbr_block=nbr_block,
        max_tasks=max_tasks,
    )


def blocksparse_build_taskcsr_buckets(
    taskcsr_x: BlockSparseTaskCSR,
    taskcsr_y: BlockSparseTaskCSR,
    *,
    bucket_bounds: Optional[Sequence[int]] = None,
) -> BlockSparseTaskCSRBuckets:
    """Partition programs into degree buckets (FlashAttention-varlen analogue)."""
    device = taskcsr_x.prog_cluster.device
    if taskcsr_y.prog_cluster.device != device:
        raise ValueError("taskcsr_x and taskcsr_y must be on the same device.")

    max_tasks = int(max(int(taskcsr_x.max_tasks), int(taskcsr_y.max_tasks)))
    if max_tasks <= 0:
        return BlockSparseTaskCSRBuckets(buckets=())

    total_prog = int(taskcsr_x.prog_cluster.numel()) + int(taskcsr_y.prog_cluster.numel())
    if total_prog < 4096 or max_tasks <= 32:
        # For small problems, bucket construction + extra kernel launches are
        # often slower than the padded task-CSR loop.
        return BlockSparseTaskCSRBuckets(buckets=((taskcsr_x, taskcsr_y),))

    if bucket_bounds is None:
        bounds: list[int] = []
        # Small buckets add launch overhead and rarely help (deg<=7 is cheap even if padded).
        b = 8
        while b < max_tasks:
            bounds.append(int(b))
            b *= 2
        bounds.append(int(max_tasks))
    else:
        bounds = [int(v) for v in bucket_bounds]
        if not bounds:
            raise ValueError("bucket_bounds must be non-empty.")
        if any(v <= 0 for v in bounds):
            raise ValueError("bucket_bounds entries must be > 0.")
        if sorted(bounds) != bounds:
            raise ValueError("bucket_bounds must be sorted ascending.")
        if bounds[-1] < max_tasks:
            bounds = list(bounds) + [int(max_tasks)]

    deg_x = (taskcsr_x.row_ptr[1:] - taskcsr_x.row_ptr[:-1]).to(torch.int32)
    deg_y = (taskcsr_y.row_ptr[1:] - taskcsr_y.row_ptr[:-1]).to(torch.int32)

    prev = 0
    bucket_pairs: list[tuple[BlockSparseTaskCSR, BlockSparseTaskCSR]] = []
    for bound in bounds:
        mask_x = (deg_x > int(prev)) & (deg_x <= int(bound))
        mask_y = (deg_y > int(prev)) & (deg_y <= int(bound))
        idx_x = torch.nonzero(mask_x, as_tuple=False).flatten().to(torch.int32)
        idx_y = torch.nonzero(mask_y, as_tuple=False).flatten().to(torch.int32)
        bx = _taskcsr_select(taskcsr_x, idx_x)
        by = _taskcsr_select(taskcsr_y, idx_y)
        if bx.prog_cluster.numel() + by.prog_cluster.numel() > 0:
            bucket_pairs.append((bx, by))
        prev = int(bound)

    return BlockSparseTaskCSRBuckets(buckets=tuple(bucket_pairs))


@triton.jit
def _geomloss_symmetric_step_sqeuclid_taskcsr_impl(
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
    prog_x_cluster_ptr,
    prog_x_block_ptr,
    row_ptr_x_ptr,
    nbr_y_cluster_ptr,
    nbr_y_block_ptr,
    prog_y_cluster_ptr,
    prog_y_block_ptr,
    row_ptr_y_ptr,
    nbr_x_cluster_ptr,
    nbr_x_block_ptr,
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
    MAX_TASKS_X: tl.constexpr,
    MAX_TASKS_Y: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps_log2 = inv_eps * log2e

    if pid < n_prog_x:
        cluster = tl.load(prog_x_cluster_ptr + pid).to(tl.int32)
        blk = tl.load(prog_x_block_ptr + pid).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + cluster).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + cluster + 1).to(tl.int32)

        offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = (offs_m < end_x) & (offs_m < n)

        f_old = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
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
        elif D == 3:
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
        else:
            x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(
                tl.float32
            )

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

            g_v = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(
                tl.float32
            )
            logb_v = tl.load(
                logb_ptr + offs_n * stride_logb, mask=mask_n, other=-float("inf")
            ).to(tl.float32)
            if USE_EXP2:
                logb_v = logb_v * log2e

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
                y1v = tl.load(
                    y_ptr + offs_n * stride_y0 + 1 * stride_y1,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                y2 = tl.fma(y1v, y1v, y0 * y0)
                dot = x0[:, None] * y0[None, :] + x1v[:, None] * y1v[None, :]
            elif D == 3:
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
                dot = (
                    x0[:, None] * y0[None, :]
                    + x1v[:, None] * y1v[None, :]
                    + x2v[:, None] * y2v[None, :]
                )
            else:
                y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(
                    tl.float32
                )
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
                vals = tl.fma(g_v[None, :] - cost, inv_eps_log2, logb_v[None, :])
                block_max = tl.max(vals, axis=1)
                new_m = tl.maximum(m_i, block_max)
                alpha_m = tl.exp2(m_i - new_m)
                w = tl.exp2(vals - new_m[:, None])
            else:
                vals = (g_v[None, :] - cost) * inv_eps + logb_v[None, :]
                block_max = tl.max(vals, axis=1)
                new_m = tl.maximum(m_i, block_max)
                alpha_m = tl.exp(m_i - new_m)
                w = tl.exp(vals - new_m[:, None])
            w = tl.where(mask_mn, w, 0.0)
            s_i = s_i * alpha_m + tl.sum(w, axis=1)
            m_i = new_m

        valid = (m_i != -float("inf")) & (s_i > 0)
        lse = (m_i + tl.log2(s_i)) * ln2 if USE_EXP2 else (m_i + tl.log(s_i))
        cand = -eps * lse
        cand = tl.where(valid, cand, f_old)
        f_new = (1.0 - alpha) * f_old + alpha * cand
        tl.store(f_out_ptr + offs_m * stride_f_out, f_new, mask=mask_m)
        return

    pid_y = pid - n_prog_x
    cluster = tl.load(prog_y_cluster_ptr + pid_y).to(tl.int32)
    blk = tl.load(prog_y_block_ptr + pid_y).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cluster + 1).to(tl.int32)

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < end_y) & (offs_n < m)

    g_old = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
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
    elif D == 3:
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
    else:
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(
            tl.float32
        )

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    row_start = tl.load(row_ptr_y_ptr + pid_y).to(tl.int32)
    row_end = tl.load(row_ptr_y_ptr + pid_y + 1).to(tl.int32)

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

        f_v = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(
            tl.float32
        )
        loga_v = tl.load(
            loga_ptr + offs_m * stride_loga, mask=mask_m, other=-float("inf")
        ).to(tl.float32)
        if USE_EXP2:
            loga_v = loga_v * log2e

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
            x1v = tl.load(
                x_ptr + offs_m * stride_x0 + 1 * stride_x1,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.fma(x1v, x1v, x0 * x0)
            dot = x0[:, None] * y0[None, :] + x1v[:, None] * y1v[None, :]
        elif D == 3:
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
            dot = (
                x0[:, None] * y0[None, :]
                + x1v[:, None] * y1v[None, :]
                + x2v[:, None] * y2v[None, :]
            )
        else:
            x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(
                tl.float32
            )
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
            vals = tl.fma(f_v[:, None] - cost, inv_eps_log2, loga_v[:, None])
            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            alpha_m = tl.exp2(m_j - new_m)
            w = tl.exp2(vals - new_m[None, :])
        else:
            vals = (f_v[:, None] - cost) * inv_eps + loga_v[:, None]
            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            alpha_m = tl.exp(m_j - new_m)
            w = tl.exp(vals - new_m[None, :])
        w = tl.where(mask_mn, w, 0.0)
        s_j = s_j * alpha_m + tl.sum(w, axis=0)
        m_j = new_m

    valid = (m_j != -float("inf")) & (s_j > 0)
    lse = (m_j + tl.log2(s_j)) * ln2 if USE_EXP2 else (m_j + tl.log(s_j))
    cand = -eps * lse
    cand = tl.where(valid, cand, g_old)
    g_new = (1.0 - alpha) * g_old + alpha * cand
    tl.store(g_out_ptr + offs_n * stride_g_out, g_new, mask=mask_n)


def geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
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
    taskcsr_x: Optional[BlockSparseTaskCSR] = None,
    taskcsr_y: Optional[BlockSparseTaskCSR] = None,
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
) -> Tuple[BlockSparseTaskCSR, BlockSparseTaskCSR]:
    """Ranges-based blocksparse step without atomics (task-CSR, padded to max tasks)."""
    if taskcsr_x is None or taskcsr_y is None:
        tasks = blocksparse_build_tasks_from_csr(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            col_idx_x=col_idx_x,
            block_m=int(block_m),
            block_n=int(block_n),
        )
        taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
        taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    n_prog_x = int(taskcsr_x.prog_cluster.numel())
    n_prog_y = int(taskcsr_y.prog_cluster.numel())
    if n_prog_x + n_prog_y == 0:
        return taskcsr_x, taskcsr_y

    grid = (n_prog_x + n_prog_y,)

    _geomloss_symmetric_step_sqeuclid_taskcsr_impl[grid](
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
        taskcsr_x.prog_cluster,
        taskcsr_x.prog_block,
        taskcsr_x.row_ptr,
        taskcsr_x.nbr_cluster,
        taskcsr_x.nbr_block,
        taskcsr_y.prog_cluster,
        taskcsr_y.prog_block,
        taskcsr_y.row_ptr,
        taskcsr_y.nbr_cluster,
        taskcsr_y.nbr_block,
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
        MAX_TASKS_X=int(taskcsr_x.max_tasks),
        MAX_TASKS_Y=int(taskcsr_y.max_tasks),
        D=int(x.shape[1]),
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        ALLOW_TF32=bool(allow_tf32),
        USE_EXP2=bool(use_exp2),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return taskcsr_x, taskcsr_y


def geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed(
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
    taskcsr_x: Optional[BlockSparseTaskCSR] = None,
    taskcsr_y: Optional[BlockSparseTaskCSR] = None,
    buckets: Optional[BlockSparseTaskCSRBuckets] = None,
    bucket_bounds: Optional[Sequence[int]] = None,
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
) -> Tuple[BlockSparseTaskCSR, BlockSparseTaskCSR, BlockSparseTaskCSRBuckets]:
    """Bucketed task-CSR symmetric step (reduces MAX_TASKS padding).

    Buckets should be built once and reused across iterations for best performance.
    """
    if taskcsr_x is None or taskcsr_y is None:
        tasks = blocksparse_build_tasks_from_csr(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            col_idx_x=col_idx_x,
            block_m=int(block_m),
            block_n=int(block_n),
        )
        taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
        taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    if buckets is None:
        buckets = blocksparse_build_taskcsr_buckets(
            taskcsr_x, taskcsr_y, bucket_bounds=bucket_bounds
        )

    # Any programs that are absent (deg==0) should keep their previous values.
    f_out.copy_(f)
    g_out.copy_(g)

    for bx, by in buckets.buckets:
        geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
            x,
            y,
            f,
            g,
            loga,
            logb,
            x2,
            y2,
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            col_idx_x=col_idx_x,
            row_ptr_y=row_ptr_y,
            col_idx_y=col_idx_y,
            taskcsr_x=bx,
            taskcsr_y=by,
            f_out=f_out,
            g_out=g_out,
            eps=float(eps),
            alpha=float(alpha),
            block_m=int(block_m),
            block_n=int(block_n),
            block_k=int(block_k),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
        )

    return taskcsr_x, taskcsr_y, buckets
