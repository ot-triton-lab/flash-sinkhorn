from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class BlockSparseTasks:
    x_cluster: torch.Tensor
    x_block: torch.Tensor
    y_cluster: torch.Tensor
    y_block: torch.Tensor


def _ceil_div(a: torch.Tensor, b: int) -> torch.Tensor:
    return (a + int(b) - 1) // int(b)


def blocksparse_build_tasks_from_csr(
    *,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    col_idx_x: torch.Tensor,
    block_m: int,
    block_n: int,
) -> BlockSparseTasks:
    """Build a KeOps-like (ranges) task list for block-sparse reductions.

    This expands the kept cluster pairs (CSR on clusters) into a flat list of
    (x_cluster, x_block, y_cluster, y_block) tasks, without any padding to
    `max_deg` or `max_blocks`.
    """
    device = offsets_x.device
    if offsets_x.ndim != 1 or offsets_y.ndim != 1:
        raise ValueError("offsets_x/offsets_y must be 1D.")
    if row_ptr_x.ndim != 1 or col_idx_x.ndim != 1:
        raise ValueError("row_ptr_x/col_idx_x must be 1D.")
    if int(block_m) <= 0 or int(block_n) <= 0:
        raise ValueError("block sizes must be > 0.")

    cx = int(offsets_x.numel() - 1)
    if cx <= 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return BlockSparseTasks(empty, empty, empty, empty)

    deg = (row_ptr_x[1:] - row_ptr_x[:-1]).to(torch.int64)
    if deg.numel() != cx:
        raise ValueError("row_ptr_x must have len(offsets_x).")
    num_edges = int(deg.sum().item())
    if num_edges == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return BlockSparseTasks(empty, empty, empty, empty)

    x_cluster_edges = torch.repeat_interleave(
        torch.arange(cx, device=device, dtype=torch.int32), deg
    )
    y_cluster_edges = col_idx_x.to(device=device, dtype=torch.int32)
    if y_cluster_edges.numel() != num_edges:
        raise ValueError("col_idx_x must have row_ptr_x[-1] entries.")

    counts_x = (offsets_x[1:] - offsets_x[:-1]).to(torch.int64).clamp_min(0)
    counts_y = (offsets_y[1:] - offsets_y[:-1]).to(torch.int64).clamp_min(0)
    nblocks_x = _ceil_div(counts_x, int(block_m))
    nblocks_y = _ceil_div(counts_y, int(block_n))

    # Expand edges by x blocks.
    bx = nblocks_x[x_cluster_edges].to(torch.int64)
    total_edge_blocks = int(bx.sum().item())
    if total_edge_blocks == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return BlockSparseTasks(empty, empty, empty, empty)

    edge_ids = torch.arange(num_edges, device=device, dtype=torch.int64)
    edge_rep = torch.repeat_interleave(edge_ids, bx)
    x_cluster_eb = x_cluster_edges[edge_rep]
    y_cluster_eb = y_cluster_edges[edge_rep]

    starts = torch.cumsum(bx, dim=0) - bx
    starts_rep = torch.repeat_interleave(starts, bx)
    x_block_eb = (torch.arange(total_edge_blocks, device=device, dtype=torch.int64) - starts_rep).to(
        torch.int32
    )

    # Expand by y blocks.
    by = nblocks_y[y_cluster_eb].to(torch.int64)
    total_tasks = int(by.sum().item())
    if total_tasks == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return BlockSparseTasks(empty, empty, empty, empty)

    eb_ids = torch.arange(total_edge_blocks, device=device, dtype=torch.int64)
    eb_rep = torch.repeat_interleave(eb_ids, by)
    x_cluster = x_cluster_eb[eb_rep]
    y_cluster = y_cluster_eb[eb_rep]
    x_block = x_block_eb[eb_rep]

    starts2 = torch.cumsum(by, dim=0) - by
    starts2_rep = torch.repeat_interleave(starts2, by)
    y_block = (torch.arange(total_tasks, device=device, dtype=torch.int64) - starts2_rep).to(
        torch.int32
    )

    return BlockSparseTasks(
        x_cluster.to(torch.int32),
        x_block.to(torch.int32),
        y_cluster.to(torch.int32),
        y_block.to(torch.int32),
    )


@triton.jit
def _lse_max_sqeuclid_tasks_atomic_impl(
    x_ptr,
    y_ptr,
    g_ptr,
    logw_ptr,
    x2_ptr,
    y2_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    task_x_cluster_ptr,
    task_x_block_ptr,
    task_y_cluster_ptr,
    task_y_block_ptr,
    m_out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_g,
    stride_logw,
    stride_x2,
    stride_y2,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    x_cluster = tl.load(task_x_cluster_ptr + pid).to(tl.int32)
    x_block = tl.load(task_x_block_ptr + pid).to(tl.int32)
    y_cluster = tl.load(task_y_cluster_ptr + pid).to(tl.int32)
    y_block = tl.load(task_y_block_ptr + pid).to(tl.int32)

    start_x = tl.load(offsets_x_ptr + x_cluster).to(tl.int32)
    end_x = tl.load(offsets_x_ptr + x_cluster + 1).to(tl.int32)
    start_y = tl.load(offsets_y_ptr + y_cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + y_cluster + 1).to(tl.int32)

    offs_m = start_x + x_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = (offs_m < end_x) & (offs_m < n)

    offs_n = start_y + y_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < end_y) & (offs_n < m)

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    mask_mn = mask_m[:, None] & mask_n[None, :]
    g = tl.load(
        g_ptr + offs_n[None, :] * stride_g + offs_m[:, None] * 0,
        mask=mask_mn,
        other=0.0,
    ).to(tl.float32)
    logw = tl.load(
        logw_ptr + offs_n[None, :] * stride_logw + offs_m[:, None] * 0,
        mask=mask_mn,
        other=-float("inf"),
    ).to(tl.float32)
    if USE_EXP2:
        logw = logw * log2e

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
        vals = tl.fma(g - cost, inv_eps_log2, logw)
    else:
        vals = (g - cost) * inv_eps + logw
    vals = tl.where(mask_mn, vals, -float("inf"))

    block_max = tl.max(vals, axis=1)
    tl.atomic_max(m_out_ptr + offs_m, block_max, mask=mask_m)


@triton.jit
def _lse_sumexp_sqeuclid_tasks_atomic_impl(
    x_ptr,
    y_ptr,
    g_ptr,
    logw_ptr,
    x2_ptr,
    y2_ptr,
    offsets_x_ptr,
    offsets_y_ptr,
    task_x_cluster_ptr,
    task_x_block_ptr,
    task_y_cluster_ptr,
    task_y_block_ptr,
    m_in_ptr,
    l_out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_g,
    stride_logw,
    stride_x2,
    stride_y2,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    x_cluster = tl.load(task_x_cluster_ptr + pid).to(tl.int32)
    x_block = tl.load(task_x_block_ptr + pid).to(tl.int32)
    y_cluster = tl.load(task_y_cluster_ptr + pid).to(tl.int32)
    y_block = tl.load(task_y_block_ptr + pid).to(tl.int32)

    start_x = tl.load(offsets_x_ptr + x_cluster).to(tl.int32)
    end_x = tl.load(offsets_x_ptr + x_cluster + 1).to(tl.int32)
    start_y = tl.load(offsets_y_ptr + y_cluster).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + y_cluster + 1).to(tl.int32)

    offs_m = start_x + x_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = (offs_m < end_x) & (offs_m < n)

    offs_n = start_y + y_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < end_y) & (offs_n < m)

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)
    m_i = tl.load(m_in_ptr + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)

    mask_mn = mask_m[:, None] & mask_n[None, :]
    g = tl.load(
        g_ptr + offs_n[None, :] * stride_g + offs_m[:, None] * 0,
        mask=mask_mn,
        other=0.0,
    ).to(tl.float32)
    logw = tl.load(
        logw_ptr + offs_n[None, :] * stride_logw + offs_m[:, None] * 0,
        mask=mask_mn,
        other=-float("inf"),
    ).to(tl.float32)
    if USE_EXP2:
        logw = logw * log2e

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
        vals = tl.fma(g - cost, inv_eps_log2, logw)
        w = tl.exp2(vals - m_i[:, None])
    else:
        vals = (g - cost) * inv_eps + logw
        w = tl.exp(vals - m_i[:, None])
    w = tl.where(mask_mn, w, 0.0)
    partial = tl.sum(w, axis=1)
    tl.atomic_add(l_out_ptr + offs_m, partial, mask=mask_m)


@triton.jit
def _lse_finalize_from_ml_impl(
    f_ptr,
    m_ptr,
    l_ptr,
    f_out_ptr,
    n,
    stride_f,
    stride_m,
    stride_l,
    stride_f_out,
    eps,
    alpha,
    BLOCK_M: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < n

    f_old = tl.load(f_ptr + offs_m * stride_f, mask=mask, other=0.0).to(tl.float32)
    m_i = tl.load(m_ptr + offs_m * stride_m, mask=mask, other=-float("inf")).to(tl.float32)
    l_i = tl.load(l_ptr + offs_m * stride_l, mask=mask, other=0.0).to(tl.float32)

    valid = (m_i != -float("inf")) & (l_i > 0)
    if USE_EXP2:
        ln2 = 0.6931471805599453
        lse = (m_i + tl.log2(l_i)) * ln2
    else:
        lse = m_i + tl.log(l_i)
    cand = -eps * lse
    cand = tl.where(valid, cand, f_old)
    f_new = (1.0 - alpha) * f_old + alpha * cand
    tl.store(f_out_ptr + offs_m * stride_f_out, f_new, mask=mask)


def _atomic_lse_update_sqeuclid(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    f_in: torch.Tensor,
    g: torch.Tensor,
    logw: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    tasks: BlockSparseTasks,
    eps: float,
    alpha: float,
    block_m: int,
    block_n: int,
    block_k: int,
    allow_tf32: bool,
    use_exp2: bool,
    m_buf: Optional[torch.Tensor] = None,
    l_buf: Optional[torch.Tensor] = None,
    f_out: torch.Tensor,
) -> None:
    if tasks.x_cluster.numel() == 0:
        f_out.copy_(f_in)
        return

    n = int(x.shape[0])
    m = int(y.shape[0])
    d = int(x.shape[1])

    if m_buf is None:
        m_buf = torch.empty((n,), device=x.device, dtype=torch.float32)
    if l_buf is None:
        l_buf = torch.empty((n,), device=x.device, dtype=torch.float32)
    if m_buf.shape != (n,) or l_buf.shape != (n,):
        raise ValueError("m_buf and l_buf must have shape (x.shape[0],).")
    if m_buf.device != x.device or l_buf.device != x.device:
        raise ValueError("m_buf and l_buf must be on the same device as x.")
    if m_buf.dtype != torch.float32 or l_buf.dtype != torch.float32:
        raise ValueError("m_buf and l_buf must be float32.")

    # Reset scratch each call (avoid per-iteration allocations in the caller).
    m_buf.fill_(-float("inf"))
    l_buf.zero_()

    grid = (int(tasks.x_cluster.numel()),)
    _lse_max_sqeuclid_tasks_atomic_impl[grid](
        x,
        y,
        g,
        logw,
        x2,
        y2,
        offsets_x,
        offsets_y,
        tasks.x_cluster,
        tasks.x_block,
        tasks.y_cluster,
        tasks.y_block,
        m_buf,
        n,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        g.stride(0),
        logw.stride(0),
        x2.stride(0),
        y2.stride(0),
        float(eps),
        D=d,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        ALLOW_TF32=bool(allow_tf32),
        USE_EXP2=bool(use_exp2),
    )
    _lse_sumexp_sqeuclid_tasks_atomic_impl[grid](
        x,
        y,
        g,
        logw,
        x2,
        y2,
        offsets_x,
        offsets_y,
        tasks.x_cluster,
        tasks.x_block,
        tasks.y_cluster,
        tasks.y_block,
        m_buf,
        l_buf,
        n,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        g.stride(0),
        logw.stride(0),
        x2.stride(0),
        y2.stride(0),
        float(eps),
        D=d,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        ALLOW_TF32=bool(allow_tf32),
        USE_EXP2=bool(use_exp2),
    )

    grid_f = (triton.cdiv(n, int(block_m)),)
    _lse_finalize_from_ml_impl[grid_f](
        f_in,
        m_buf,
        l_buf,
        f_out,
        n,
        f_in.stride(0),
        m_buf.stride(0),
        l_buf.stride(0),
        f_out.stride(0),
        float(eps),
        float(alpha),
        BLOCK_M=int(block_m),
        USE_EXP2=bool(use_exp2),
        num_warps=4,
        num_stages=1,
    )


def geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic(
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
    tasks_f: Optional[BlockSparseTasks] = None,
    tasks_g: Optional[BlockSparseTasks] = None,
    scratch_f: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    scratch_g: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    eps: float,
    alpha: float,
    block_m: int,
    block_n: int,
    block_k: int,
    allow_tf32: bool,
    use_exp2: bool,
) -> None:
    """Experimental: ranges-based block-sparse symmetric step using atomics."""
    if tasks_f is None:
        tasks_f = blocksparse_build_tasks_from_csr(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            col_idx_x=col_idx_x,
            block_m=int(block_m),
            block_n=int(block_n),
        )
    if tasks_g is None:
        tasks_g = blocksparse_build_tasks_from_csr(
            offsets_x=offsets_y,
            offsets_y=offsets_x,
            row_ptr_x=row_ptr_y,
            col_idx_x=col_idx_y,
            block_m=int(block_n),
            block_n=int(block_m),
        )

    m_buf_f = l_buf_f = None
    m_buf_g = l_buf_g = None
    if scratch_f is not None:
        m_buf_f, l_buf_f = scratch_f
    if scratch_g is not None:
        m_buf_g, l_buf_g = scratch_g

    # f update uses (g, logb) reduced over y.
    _atomic_lse_update_sqeuclid(
        x=x,
        y=y,
        f_in=f,
        g=g,
        logw=logb,
        x2=x2,
        y2=y2,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        tasks=tasks_f,
        eps=float(eps),
        alpha=float(alpha),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
        m_buf=m_buf_f,
        l_buf=l_buf_f,
        f_out=f_out,
    )

    # g update uses (f, loga) reduced over x.
    _atomic_lse_update_sqeuclid(
        x=y,
        y=x,
        f_in=g,
        g=f,
        logw=loga,
        x2=y2,
        y2=x2,
        offsets_x=offsets_y,
        offsets_y=offsets_x,
        tasks=tasks_g,
        eps=float(eps),
        alpha=float(alpha),
        block_m=int(block_n),
        block_n=int(block_m),
        block_k=int(block_k),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
        m_buf=m_buf_g,
        l_buf=l_buf_g,
        f_out=g_out,
    )
