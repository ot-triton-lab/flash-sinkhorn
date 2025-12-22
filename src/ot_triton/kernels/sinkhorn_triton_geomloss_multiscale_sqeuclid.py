from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    epsilon_schedule,
    log_weights,
    max_diameter,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    _default_block_sizes as _dense_default_block_sizes,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    _geomloss_symmetric_step_sqeuclid_impl as _dense_step_kernel,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_sqeuclid import (
    blocksparse_prepare_metadata,
    geomloss_blocksparse_grad_sqeuclid,
    geomloss_blocksparse_symmetric_step_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    BlockSparseTasks,
    blocksparse_build_tasks_from_csr,
    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    BlockSparseTaskCSR,
    BlockSparseTaskCSRBuckets,
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed,
)
from ot_triton.kernels.sinkhorn_triton_sqeuclid import apply_lse_kernel_sqeuclid


@dataclass(frozen=True)
class _VoxelClusterization:
    x_fine_sorted: torch.Tensor
    w_fine_sorted: torch.Tensor
    perm_fine: torch.Tensor
    offsets: torch.Tensor
    x_coarse: torch.Tensor
    w_coarse: torch.Tensor
    scale: float


def _voxel_clusterize(
    w: torch.Tensor,
    x: torch.Tensor,
    *,
    scale: float,
) -> _VoxelClusterization:
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if x.ndim != 2 or w.ndim != 1:
        raise ValueError("Expected x shaped (N,D) and w shaped (N,).")
    if w.shape[0] != x.shape[0]:
        raise ValueError("w and x must have the same length.")

    device = x.device
    x_f = x.float()
    w_f = w.float()

    mins = x_f.min(dim=0).values
    coords = torch.floor((x_f - mins) / float(scale)).to(torch.int32)

    # Unique voxel coordinates define coarse clusters (GeomLoss-style grid_cluster).
    # Avoid `torch.unique(coords, dim=0)` (expensive); pack coords to a 1D key.
    if int(x.shape[1]) == 1:
        key = coords[:, 0].to(torch.int64)
    elif int(x.shape[1]) == 2:
        c0 = coords[:, 0].to(torch.int64)
        c1 = coords[:, 1].to(torch.int64)
        base0 = int(c0.max().item()) + 1 if c0.numel() else 1
        key = c0 + base0 * c1
    else:
        c0 = coords[:, 0].to(torch.int64)
        c1 = coords[:, 1].to(torch.int64)
        c2 = coords[:, 2].to(torch.int64)
        base0 = int(c0.max().item()) + 1 if c0.numel() else 1
        base1 = int(c1.max().item()) + 1 if c1.numel() else 1
        key = c0 + base0 * c1 + (base0 * base1) * c2

    _, labels = torch.unique(key, return_inverse=True)
    labels = labels.to(torch.int64)
    n_clusters = int(labels.max().item()) + 1 if labels.numel() else 0

    w_coarse = torch.zeros((n_clusters,), device=device, dtype=torch.float32)
    w_coarse.index_add_(0, labels, w_f)
    x_sum = torch.zeros((n_clusters, x.shape[1]), device=device, dtype=torch.float32)
    x_sum.index_add_(0, labels, w_f[:, None] * x_f)
    x_coarse = x_sum / w_coarse[:, None]

    perm = torch.argsort(labels)
    x_fine_sorted = x[perm]
    w_fine_sorted = w[perm]

    labels_sorted = labels[perm]
    counts = torch.bincount(labels_sorted, minlength=n_clusters).to(torch.int32)
    offsets = torch.empty((n_clusters + 1,), device=device, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)

    return _VoxelClusterization(
        x_fine_sorted=x_fine_sorted,
        w_fine_sorted=w_fine_sorted,
        perm_fine=perm,
        offsets=offsets,
        x_coarse=x_coarse,
        w_coarse=w_coarse,
        scale=float(scale),
    )


def _voxel_cluster_centroids(
    w: torch.Tensor,
    x: torch.Tensor,
    *,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voxel clustering that returns only (centroids, weights)."""
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if x.ndim != 2 or w.ndim != 1:
        raise ValueError("Expected x shaped (N,D) and w shaped (N,).")
    if w.shape[0] != x.shape[0]:
        raise ValueError("w and x must have the same length.")

    device = x.device
    x_f = x.float()
    w_f = w.float()
    mins = x_f.min(dim=0).values
    coords = torch.floor((x_f - mins) / float(scale)).to(torch.int32)
    if int(x.shape[1]) == 1:
        key = coords[:, 0].to(torch.int64)
    elif int(x.shape[1]) == 2:
        c0 = coords[:, 0].to(torch.int64)
        c1 = coords[:, 1].to(torch.int64)
        base0 = int(c0.max().item()) + 1 if c0.numel() else 1
        key = c0 + base0 * c1
    else:
        c0 = coords[:, 0].to(torch.int64)
        c1 = coords[:, 1].to(torch.int64)
        c2 = coords[:, 2].to(torch.int64)
        base0 = int(c0.max().item()) + 1 if c0.numel() else 1
        base1 = int(c1.max().item()) + 1 if c1.numel() else 1
        key = c0 + base0 * c1 + (base0 * base1) * c2

    _, labels = torch.unique(key, return_inverse=True)
    labels = labels.to(torch.int64)
    n_clusters = int(labels.max().item()) + 1 if labels.numel() else 0

    w_coarse = torch.zeros((n_clusters,), device=device, dtype=torch.float32)
    w_coarse.index_add_(0, labels, w_f)
    x_sum = torch.zeros((n_clusters, x.shape[1]), device=device, dtype=torch.float32)
    x_sum.index_add_(0, labels, w_f[:, None] * x_f)
    x_coarse = x_sum / w_coarse[:, None]
    return x_coarse, w_coarse


def _jump_index_for_scale(eps_list: Sequence[float], *, scale2: float) -> int:
    # Match GeomLoss' point-cloud multiscale heuristic: jump between k-1 and k
    # when eps_list[k] becomes smaller than scale^2 (p=2), skipping the first 2 eps.
    jump = len(eps_list) - 1
    for k in range(2, len(eps_list)):
        if float(scale2) > float(eps_list[k]):
            jump = k - 1
            break
    return int(jump)


def _csr_from_keep(keep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if keep.ndim != 2:
        raise ValueError("keep must be 2D.")
    device = keep.device
    rows, cols = keep.shape
    deg = keep.sum(dim=1).to(torch.int32)
    row_ptr = torch.empty((rows + 1,), device=device, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(deg, dim=0)
    if int(row_ptr[-1].item()) == 0:
        col_idx = torch.empty((0,), device=device, dtype=torch.int32)
        return row_ptr, col_idx

    # `torch.nonzero` returns indices in row-major (lexicographic) order, so
    # columns are already grouped by row and sorted; no explicit sort needed.
    idx = torch.nonzero(keep, as_tuple=False)
    col_idx = idx[:, 1].to(torch.int32).contiguous()
    return row_ptr, col_idx


def _multiscale_default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    """Block sizes tuned for d in {1,2,3} multiscale kernels.

    In the fine (blocksparse) step, smaller tiles reduce register pressure for
    D<=3 and tend to improve occupancy/throughput versus the dense defaults.
    """
    if d not in (1, 2, 3):
        return _dense_default_block_sizes(d, dtype, allow_tf32)
    # Empirically best on A100 for the task-CSR fine step: reduce task padding by
    # using a wider y-tile, and run with a single warp.
    return 32, 64, 16, 1


def _bench_cuda_ms(fn, *, warmup: int, rep: int) -> float:
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be >= 0 and rep must be > 0.")
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(rep)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(rep)


def _multiscale_autotune_key(
    *, x: torch.Tensor, y: torch.Tensor, allow_tf32: bool, use_exp2: bool
) -> Tuple[int, int, str, int, bool, bool, int, int]:
    if not x.is_cuda:
        raise ValueError("Expected CUDA tensor for autotune key.")
    props = torch.cuda.get_device_properties(x.device)
    n, d = x.shape
    m = int(y.shape[0])
    n_bucket = int(math.log2(max(1, int(n))))
    m_bucket = int(math.log2(max(1, int(m))))
    return (
        int(props.major),
        int(props.minor),
        str(x.dtype),
        int(d),
        bool(allow_tf32),
        bool(use_exp2),
        int(n_bucket),
        int(m_bucket),
    )


_MULTISCALE_TASKCSR_AUTOTUNE_CACHE: dict[
    Tuple[int, int, str, int, bool, bool, int, int], Tuple[int, int, int, int]
] = {}


def _autotune_taskcsr_bn(
    *,
    x_fine: torch.Tensor,
    y_fine: torch.Tensor,
    f_in: torch.Tensor,
    g_in: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    col_idx_x: torch.Tensor,
    row_ptr_y: torch.Tensor,
    col_idx_y: torch.Tensor,
    eps: float,
    allow_tf32: bool,
    use_exp2: bool,
    bm: int,
    bk: int,
    num_stages: int,
) -> Tuple[int, int]:
    """Select a good `block_n` (power-of-two) for the fine task-CSR step.

    We keep bm fixed (default 32) and benchmark a small set of bn values.
    Returns (best_bn, num_warps).
    """
    from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
        blocksparse_build_tasks_from_csr,
    )
    from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
        blocksparse_build_taskcsr,
        geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
    )

    if bm & (bm - 1) != 0:
        raise ValueError("bm must be a power of two.")
    if bk < 16:
        bk = 16

    candidates = [16, 32, 64, 128]
    candidates = [bn for bn in candidates if bn <= 128]

    f_out = torch.empty_like(f_in)
    g_out = torch.empty_like(g_in)

    best = None
    best_ms = float("inf")
    best_bn = 64
    best_nw = 1

    with torch.no_grad():
        for bn in candidates:
            if bn & (bn - 1) != 0:
                continue

            tasks = blocksparse_build_tasks_from_csr(
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                block_m=int(bm),
                block_n=int(bn),
            )
            taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
            taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

            # Choose warps conservatively: 1 warp is typically best for D<=3.
            nw = 1

            def launch():
                geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
                    x_fine,
                    y_fine,
                    f_in,
                    g_in,
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
                    taskcsr_x=taskcsr_x,
                    taskcsr_y=taskcsr_y,
                    f_out=f_out,
                    g_out=g_out,
                    eps=float(eps),
                    alpha=0.5,
                    block_m=int(bm),
                    block_n=int(bn),
                    block_k=int(bk),
                    num_warps=int(nw),
                    num_stages=int(num_stages),
                    allow_tf32=bool(allow_tf32),
                    use_exp2=bool(use_exp2),
                )

            # Compile + warm.
            launch()
            torch.cuda.synchronize()

            ms = _bench_cuda_ms(launch, warmup=2, rep=5)
            if ms < best_ms:
                best_ms = ms
                best_bn = int(bn)
                best_nw = int(nw)
                best = (best_bn, best_nw)

    if best is None:
        return 64, 1
    return best_bn, best_nw


def _estimate_taskcsr_max_tasks(
    *,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    row_ptr_x: torch.Tensor,
    col_idx_x: torch.Tensor,
    row_ptr_y: torch.Tensor,
    col_idx_y: torch.Tensor,
    block_m: int,
    block_n: int,
) -> Tuple[int, int]:
    """Estimate max tasks per x-block / y-block program for task-CSR backends.

    A task for a given x-block corresponds to iterating over all y-blocks in all
    neighbor y-clusters; this returns the maximum such count for x- and y-side.
    """
    if offsets_x.numel() <= 1 or offsets_y.numel() <= 1:
        return 0, 0
    device = offsets_x.device
    cx = int(offsets_x.numel() - 1)
    cy = int(offsets_y.numel() - 1)

    counts_x = (offsets_x[1:] - offsets_x[:-1]).to(torch.int64).clamp_min(0)
    counts_y = (offsets_y[1:] - offsets_y[:-1]).to(torch.int64).clamp_min(0)
    blocks_x = (counts_x + int(block_m) - 1) // int(block_m)
    blocks_y = (counts_y + int(block_n) - 1) // int(block_n)

    deg_x = (row_ptr_x[1:] - row_ptr_x[:-1]).to(torch.int64)
    deg_y = (row_ptr_y[1:] - row_ptr_y[:-1]).to(torch.int64)

    if deg_x.numel() != cx or deg_y.numel() != cy:
        raise ValueError("row_ptr length does not match offsets length.")

    max_tasks_x = 0
    if int(deg_x.sum().item()) > 0:
        x_edges = torch.repeat_interleave(
            torch.arange(cx, device=device, dtype=torch.int64), deg_x
        )
        y_edges = col_idx_x.to(torch.int64)
        sum_y_blocks = torch.zeros((cx,), device=device, dtype=torch.int64)
        sum_y_blocks.index_add_(0, x_edges, blocks_y[y_edges])
        max_tasks_x = int(sum_y_blocks.max().item()) if sum_y_blocks.numel() else 0

    max_tasks_y = 0
    if int(deg_y.sum().item()) > 0:
        y_edges = torch.repeat_interleave(
            torch.arange(cy, device=device, dtype=torch.int64), deg_y
        )
        x_edges = col_idx_y.to(torch.int64)
        sum_x_blocks = torch.zeros((cy,), device=device, dtype=torch.int64)
        sum_x_blocks.index_add_(0, y_edges, blocks_x[x_edges])
        max_tasks_y = int(sum_x_blocks.max().item()) if sum_x_blocks.numel() else 0

    return max_tasks_x, max_tasks_y


@dataclass(frozen=True)
class _GridSorted:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    perm_x: torch.Tensor
    perm_y: torch.Tensor
    offsets_x: torch.Tensor
    offsets_y: torch.Tensor
    pid_x_cell: torch.Tensor
    pid_x_block: torch.Tensor
    pid_y_cell: torch.Tensor
    pid_y_block: torch.Tensor
    max_blocks_x: int
    max_blocks_y: int
    sx: int
    sy: int
    sz: int
    cell_size: float
    cell_diameter: float


def _cluster_scale_from_diameter(diameter: float, d: int) -> float:
    # Match GeomLoss heuristic: diameter / (sqrt(D) * 2000^(1/D)).
    return float(diameter) / (math.sqrt(float(d)) * (2000.0 ** (1.0 / float(d))))


def _grid_sizes_from_bounds(
    mins: torch.Tensor, maxs: torch.Tensor, *, h: float
) -> Tuple[int, int, int]:
    span = (maxs - mins).clamp(min=0).float()
    sizes = torch.floor(span / float(h)).to(torch.int64) + 1
    sizes = torch.clamp(sizes, min=1)
    sx = int(sizes[0].item())
    sy = int(sizes[1].item()) if sizes.numel() > 1 else 1
    sz = int(sizes[2].item()) if sizes.numel() > 2 else 1
    return sx, sy, sz


def _cell_ids(
    pts: torch.Tensor, *, mins: torch.Tensor, h: float, sx: int, sy: int, sz: int
) -> torch.Tensor:
    pts_f = pts.float()
    d = int(pts_f.shape[1])
    coords = torch.floor((pts_f - mins[:d]) / float(h)).to(torch.int64)
    coords = torch.clamp(coords, min=0)
    if d >= 1:
        coords[:, 0] = torch.clamp(coords[:, 0], max=sx - 1)
    if d >= 2:
        coords[:, 1] = torch.clamp(coords[:, 1], max=sy - 1)
    if d >= 3:
        coords[:, 2] = torch.clamp(coords[:, 2], max=sz - 1)

    if d == 1:
        return coords[:, 0].to(torch.int32)
    if d == 2:
        return (coords[:, 0] + sx * coords[:, 1]).to(torch.int32)
    return (coords[:, 0] + sx * coords[:, 1] + (sx * sy) * coords[:, 2]).to(torch.int32)


def _offsets_from_cell_ids(cell_ids: torch.Tensor, *, n_cells: int) -> torch.Tensor:
    counts = torch.bincount(cell_ids.to(torch.int64), minlength=int(n_cells)).to(torch.int32)
    offsets = torch.empty((n_cells + 1,), device=cell_ids.device, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def _pid_maps_from_offsets(
    offsets: torch.Tensor, *, block: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if offsets.numel() <= 1:
        empty = torch.empty((0,), device=offsets.device, dtype=torch.int32)
        return empty, empty, 0

    counts = offsets[1:] - offsets[:-1]
    n_cells = int(counts.numel())
    n_blocks = (counts.to(torch.int64) + int(block) - 1) // int(block)
    max_blocks = int(n_blocks.max().item()) if n_blocks.numel() else 0
    total = int(n_blocks.sum().item())
    if total == 0:
        empty = torch.empty((0,), device=offsets.device, dtype=torch.int32)
        return empty, empty, max_blocks

    pid_cell = torch.repeat_interleave(
        torch.arange(n_cells, device=offsets.device, dtype=torch.int32),
        n_blocks.to(torch.int64),
    )

    starts = (torch.cumsum(n_blocks, dim=0) - n_blocks).to(torch.int64)
    global_block = torch.arange(total, device=offsets.device, dtype=torch.int64)
    pid_block = (global_block - starts[pid_cell.to(torch.int64)]).to(torch.int32)
    return pid_cell, pid_block, max_blocks


def _build_grid_sorted(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cell_size: float,
) -> _GridSorted:
    d = int(x.shape[1])
    mins = torch.stack((x.float().min(dim=0).values, y.float().min(dim=0).values)).min(dim=0).values
    maxs = torch.stack((x.float().max(dim=0).values, y.float().max(dim=0).values)).max(dim=0).values
    sx, sy, sz = _grid_sizes_from_bounds(mins, maxs, h=cell_size)
    n_cells = int(sx * sy * sz)

    cell_x = _cell_ids(x, mins=mins, h=cell_size, sx=sx, sy=sy, sz=sz)
    cell_y = _cell_ids(y, mins=mins, h=cell_size, sx=sx, sy=sy, sz=sz)

    perm_x = torch.argsort(cell_x)
    perm_y = torch.argsort(cell_y)

    x_s = x[perm_x]
    y_s = y[perm_y]
    a_s = a[perm_x]
    b_s = b[perm_y]

    offsets_x = _offsets_from_cell_ids(cell_x, n_cells=n_cells)
    offsets_y = _offsets_from_cell_ids(cell_y, n_cells=n_cells)

    pid_x_cell, pid_x_block, max_blocks_x = _pid_maps_from_offsets(offsets_x, block=128)
    pid_y_cell, pid_y_block, max_blocks_y = _pid_maps_from_offsets(offsets_y, block=128)

    # Note: pid maps are built for a default block size and will be rebuilt per launch
    # if block sizes differ; see launch wrapper.
    cell_diameter = float(cell_size) * math.sqrt(float(d))
    return _GridSorted(
        x=x_s,
        y=y_s,
        a=a_s,
        b=b_s,
        perm_x=perm_x,
        perm_y=perm_y,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        pid_x_cell=pid_x_cell.to(device=x.device),
        pid_x_block=pid_x_block.to(device=x.device),
        pid_y_cell=pid_y_cell.to(device=x.device),
        pid_y_block=pid_y_block.to(device=x.device),
        max_blocks_x=int(max_blocks_x),
        max_blocks_y=int(max_blocks_y),
        sx=sx,
        sy=sy,
        sz=sz,
        cell_size=float(cell_size),
        cell_diameter=cell_diameter,
    )


@triton.jit
def _geomloss_symmetric_step_sqeuclid_sparse_impl(
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
    pid_x_cell_ptr,
    pid_x_block_ptr,
    pid_y_cell_ptr,
    pid_y_block_ptr,
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
    sx,
    sy,
    sz,
    eps,
    alpha,
    radius2,
    rad,
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
    # f update: one program per (x_cell, block_in_cell)
    # ------------------------------------------------------------
    if pid < n_prog_x:
        cell = tl.load(pid_x_cell_ptr + pid).to(tl.int32)
        blk = tl.load(pid_x_block_ptr + pid).to(tl.int32)

        start_x = tl.load(offsets_x_ptr + cell).to(tl.int32)
        end_x = tl.load(offsets_x_ptr + cell + 1).to(tl.int32)
        count_x = end_x - start_x

        offs_m = start_x + blk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < (start_x + count_x)
        mask_m = mask_m & (offs_m < n)

        f_old = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        if D == 1:
            cx = cell
            for dx_idx in range(0, 2 * rad + 1):
                dx = dx_idx - rad
                nbcx = cx + dx
                ok = (nbcx >= 0) & (nbcx < sx)
                nb = tl.where(ok, nbcx, 0).to(tl.int32)

                start_y = tl.load(offsets_y_ptr + nb).to(tl.int32)
                end_y = tl.load(offsets_y_ptr + nb + 1).to(tl.int32)
                count_y = tl.where(ok, end_y - start_y, 0).to(tl.int32)

                for by in range(0, max_blocks_y):
                    offs_n = start_y + by * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < (start_y + count_y)
                    mask_n = mask_n & (offs_n < m)

                    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
                    logb = tl.load(
                        logb_ptr + offs_n * stride_logb, mask=mask_n, other=-float("inf")
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
                    keep = cost <= radius2

                    if USE_EXP2:
                        vals = tl.fma(g[None, :] - cost, inv_eps_log2, logb[None, :])
                    else:
                        vals = (g[None, :] - cost) * inv_eps + logb[None, :]
                    vals = tl.where(mask_n[None, :] & keep, vals, -float("inf"))

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
        elif D == 2:
            cx = cell % sx
            cy = cell // sx
            for dy_idx in range(0, 2 * rad + 1):
                dy = dy_idx - rad
                for dx_idx in range(0, 2 * rad + 1):
                    dx = dx_idx - rad
                    nbcx = cx + dx
                    nbcy = cy + dy
                    ok = (nbcx >= 0) & (nbcx < sx) & (nbcy >= 0) & (nbcy < sy)
                    nb = nbcx + sx * nbcy
                    nb = tl.where(ok, nb, 0).to(tl.int32)

                    start_y = tl.load(offsets_y_ptr + nb).to(tl.int32)
                    end_y = tl.load(offsets_y_ptr + nb + 1).to(tl.int32)
                    count_y = tl.where(ok, end_y - start_y, 0).to(tl.int32)

                    for by in range(0, max_blocks_y):
                        offs_n = start_y + by * BLOCK_N + tl.arange(0, BLOCK_N)
                        mask_n = offs_n < (start_y + count_y)
                        mask_n = mask_n & (offs_n < m)

                        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
                        logb = tl.load(
                            logb_ptr + offs_n * stride_logb, mask=mask_n, other=-float("inf")
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
                        keep = cost <= radius2

                        if USE_EXP2:
                            vals = tl.fma(g[None, :] - cost, inv_eps_log2, logb[None, :])
                        else:
                            vals = (g[None, :] - cost) * inv_eps + logb[None, :]
                        vals = tl.where(mask_n[None, :] & keep, vals, -float("inf"))

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
        else:
            stride_xy = sx * sy
            cz = cell // stride_xy
            rem = cell - cz * stride_xy
            cy = rem // sx
            cx = rem - cy * sx
            for dz_idx in range(0, 2 * rad + 1):
                dz = dz_idx - rad
                for dy_idx in range(0, 2 * rad + 1):
                    dy = dy_idx - rad
                    for dx_idx in range(0, 2 * rad + 1):
                        dx = dx_idx - rad
                        nbcx = cx + dx
                        nbcy = cy + dy
                        nbcz = cz + dz
                        ok = (
                            (nbcx >= 0)
                            & (nbcx < sx)
                            & (nbcy >= 0)
                            & (nbcy < sy)
                            & (nbcz >= 0)
                            & (nbcz < sz)
                        )
                        nb = nbcx + sx * nbcy + stride_xy * nbcz
                        nb = tl.where(ok, nb, 0).to(tl.int32)

                        start_y = tl.load(offsets_y_ptr + nb).to(tl.int32)
                        end_y = tl.load(offsets_y_ptr + nb + 1).to(tl.int32)
                        count_y = tl.where(ok, end_y - start_y, 0).to(tl.int32)

                        for by in range(0, max_blocks_y):
                            offs_n = start_y + by * BLOCK_N + tl.arange(0, BLOCK_N)
                            mask_n = offs_n < (start_y + count_y)
                            mask_n = mask_n & (offs_n < m)

                            g = tl.load(
                                g_ptr + offs_n * stride_g, mask=mask_n, other=0.0
                            ).to(tl.float32)
                            logb = tl.load(
                                logb_ptr + offs_n * stride_logb,
                                mask=mask_n,
                                other=-float("inf"),
                            ).to(tl.float32)
                            if USE_EXP2:
                                logb = logb * log2e
                            y2 = tl.load(
                                y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
                            ).to(tl.float32)

                            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                            for k0 in range(0, D, BLOCK_K):
                                offs_k = k0 + tl.arange(0, BLOCK_K)
                                mask_k = offs_k < D
                                xk = tl.load(
                                    x_ptr
                                    + offs_m[:, None] * stride_x0
                                    + offs_k[None, :] * stride_x1,
                                    mask=mask_m[:, None] & mask_k[None, :],
                                    other=0.0,
                                )
                                yk = tl.load(
                                    y_ptr
                                    + offs_n[None, :] * stride_y0
                                    + offs_k[:, None] * stride_y1,
                                    mask=mask_n[None, :] & mask_k[:, None],
                                    other=0.0,
                                )
                                dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

                            cost = x2[:, None] + y2[None, :] - 2.0 * dot
                            keep = cost <= radius2

                            if USE_EXP2:
                                vals = tl.fma(
                                    g[None, :] - cost, inv_eps_log2, logb[None, :]
                                )
                            else:
                                vals = (g[None, :] - cost) * inv_eps + logb[None, :]
                            vals = tl.where(mask_n[None, :] & keep, vals, -float("inf"))

                            block_max = tl.max(vals, axis=1)
                            new_m = tl.maximum(m_i, block_max)
                            new_m_neg_inf = new_m == -float("inf")
                            if USE_EXP2:
                                alpha_m = tl.where(
                                    new_m_neg_inf, 0.0, tl.exp2(m_i - new_m)
                                )
                                w = tl.where(
                                    new_m_neg_inf[:, None],
                                    0.0,
                                    tl.exp2(vals - new_m[:, None]),
                                )
                            else:
                                alpha_m = tl.where(
                                    new_m_neg_inf, 0.0, tl.exp(m_i - new_m)
                                )
                                w = tl.where(
                                    new_m_neg_inf[:, None],
                                    0.0,
                                    tl.exp(vals - new_m[:, None]),
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
    # g update: one program per (y_cell, block_in_cell)
    # ------------------------------------------------------------
    pid_y = pid - n_prog_x
    cell = tl.load(pid_y_cell_ptr + pid_y).to(tl.int32)
    blk = tl.load(pid_y_block_ptr + pid_y).to(tl.int32)

    start_y = tl.load(offsets_y_ptr + cell).to(tl.int32)
    end_y = tl.load(offsets_y_ptr + cell + 1).to(tl.int32)
    count_y = end_y - start_y

    offs_n = start_y + blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (start_y + count_y)
    mask_n = mask_n & (offs_n < m)

    g_old = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    if D == 1:
        cx = cell
        for dx_idx in range(0, 2 * rad + 1):
            dx = dx_idx - rad
            nbcx = cx + dx
            ok = (nbcx >= 0) & (nbcx < sx)
            nb = tl.where(ok, nbcx, 0).to(tl.int32)

            start_x = tl.load(offsets_x_ptr + nb).to(tl.int32)
            end_x = tl.load(offsets_x_ptr + nb + 1).to(tl.int32)
            count_x = tl.where(ok, end_x - start_x, 0).to(tl.int32)

            for bx in range(0, max_blocks_x):
                offs_m = start_x + bx * BLOCK_M + tl.arange(0, BLOCK_M)
                mask_m = offs_m < (start_x + count_x)
                mask_m = mask_m & (offs_m < n)

                f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
                loga = tl.load(
                    loga_ptr + offs_m * stride_loga, mask=mask_m, other=-float("inf")
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
                keep = cost <= radius2

                if USE_EXP2:
                    vals = tl.fma(f[:, None] - cost, inv_eps_log2, loga[:, None])
                else:
                    vals = (f[:, None] - cost) * inv_eps + loga[:, None]
                vals = tl.where(mask_m[:, None] & keep, vals, -float("inf"))

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
    elif D == 2:
        cx = cell % sx
        cy = cell // sx
        for dy_idx in range(0, 2 * rad + 1):
            dy = dy_idx - rad
            for dx_idx in range(0, 2 * rad + 1):
                dx = dx_idx - rad
                nbcx = cx + dx
                nbcy = cy + dy
                ok = (nbcx >= 0) & (nbcx < sx) & (nbcy >= 0) & (nbcy < sy)
                nb = nbcx + sx * nbcy
                nb = tl.where(ok, nb, 0).to(tl.int32)

                start_x = tl.load(offsets_x_ptr + nb).to(tl.int32)
                end_x = tl.load(offsets_x_ptr + nb + 1).to(tl.int32)
                count_x = tl.where(ok, end_x - start_x, 0).to(tl.int32)

                for bx in range(0, max_blocks_x):
                    offs_m = start_x + bx * BLOCK_M + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < (start_x + count_x)
                    mask_m = mask_m & (offs_m < n)

                    f = tl.load(
                        f_ptr + offs_m * stride_f, mask=mask_m, other=0.0
                    ).to(tl.float32)
                    loga = tl.load(
                        loga_ptr + offs_m * stride_loga, mask=mask_m, other=-float("inf")
                    ).to(tl.float32)
                    if USE_EXP2:
                        loga = loga * log2e
                    x2 = tl.load(
                        x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
                    ).to(tl.float32)

                    dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                    for k0 in range(0, D, BLOCK_K):
                        offs_k = k0 + tl.arange(0, BLOCK_K)
                        mask_k = offs_k < D
                        xk = tl.load(
                            x_ptr
                            + offs_m[:, None] * stride_x0
                            + offs_k[None, :] * stride_x1,
                            mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0,
                        )
                        yk = tl.load(
                            y_ptr
                            + offs_n[None, :] * stride_y0
                            + offs_k[:, None] * stride_y1,
                            mask=mask_n[None, :] & mask_k[:, None],
                            other=0.0,
                        )
                        dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

                    cost = x2[:, None] + y2[None, :] - 2.0 * dot
                    keep = cost <= radius2

                    if USE_EXP2:
                        vals = tl.fma(f[:, None] - cost, inv_eps_log2, loga[:, None])
                    else:
                        vals = (f[:, None] - cost) * inv_eps + loga[:, None]
                    vals = tl.where(mask_m[:, None] & keep, vals, -float("inf"))

                    block_max = tl.max(vals, axis=0)
                    new_m = tl.maximum(m_j, block_max)
                    new_m_neg_inf = new_m == -float("inf")
                    if USE_EXP2:
                        alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
                        w = tl.where(
                            new_m_neg_inf[None, :],
                            0.0,
                            tl.exp2(vals - new_m[None, :]),
                        )
                    else:
                        alpha_m = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
                        w = tl.where(
                            new_m_neg_inf[None, :],
                            0.0,
                            tl.exp(vals - new_m[None, :]),
                        )

                    s_j = s_j * alpha_m + tl.sum(w, axis=0)
                    m_j = new_m
    else:
        stride_xy = sx * sy
        cz = cell // stride_xy
        rem = cell - cz * stride_xy
        cy = rem // sx
        cx = rem - cy * sx
        for dz_idx in range(0, 2 * rad + 1):
            dz = dz_idx - rad
            for dy_idx in range(0, 2 * rad + 1):
                dy = dy_idx - rad
                for dx_idx in range(0, 2 * rad + 1):
                    dx = dx_idx - rad
                    nbcx = cx + dx
                    nbcy = cy + dy
                    nbcz = cz + dz
                    ok = (
                        (nbcx >= 0)
                        & (nbcx < sx)
                        & (nbcy >= 0)
                        & (nbcy < sy)
                        & (nbcz >= 0)
                        & (nbcz < sz)
                    )
                    nb = nbcx + sx * nbcy + stride_xy * nbcz
                    nb = tl.where(ok, nb, 0).to(tl.int32)

                    start_x = tl.load(offsets_x_ptr + nb).to(tl.int32)
                    end_x = tl.load(offsets_x_ptr + nb + 1).to(tl.int32)
                    count_x = tl.where(ok, end_x - start_x, 0).to(tl.int32)

                    for bx in range(0, max_blocks_x):
                        offs_m = start_x + bx * BLOCK_M + tl.arange(0, BLOCK_M)
                        mask_m = offs_m < (start_x + count_x)
                        mask_m = mask_m & (offs_m < n)

                        f = tl.load(
                            f_ptr + offs_m * stride_f, mask=mask_m, other=0.0
                        ).to(tl.float32)
                        loga = tl.load(
                            loga_ptr + offs_m * stride_loga,
                            mask=mask_m,
                            other=-float("inf"),
                        ).to(tl.float32)
                        if USE_EXP2:
                            loga = loga * log2e
                        x2 = tl.load(
                            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
                        ).to(tl.float32)

                        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                        for k0 in range(0, D, BLOCK_K):
                            offs_k = k0 + tl.arange(0, BLOCK_K)
                            mask_k = offs_k < D
                            xk = tl.load(
                                x_ptr
                                + offs_m[:, None] * stride_x0
                                + offs_k[None, :] * stride_x1,
                                mask=mask_m[:, None] & mask_k[None, :],
                                other=0.0,
                            )
                            yk = tl.load(
                                y_ptr
                                + offs_n[None, :] * stride_y0
                                + offs_k[:, None] * stride_y1,
                                mask=mask_n[None, :] & mask_k[:, None],
                                other=0.0,
                            )
                            dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

                        cost = x2[:, None] + y2[None, :] - 2.0 * dot
                        keep = cost <= radius2

                        if USE_EXP2:
                            vals = tl.fma(
                                f[:, None] - cost, inv_eps_log2, loga[:, None]
                            )
                        else:
                            vals = (f[:, None] - cost) * inv_eps + loga[:, None]
                        vals = tl.where(mask_m[:, None] & keep, vals, -float("inf"))

                        block_max = tl.max(vals, axis=0)
                        new_m = tl.maximum(m_j, block_max)
                        new_m_neg_inf = new_m == -float("inf")
                        if USE_EXP2:
                            alpha_m = tl.where(
                                new_m_neg_inf, 0.0, tl.exp2(m_j - new_m)
                            )
                            w = tl.where(
                                new_m_neg_inf[None, :],
                                0.0,
                                tl.exp2(vals - new_m[None, :]),
                            )
                        else:
                            alpha_m = tl.where(
                                new_m_neg_inf, 0.0, tl.exp(m_j - new_m)
                            )
                            w = tl.where(
                                new_m_neg_inf[None, :],
                                0.0,
                                tl.exp(vals - new_m[None, :]),
                            )

                        s_j = s_j * alpha_m + tl.sum(w, axis=0)
                        m_j = new_m

    valid = (m_j != -float("inf")) & (s_j > 0)
    lse = (m_j + tl.log2(s_j)) * ln2 if USE_EXP2 else (m_j + tl.log(s_j))
    cand = -eps * lse
    cand = tl.where(valid, cand, g_old)
    g_new = (1.0 - alpha) * g_old + alpha * cand
    tl.store(g_out_ptr + offs_n * stride_g_out, g_new, mask=mask_n)


def sinkhorn_geomloss_multiscale_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float = 0.05,
    scaling: float = 0.5,
    use_epsilon_scaling: bool = True,
    last_extrapolation: bool = True,
    allow_tf32: bool = True,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    diameter: Optional[float] = None,
    eps_list: Optional[Sequence[float]] = None,
    truncate: float = 5.0,
    cluster_scale: Optional[float] = None,
    max_coarse_levels: int = 1,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    use_exp2: bool = True,
    autotune: bool = True,
    blocksparse_backend: str = "auto",
    return_prelast: bool = False,
    return_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if d not in (1, 2, 3):
        raise ValueError("Multiscale backend is only supported for d in {1,2,3}.")
    if a.shape[0] != n or b.shape[0] != m:
        raise ValueError("a and b shapes must match x and y.")
    if a.device != x.device or b.device != x.device:
        raise ValueError("a and b must be on the same device as x and y.")
    if truncate <= 0:
        raise ValueError("truncate must be > 0 for the multiscale blocksparse backend.")
    if max_coarse_levels < 1:
        raise ValueError("max_coarse_levels must be >= 1.")
    if blocksparse_backend not in (
        "auto",
        "padded",
        "ranges_atomic",
        "taskcsr",
        "taskcsr_bucketed",
    ):
        raise ValueError(
            "blocksparse_backend must be one of "
            "{'auto','padded','ranges_atomic','taskcsr','taskcsr_bucketed'}."
        )

    if eps_list is None:
        if use_epsilon_scaling:
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = epsilon_schedule(diameter, blur, scaling, p=2.0)
        else:
            if eps is None or n_iters is None:
                raise ValueError(
                    "When use_epsilon_scaling=False, provide eps and n_iters."
                )
            eps_list = [float(eps)] * int(n_iters)

    if len(eps_list) == 0:
        raise ValueError("eps_list must be non-empty.")
    if n_iters is not None:
        eps_list = list(eps_list)[: int(n_iters)]
        if len(eps_list) == 0:
            raise ValueError("n_iters is 0 after slicing eps_list.")

    if diameter is None:
        diameter = max_diameter(x, y)
    if cluster_scale is None:
        cluster_scale = _cluster_scale_from_diameter(float(diameter), d)
    if cluster_scale <= 0:
        raise ValueError("cluster_scale must be > 0.")

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = bool(autotune) and not manual_blocks

    if block_m is None or block_n is None or block_k is None or num_warps is None:
        bm, bn, bk, nw = _multiscale_default_block_sizes(d, x.dtype, allow_tf32)
        bm = bm if block_m is None else block_m
        bn = bn if block_n is None else block_n
        bk = bk if block_k is None else block_k
        nw = nw if num_warps is None else num_warps
    else:
        bm, bn, bk, nw = block_m, block_n, block_k, num_warps
    if bk < 16:
        bk = 16

    # ------------------------------------------------------------------
    # Build the base (GeomLoss-style) voxel clustering at `cluster_scale`.
    # ------------------------------------------------------------------
    x_base = _voxel_clusterize(a, x, scale=float(cluster_scale))
    y_base = _voxel_clusterize(b, y, scale=float(cluster_scale))

    fine_perm_x = x_base.perm_fine
    fine_perm_y = y_base.perm_fine
    offsets_x = x_base.offsets
    offsets_y = y_base.offsets

    x_fine = x_base.x_fine_sorted.contiguous()
    y_fine = y_base.x_fine_sorted.contiguous()
    a_fine = x_base.w_fine_sorted.float().contiguous()
    b_fine = y_base.w_fine_sorted.float().contiguous()

    # ------------------------------------------------------------------
    # Build a true multi-level hierarchy by coarsening the base clusters.
    # ------------------------------------------------------------------
    levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    level_x = x_base.x_coarse.contiguous()
    level_y = y_base.x_coarse.contiguous()
    level_a = x_base.w_coarse.contiguous()
    level_b = y_base.w_coarse.contiguous()
    level_scale = float(cluster_scale)
    levels.append((level_x, level_y, level_a, level_b, level_scale))

    for _ in range(1, int(max_coarse_levels)):
        next_scale = float(level_scale) * 2.0
        if next_scale > float(diameter):
            break
        level_x, level_a = _voxel_cluster_centroids(level_a, level_x, scale=next_scale)
        level_y, level_b = _voxel_cluster_centroids(level_b, level_y, scale=next_scale)
        level_scale = next_scale
        levels.append((level_x.contiguous(), level_y.contiguous(), level_a.contiguous(), level_b.contiguous(), level_scale))

    # Coarsest -> ... -> base.
    levels = list(reversed(levels))
    jump_indices = [
        _jump_index_for_scale(eps_list, scale2=float(scale) * float(scale))
        for (_, _, _, _, scale) in levels
    ]

    # Fine level comes after the base level.
    base_jump_to_fine = jump_indices[-1]

    # ------------------------------------------------------------------
    # Helper: dense symmetric step at a given level.
    # ------------------------------------------------------------------
    def _dense_step(
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        loga_l: torch.Tensor,
        logb_l: torch.Tensor,
        x2_l: torch.Tensor,
        y2_l: torch.Tensor,
        f_in: torch.Tensor,
        g_in: torch.Tensor,
        f_out: torch.Tensor,
        g_out: torch.Tensor,
        *,
        step_eps: float,
        alpha: float,
    ) -> None:
        if x_l.dtype == torch.float16:
            dtype_id_l = 0
        elif x_l.dtype == torch.bfloat16:
            dtype_id_l = 1
        elif x_l.dtype == torch.float32:
            dtype_id_l = 2
        else:
            raise ValueError(f"Unsupported dtype for x/y at this level: {x_l.dtype}")

        blocks_f = triton.cdiv(x_l.shape[0], bm)
        blocks_g = triton.cdiv(y_l.shape[0], bn)
        grid_1d = (blocks_f + blocks_g,)
        _dense_step_kernel[grid_1d](
            x_l,
            y_l,
            f_in,
            g_in,
            loga_l,
            logb_l,
            x2_l,
            y2_l,
            f_out,
            g_out,
            x_l.shape[0],
            y_l.shape[0],
            x_l.stride(0),
            x_l.stride(1),
            y_l.stride(0),
            y_l.stride(1),
            f_in.stride(0),
            g_in.stride(0),
            loga_l.stride(0),
            logb_l.stride(0),
            x2_l.stride(0),
            y2_l.stride(0),
            f_out.stride(0),
            g_out.stride(0),
            float(step_eps),
            float(alpha),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id_l,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            USE_EXP2=use_exp2,
            num_warps=nw,
            num_stages=num_stages,
        )

    # ------------------------------------------------------------------
    # Run the multi-level loop (coarse levels -> blocksparse fine).
    # ------------------------------------------------------------------
    level_idx = 0
    x_l, y_l, a_l, b_l, scale_l = levels[level_idx]
    loga_l = log_weights(a_l).contiguous()
    logb_l = log_weights(b_l).contiguous()
    x2_l = (x_l.float() * x_l.float()).sum(dim=1).contiguous()
    y2_l = (y_l.float() * y_l.float()).sum(dim=1).contiguous()

    f0 = torch.zeros((x_l.shape[0],), device=x.device, dtype=torch.float32)
    g0 = torch.zeros((y_l.shape[0],), device=x.device, dtype=torch.float32)
    f1 = torch.empty_like(f0)
    g1 = torch.empty_like(g0)

    # Fine-level blocksparse state (filled when we jump).
    row_ptr_x: Optional[torch.Tensor] = None
    col_idx_x: Optional[torch.Tensor] = None
    row_ptr_y: Optional[torch.Tensor] = None
    col_idx_y: Optional[torch.Tensor] = None
    pid_x_cluster: Optional[torch.Tensor] = None
    pid_x_block: Optional[torch.Tensor] = None
    pid_y_cluster: Optional[torch.Tensor] = None
    pid_y_block: Optional[torch.Tensor] = None
    max_blocks_x: Optional[int] = None
    max_blocks_y: Optional[int] = None
    max_deg_x: Optional[int] = None
    max_deg_y: Optional[int] = None
    tasks_f: Optional[BlockSparseTasks] = None
    tasks_g: Optional[BlockSparseTasks] = None
    taskcsr_x: Optional[BlockSparseTaskCSR] = None
    taskcsr_y: Optional[BlockSparseTaskCSR] = None
    taskcsr_buckets: Optional[BlockSparseTaskCSRBuckets] = None
    scratch_f: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    scratch_g: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    is_fine = False

    # Init at eps_list[0] (GeomLoss-style).
    _dense_step(
        x_l, y_l, loga_l, logb_l, x2_l, y2_l, f0, g0, f1, g1, step_eps=eps_list[0], alpha=1.0
    )
    f0, f1 = f1, f0
    g0, g1 = g1, g0

    for it, step_eps in enumerate(eps_list):
        if is_fine:
            assert row_ptr_x is not None and col_idx_x is not None
            assert row_ptr_y is not None and col_idx_y is not None
            if blocksparse_backend == "ranges_atomic":
                assert tasks_f is not None and tasks_g is not None
                geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic(
                    x_l,
                    y_l,
                    f0,
                    g0,
                    loga_l,
                    logb_l,
                    x2_l,
                    y2_l,
                    offsets_x=offsets_x,
                    offsets_y=offsets_y,
                    row_ptr_x=row_ptr_x,
                    col_idx_x=col_idx_x,
                    row_ptr_y=row_ptr_y,
                    col_idx_y=col_idx_y,
                    tasks_f=tasks_f,
                    tasks_g=tasks_g,
                    scratch_f=scratch_f,
                    scratch_g=scratch_g,
                    f_out=f1,
                    g_out=g1,
                    eps=float(step_eps),
                    alpha=0.5,
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    allow_tf32=allow_tf32,
                    use_exp2=use_exp2,
                )
            elif blocksparse_backend == "taskcsr":
                assert taskcsr_x is not None and taskcsr_y is not None
                geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
                    x_l,
                    y_l,
                    f0,
                    g0,
                    loga_l,
                    logb_l,
                    x2_l,
                    y2_l,
                    offsets_x=offsets_x,
                    offsets_y=offsets_y,
                    row_ptr_x=row_ptr_x,
                    col_idx_x=col_idx_x,
                    row_ptr_y=row_ptr_y,
                    col_idx_y=col_idx_y,
                    taskcsr_x=taskcsr_x,
                    taskcsr_y=taskcsr_y,
                    f_out=f1,
                    g_out=g1,
                    eps=float(step_eps),
                    alpha=0.5,
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=nw,
                    num_stages=num_stages,
                    allow_tf32=allow_tf32,
                    use_exp2=use_exp2,
                )
            elif blocksparse_backend == "taskcsr_bucketed":
                assert taskcsr_x is not None and taskcsr_y is not None
                assert taskcsr_buckets is not None
                geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed(
                    x_l,
                    y_l,
                    f0,
                    g0,
                    loga_l,
                    logb_l,
                    x2_l,
                    y2_l,
                    offsets_x=offsets_x,
                    offsets_y=offsets_y,
                    row_ptr_x=row_ptr_x,
                    col_idx_x=col_idx_x,
                    row_ptr_y=row_ptr_y,
                    col_idx_y=col_idx_y,
                    taskcsr_x=taskcsr_x,
                    taskcsr_y=taskcsr_y,
                    buckets=taskcsr_buckets,
                    f_out=f1,
                    g_out=g1,
                    eps=float(step_eps),
                    alpha=0.5,
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=nw,
                    num_stages=num_stages,
                    allow_tf32=allow_tf32,
                    use_exp2=use_exp2,
                )
            else:
                assert pid_x_cluster is not None and pid_x_block is not None
                assert pid_y_cluster is not None and pid_y_block is not None
                assert max_blocks_x is not None and max_blocks_y is not None
                assert max_deg_x is not None and max_deg_y is not None
                geomloss_blocksparse_symmetric_step_sqeuclid(
                    x_l,
                    y_l,
                    f0,
                    g0,
                    loga_l,
                    logb_l,
                    x2_l,
                    y2_l,
                    offsets_x=offsets_x,
                    offsets_y=offsets_y,
                    row_ptr_x=row_ptr_x,
                    col_idx_x=col_idx_x,
                    row_ptr_y=row_ptr_y,
                    col_idx_y=col_idx_y,
                    pid_x_cluster=pid_x_cluster,
                    pid_x_block=pid_x_block,
                    pid_y_cluster=pid_y_cluster,
                    pid_y_block=pid_y_block,
                    max_blocks_x=max_blocks_x,
                    max_blocks_y=max_blocks_y,
                    max_deg_x=max_deg_x,
                    max_deg_y=max_deg_y,
                    f_out=f1,
                    g_out=g1,
                    eps=float(step_eps),
                    alpha=0.5,
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=nw,
                    num_stages=num_stages,
                    allow_tf32=allow_tf32,
                    use_exp2=use_exp2,
                )
        else:
            _dense_step(
                x_l,
                y_l,
                loga_l,
                logb_l,
                x2_l,
                y2_l,
                f0,
                g0,
                f1,
                g1,
                step_eps=float(step_eps),
                alpha=0.5,
            )

        f0, f1 = f1, f0
        g0, g1 = g1, g0

        # Jump to the next level after this iteration if requested.
        if not is_fine and it == jump_indices[level_idx]:
            f_curr = f0
            g_curr = g0
            x_curr = x_l
            y_curr = y_l
            loga_curr = loga_l
            logb_curr = logb_l

            if level_idx < len(levels) - 1:
                # Coarse -> less coarse jump (clusters only).
                next_x, next_y, _, _, _ = levels[level_idx + 1]

                g_aug = g_curr + float(step_eps) * logb_curr
                f_zero = torch.zeros(
                    (next_x.shape[0],), device=x.device, dtype=torch.float32
                )
                f_lse, _ = apply_lse_kernel_sqeuclid(
                    next_x,
                    y_curr,
                    f_zero,
                    g_aug,
                    float(step_eps),
                    axis=1,
                    use_exp2=use_exp2,
                    allow_tf32=allow_tf32,
                )
                f_next = -f_lse

                f_aug = f_curr + float(step_eps) * loga_curr
                g_zero = torch.zeros(
                    (next_y.shape[0],), device=x.device, dtype=torch.float32
                )
                g_lse, _ = apply_lse_kernel_sqeuclid(
                    x_curr,
                    next_y,
                    f_aug,
                    g_zero,
                    float(step_eps),
                    axis=0,
                    use_exp2=use_exp2,
                    allow_tf32=allow_tf32,
                )
                g_next = -g_lse

                # Switch level.
                level_idx += 1
                x_l, y_l, a_l, b_l, _ = levels[level_idx]
                loga_l = log_weights(a_l).contiguous()
                logb_l = log_weights(b_l).contiguous()
                x2_l = (x_l.float() * x_l.float()).sum(dim=1).contiguous()
                y2_l = (y_l.float() * y_l.float()).sum(dim=1).contiguous()

                f0 = f_next
                g0 = g_next
                f1 = torch.empty_like(f0)
                g1 = torch.empty_like(g0)
            else:
                # Base clusters -> fine points (build blocksparse pattern and extrapolate).
                x_base_c = x_curr
                y_base_c = y_curr

                x2_c = (x_base_c.float() * x_base_c.float()).sum(dim=1)
                y2_c = (y_base_c.float() * y_base_c.float()).sum(dim=1)
                cost_c = x2_c[:, None] + y2_c[None, :] - 2.0 * (
                    x_base_c.float() @ y_base_c.float().T
                )
                keep = (f_curr[:, None] + g_curr[None, :]) > (
                    cost_c - float(truncate) * float(step_eps)
                )

                row_ptr_x, col_idx_x = _csr_from_keep(keep)
                row_ptr_y, col_idx_y = _csr_from_keep(keep.T)

                g_aug = g_curr + float(step_eps) * logb_curr
                f_zero = torch.zeros(
                    (x_fine.shape[0],), device=x.device, dtype=torch.float32
                )
                f_lse, _ = apply_lse_kernel_sqeuclid(
                    x_fine,
                    y_base_c,
                    f_zero,
                    g_aug,
                    float(step_eps),
                    axis=1,
                    use_exp2=use_exp2,
                    allow_tf32=allow_tf32,
                )
                f_next = -f_lse

                f_aug = f_curr + float(step_eps) * loga_curr
                g_zero = torch.zeros(
                    (y_fine.shape[0],), device=x.device, dtype=torch.float32
                )
                g_lse, _ = apply_lse_kernel_sqeuclid(
                    x_base_c,
                    y_fine,
                    f_aug,
                    g_zero,
                    float(step_eps),
                    axis=0,
                    use_exp2=use_exp2,
                    allow_tf32=allow_tf32,
                )
                g_next = -g_lse

                # Switch to fine level.
                is_fine = True
                x_l, y_l = x_fine, y_fine
                a_l, b_l = a_fine, b_fine
                loga_l = log_weights(a_l).contiguous()
                logb_l = log_weights(b_l).contiguous()
                x2_l = (x_l.float() * x_l.float()).sum(dim=1).contiguous()
                y2_l = (y_l.float() * y_l.float()).sum(dim=1).contiguous()

                f0 = f_next
                g0 = g_next
                f1 = torch.empty_like(f0)
                g1 = torch.empty_like(g0)

                # Decide + build fine-level sparse backend.
                if blocksparse_backend == "auto" and use_autotune:
                    blocksparse_backend = "taskcsr"

                if blocksparse_backend in ("taskcsr", "taskcsr_bucketed") and use_autotune:
                    key = _multiscale_autotune_key(
                        x=x_l, y=y_l, allow_tf32=allow_tf32, use_exp2=use_exp2
                    )
                    cached = _MULTISCALE_TASKCSR_AUTOTUNE_CACHE.get(key)
                    if cached is None:
                        bn_tuned, nw_tuned = _autotune_taskcsr_bn(
                            x_fine=x_l,
                            y_fine=y_l,
                            f_in=f0,
                            g_in=g0,
                            loga=loga_l,
                            logb=logb_l,
                            x2=x2_l,
                            y2=y2_l,
                            offsets_x=offsets_x,
                            offsets_y=offsets_y,
                            row_ptr_x=row_ptr_x,
                            col_idx_x=col_idx_x,
                            row_ptr_y=row_ptr_y,
                            col_idx_y=col_idx_y,
                            eps=float(step_eps),
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            bm=int(bm),
                            bk=int(bk),
                            num_stages=int(num_stages),
                        )
                        bn = int(bn_tuned)
                        nw = int(nw_tuned)
                        _MULTISCALE_TASKCSR_AUTOTUNE_CACHE[key] = (int(bm), bn, int(bk), nw)
                    else:
                        bm, bn, bk, nw = cached

                if blocksparse_backend == "ranges_atomic":
                    tasks_f = blocksparse_build_tasks_from_csr(
                        offsets_x=offsets_x,
                        offsets_y=offsets_y,
                        row_ptr_x=row_ptr_x,
                        col_idx_x=col_idx_x,
                        block_m=int(bm),
                        block_n=int(bn),
                    )
                    tasks_g = blocksparse_build_tasks_from_csr(
                        offsets_x=offsets_y,
                        offsets_y=offsets_x,
                        row_ptr_x=row_ptr_y,
                        col_idx_x=col_idx_y,
                        block_m=int(bn),
                        block_n=int(bm),
                    )
                    scratch_f = (torch.empty_like(f0), torch.empty_like(f0))
                    scratch_g = (torch.empty_like(g0), torch.empty_like(g0))
                elif blocksparse_backend in ("taskcsr", "taskcsr_bucketed"):
                    tasks = blocksparse_build_tasks_from_csr(
                        offsets_x=offsets_x,
                        offsets_y=offsets_y,
                        row_ptr_x=row_ptr_x,
                        col_idx_x=col_idx_x,
                        block_m=int(bm),
                        block_n=int(bn),
                    )
                    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
                    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")
                    if blocksparse_backend == "taskcsr_bucketed":
                        taskcsr_buckets = blocksparse_build_taskcsr_buckets(
                            taskcsr_x, taskcsr_y
                        )
                else:
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
                        block_m=int(bm),
                        block_n=int(bn),
                    )

    # ------------------------------------------------------------------
    # Final full update (last_extrapolation) on fine.
    # ------------------------------------------------------------------
    f_grad = f0
    g_grad = g0
    if last_extrapolation:
        if not is_fine:
            raise RuntimeError("Internal error: multiscale did not reach the fine level.")
        assert row_ptr_x is not None and col_idx_x is not None
        assert row_ptr_y is not None and col_idx_y is not None
        if blocksparse_backend == "ranges_atomic":
            assert tasks_f is not None and tasks_g is not None
            geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic(
                x_l,
                y_l,
                f0,
                g0,
                loga_l,
                logb_l,
                x2_l,
                y2_l,
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                row_ptr_y=row_ptr_y,
                col_idx_y=col_idx_y,
                tasks_f=tasks_f,
                tasks_g=tasks_g,
                scratch_f=scratch_f,
                scratch_g=scratch_g,
                f_out=f1,
                g_out=g1,
                eps=float(eps_list[-1]),
                alpha=1.0,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )
        elif blocksparse_backend == "taskcsr":
            assert taskcsr_x is not None and taskcsr_y is not None
            geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
                x_l,
                y_l,
                f0,
                g0,
                loga_l,
                logb_l,
                x2_l,
                y2_l,
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                row_ptr_y=row_ptr_y,
                col_idx_y=col_idx_y,
                taskcsr_x=taskcsr_x,
                taskcsr_y=taskcsr_y,
                f_out=f1,
                g_out=g1,
                eps=float(eps_list[-1]),
                alpha=1.0,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=nw,
                num_stages=num_stages,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )
        elif blocksparse_backend == "taskcsr_bucketed":
            assert taskcsr_x is not None and taskcsr_y is not None
            assert taskcsr_buckets is not None
            geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed(
                x_l,
                y_l,
                f0,
                g0,
                loga_l,
                logb_l,
                x2_l,
                y2_l,
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                row_ptr_y=row_ptr_y,
                col_idx_y=col_idx_y,
                taskcsr_x=taskcsr_x,
                taskcsr_y=taskcsr_y,
                buckets=taskcsr_buckets,
                f_out=f1,
                g_out=g1,
                eps=float(eps_list[-1]),
                alpha=1.0,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=nw,
                num_stages=num_stages,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )
        else:
            assert pid_x_cluster is not None and pid_x_block is not None
            assert pid_y_cluster is not None and pid_y_block is not None
            assert max_blocks_x is not None and max_blocks_y is not None
            assert max_deg_x is not None and max_deg_y is not None
            geomloss_blocksparse_symmetric_step_sqeuclid(
                x_l,
                y_l,
                f0,
                g0,
                loga_l,
                logb_l,
                x2_l,
                y2_l,
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                row_ptr_y=row_ptr_y,
                col_idx_y=col_idx_y,
                pid_x_cluster=pid_x_cluster,
                pid_x_block=pid_x_block,
                pid_y_cluster=pid_y_cluster,
                pid_y_block=pid_y_block,
                max_blocks_x=max_blocks_x,
                max_blocks_y=max_blocks_y,
                max_deg_x=max_deg_x,
                max_deg_y=max_deg_y,
                f_out=f1,
                g_out=g1,
                eps=float(eps_list[-1]),
                alpha=1.0,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=nw,
                num_stages=num_stages,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )
        f_cost, g_cost = f1, g1
    else:
        f_cost, g_cost = f0, g0

    # Unsort to match input ordering.
    f_cost_u = torch.empty((n,), device=x.device, dtype=torch.float32)
    g_cost_u = torch.empty((m,), device=x.device, dtype=torch.float32)
    f_cost_u[fine_perm_x] = f_cost
    g_cost_u[fine_perm_y] = g_cost

    if not return_prelast and not return_state:
        return f_cost_u, g_cost_u

    f_grad_u = torch.empty((n,), device=x.device, dtype=torch.float32)
    g_grad_u = torch.empty((m,), device=x.device, dtype=torch.float32)
    f_grad_u[fine_perm_x] = f_grad
    g_grad_u[fine_perm_y] = g_grad

    if return_state:
        if row_ptr_x is None or col_idx_x is None or row_ptr_y is None or col_idx_y is None:
            raise RuntimeError("Internal error: missing blocksparse state.")
        state = (
            fine_perm_x,
            fine_perm_y,
            offsets_x,
            offsets_y,
            row_ptr_x,
            col_idx_x,
            row_ptr_y,
            col_idx_y,
        )
        if return_prelast:
            return f_cost_u, g_cost_u, f_grad_u, g_grad_u, state  # type: ignore[return-value]
        return f_cost_u, g_cost_u, state  # type: ignore[return-value]

    return f_cost_u, g_cost_u, f_grad_u, g_grad_u
