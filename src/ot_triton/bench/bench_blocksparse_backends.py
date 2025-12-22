import argparse

import torch
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_sqeuclid import (
    blocksparse_prepare_metadata,
    geomloss_blocksparse_symmetric_step_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import log_weights


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty((counts.numel() + 1,), device=counts.device, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
    return offsets


def _csr_from_keep(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    idx = torch.nonzero(keep, as_tuple=False)
    keys = (idx[:, 0].to(torch.int64) * int(cols) + idx[:, 1].to(torch.int64))
    order = torch.argsort(keys)
    col_idx = idx[order, 1].to(torch.int32).contiguous()
    return row_ptr, col_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark blocksparse symmetric Sinkhorn step backends (padded/taskcsr/ranges_atomic)."
    )
    parser.add_argument("--n", type=int, default=262144)
    parser.add_argument("--m", type=int, default=262144)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=1024)
    parser.add_argument(
        "--band",
        type=int,
        default=2,
        help="Cluster-neighborhood half-width (deg ~ 2*band+1).",
    )
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument("--no-check", action="store_true", help="Skip parity checks.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = torch.randn(args.m, args.d, device=device, dtype=torch.float32)

    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    loga = log_weights(a).contiguous()
    logb = log_weights(b).contiguous()
    x2 = (x * x).sum(dim=1).contiguous()
    y2 = (y * y).sum(dim=1).contiguous()

    cx = max(1, min(int(args.clusters), int(args.n)))
    cy = max(1, min(int(args.clusters), int(args.m)))
    sx = _ceil_div(args.n, cx)
    sy = _ceil_div(args.m, cy)
    counts_x = torch.full((cx,), sx, device=device, dtype=torch.int32)
    counts_y = torch.full((cy,), sy, device=device, dtype=torch.int32)
    counts_x[-1] = int(args.n) - int(sx) * (cx - 1)
    counts_y[-1] = int(args.m) - int(sy) * (cy - 1)
    offsets_x = _offsets_from_counts(counts_x)
    offsets_y = _offsets_from_counts(counts_y)

    jy = torch.arange(cy, device=device, dtype=torch.int64).view(1, cy)
    j0 = (torch.arange(cx, device=device, dtype=torch.int64) * int(cy) // int(cx)).view(cx, 1)
    band = int(args.band)
    keep = (jy >= (j0 - band)) & (jy <= (j0 + band))
    row_ptr_x, col_idx_x = _csr_from_keep(keep)
    row_ptr_y, col_idx_y = _csr_from_keep(keep.t())

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
        block_m=int(args.block_m),
        block_n=int(args.block_n),
    )

    tasks_f = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        block_m=int(args.block_m),
        block_n=int(args.block_n),
    )
    tasks_g = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_y,
        offsets_y=offsets_x,
        row_ptr_x=row_ptr_y,
        col_idx_x=col_idx_y,
        block_m=int(args.block_n),
        block_n=int(args.block_m),
    )
    scratch_f = (torch.empty((args.n,), device=device, dtype=torch.float32), torch.empty((args.n,), device=device, dtype=torch.float32))
    scratch_g = (torch.empty((args.m,), device=device, dtype=torch.float32), torch.empty((args.m,), device=device, dtype=torch.float32))

    tasks = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        block_m=int(args.block_m),
        block_n=int(args.block_n),
    )
    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    f = torch.zeros((args.n,), device=device, dtype=torch.float32)
    g = torch.zeros((args.m,), device=device, dtype=torch.float32)
    f_out = torch.empty_like(f)
    g_out = torch.empty_like(g)

    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2

    def run_padded():
        geomloss_blocksparse_symmetric_step_sqeuclid(
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
            pid_x_cluster=pid_x_cluster,
            pid_x_block=pid_x_block,
            pid_y_cluster=pid_y_cluster,
            pid_y_block=pid_y_block,
            max_blocks_x=max_blocks_x,
            max_blocks_y=max_blocks_y,
            max_deg_x=max_deg_x,
            max_deg_y=max_deg_y,
            f_out=f_out,
            g_out=g_out,
            eps=float(args.eps),
            alpha=float(args.alpha),
            block_m=int(args.block_m),
            block_n=int(args.block_n),
            block_k=int(args.block_k),
            num_warps=int(args.warps),
            num_stages=int(args.stages),
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

    def run_taskcsr():
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
            taskcsr_x=taskcsr_x,
            taskcsr_y=taskcsr_y,
            f_out=f_out,
            g_out=g_out,
            eps=float(args.eps),
            alpha=float(args.alpha),
            block_m=int(args.block_m),
            block_n=int(args.block_n),
            block_k=int(args.block_k),
            num_warps=int(args.warps),
            num_stages=int(args.stages),
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

    def run_ranges_atomic():
        geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic(
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
            tasks_f=tasks_f,
            tasks_g=tasks_g,
            scratch_f=scratch_f,
            scratch_g=scratch_g,
            f_out=f_out,
            g_out=g_out,
            eps=float(args.eps),
            alpha=float(args.alpha),
            block_m=int(args.block_m),
            block_n=int(args.block_n),
            block_k=int(args.block_k),
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

    # Pre-compile outside timed region.
    run_padded()
    run_taskcsr()
    run_ranges_atomic()
    torch.cuda.synchronize()

    ms_padded = testing.do_bench(run_padded, warmup=args.warmup, rep=args.rep)
    ms_taskcsr = testing.do_bench(run_taskcsr, warmup=args.warmup, rep=args.rep)
    ms_atomic = testing.do_bench(run_ranges_atomic, warmup=args.warmup, rep=args.rep)

    print(
        f"n={args.n} m={args.m} d={args.d} eps={args.eps} alpha={args.alpha} "
        f"clusters=({cx},{cy}) band={band} "
        f"block=({args.block_m},{args.block_n},{args.block_k}) "
        f"tf32={'on' if allow_tf32 else 'off'} exp2={'on' if use_exp2 else 'off'}"
    )
    print(
        f"padded: n_prog=({int(pid_x_cluster.numel())},{int(pid_y_cluster.numel())}) "
        f"max_blocks=({max_blocks_x},{max_blocks_y}) max_deg=({max_deg_x},{max_deg_y})"
    )
    print(
        f"taskcsr: max_tasks=({taskcsr_x.max_tasks},{taskcsr_y.max_tasks}) "
        f"ranges_atomic: tasks=({int(tasks_f.x_cluster.numel())},{int(tasks_g.x_cluster.numel())})"
    )
    print(f"padded_ms={ms_padded:.3f}")
    print(f"taskcsr_ms={ms_taskcsr:.3f}")
    print(f"ranges_atomic_ms={ms_atomic:.3f}")

    if not args.no_check:
        run_padded()
        f_pad = f_out.clone()
        g_pad = g_out.clone()

        run_taskcsr()
        torch.testing.assert_close(f_out, f_pad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g_out, g_pad, rtol=1e-4, atol=1e-4)

        run_ranges_atomic()
        torch.testing.assert_close(f_out, f_pad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g_out, g_pad, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    main()
