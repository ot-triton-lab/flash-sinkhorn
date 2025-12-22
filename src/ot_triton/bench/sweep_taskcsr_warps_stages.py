import argparse
from dataclasses import dataclass

import torch

from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import log_weights


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


@dataclass(frozen=True)
class _FineState:
    x: torch.Tensor
    y: torch.Tensor
    loga: torch.Tensor
    logb: torch.Tensor
    x2: torch.Tensor
    y2: torch.Tensor
    offsets_x: torch.Tensor
    offsets_y: torch.Tensor
    row_ptr_x: torch.Tensor
    col_idx_x: torch.Tensor
    row_ptr_y: torch.Tensor
    col_idx_y: torch.Tensor


def _build_fine_state(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    blur: float,
    scaling: float,
    truncate: float,
    max_coarse_levels: int,
    allow_tf32: bool,
    use_exp2: bool,
) -> _FineState:
    _, _, state = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        blur=float(blur),
        scaling=float(scaling),
        truncate=float(truncate),
        max_coarse_levels=int(max_coarse_levels),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
        autotune=False,
        blocksparse_backend="taskcsr",
        return_state=True,
    )
    (
        perm_x,
        perm_y,
        offsets_x,
        offsets_y,
        row_ptr_x,
        col_idx_x,
        row_ptr_y,
        col_idx_y,
    ) = state
    x_s = x[perm_x].contiguous()
    y_s = y[perm_y].contiguous()
    a_s = a[perm_x].contiguous()
    b_s = b[perm_y].contiguous()
    loga_s = log_weights(a_s).contiguous()
    logb_s = log_weights(b_s).contiguous()
    x2_s = (x_s * x_s).sum(dim=1).contiguous()
    y2_s = (y_s * y_s).sum(dim=1).contiguous()
    return _FineState(
        x=x_s,
        y=y_s,
        loga=loga_s,
        logb=logb_s,
        x2=x2_s,
        y2=y2_s,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        row_ptr_y=row_ptr_y,
        col_idx_y=col_idx_y,
    )


def _parse_int_list(spec: str) -> list[int]:
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("Empty integer list.")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep num_warps/num_stages for the multiscale task-CSR fine step."
    )
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--m", type=int, default=100_000)
    parser.add_argument("--d", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--blur", type=float, default=0.05)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--truncate", type=float, default=5.0)
    parser.add_argument("--max-coarse-levels", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--warps", type=str, default="1,2,4,8")
    parser.add_argument("--stages", type=str, default="1,2,3,4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    x = torch.randn(int(args.n), int(args.d), device=device, dtype=torch.float32)
    y = torch.randn(int(args.m), int(args.d), device=device, dtype=torch.float32)
    a = torch.rand(int(args.n), device=device, dtype=torch.float32) + 0.1
    b = torch.rand(int(args.m), device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2

    print(
        "building_state: "
        f"n={args.n} m={args.m} d={args.d} blur={args.blur} scaling={args.scaling} "
        f"truncate={args.truncate} tf32={'on' if allow_tf32 else 'off'} "
        f"exp2={'on' if use_exp2 else 'off'}"
    )
    fine = _build_fine_state(
        x=x,
        y=y,
        a=a,
        b=b,
        blur=float(args.blur),
        scaling=float(args.scaling),
        truncate=float(args.truncate),
        max_coarse_levels=int(args.max_coarse_levels),
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
    )

    tasks = blocksparse_build_tasks_from_csr(
        offsets_x=fine.offsets_x,
        offsets_y=fine.offsets_y,
        row_ptr_x=fine.row_ptr_x,
        col_idx_x=fine.col_idx_x,
        block_m=int(args.block_m),
        block_n=int(args.block_n),
    )
    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    f = torch.zeros((int(args.n),), device=device, dtype=torch.float32)
    g = torch.zeros((int(args.m),), device=device, dtype=torch.float32)
    f_out = torch.empty_like(f)
    g_out = torch.empty_like(g)

    warps_list = _parse_int_list(args.warps)
    stages_list = _parse_int_list(args.stages)

    results = []

    def bench_one(num_warps: int, num_stages: int) -> float:
        nonlocal f, g, f_out, g_out

        def step():
            nonlocal f, g, f_out, g_out
            geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr(
                fine.x,
                fine.y,
                f,
                g,
                fine.loga,
                fine.logb,
                fine.x2,
                fine.y2,
                offsets_x=fine.offsets_x,
                offsets_y=fine.offsets_y,
                row_ptr_x=fine.row_ptr_x,
                col_idx_x=fine.col_idx_x,
                row_ptr_y=fine.row_ptr_y,
                col_idx_y=fine.col_idx_y,
                taskcsr_x=taskcsr_x,
                taskcsr_y=taskcsr_y,
                f_out=f_out,
                g_out=g_out,
                eps=float(args.eps),
                alpha=float(args.alpha),
                block_m=int(args.block_m),
                block_n=int(args.block_n),
                block_k=int(args.block_k),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )
            f, f_out = f_out, f
            g, g_out = g_out, g

        # Compile + cache this variant without counting it.
        step()
        torch.cuda.synchronize()
        return _bench_cuda_ms(step, warmup=int(args.warmup), rep=int(args.rep))

    print(
        f"sweep: block_m={args.block_m} block_n={args.block_n} block_k={args.block_k} "
        f"eps={args.eps} alpha={args.alpha} warmup={args.warmup} rep={args.rep}"
    )
    for nw in warps_list:
        for ns in stages_list:
            ms = bench_one(int(nw), int(ns))
            results.append((ms, int(nw), int(ns)))
            print(f"num_warps={int(nw):2d} num_stages={int(ns):2d} ms={ms:8.3f}")

    best_ms, best_w, best_s = min(results, key=lambda t: t[0])
    worst_ms, worst_w, worst_s = max(results, key=lambda t: t[0])
    print(
        "best: "
        f"num_warps={best_w} num_stages={best_s} ms={best_ms:.3f} "
        f"(worst {worst_ms:.3f} at warps={worst_w}, stages={worst_s})"
    )


if __name__ == "__main__":
    main()
