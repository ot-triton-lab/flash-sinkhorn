import argparse

import torch
import triton.testing as testing

from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials_taskcsr
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multiscale (D<=3) HVP using blocksparse plan applications."
    )
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=100000)
    parser.add_argument("--d", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--truncate", type=float, default=5.0)
    parser.add_argument("--cluster-scale", type=float, default=None)
    parser.add_argument("--max-coarse-levels", type=int, default=1)
    parser.add_argument(
        "--blocksparse-backend",
        type=str,
        default="taskcsr_bucketed",
        choices=["taskcsr", "taskcsr_bucketed"],
    )
    parser.add_argument("--tau2", type=float, default=1e-5)
    parser.add_argument("--max-cg-iter", type=int, default=300)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-atol", type=float, default=1e-6)
    parser.add_argument("--no-precond", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warps", type=int, default=1)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")

    x = torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = torch.randn(args.m, args.d, device=device, dtype=torch.float32)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(args.n, args.d, device=device, dtype=torch.float32)

    eps_list = [float(args.eps)] * int(args.n_iters)
    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2
    use_preconditioner = not args.no_precond

    # Build multiscale potentials + blocksparse pattern once (excluded from timings).
    _, _, f_grad, g_grad, state = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        eps_list=eps_list,
        truncate=float(args.truncate),
        cluster_scale=None if args.cluster_scale is None else float(args.cluster_scale),
        max_coarse_levels=int(args.max_coarse_levels),
        blocksparse_backend=str(args.blocksparse_backend),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
        autotune=True,
        block_m=int(args.block_m),
        block_n=int(args.block_n),
        num_warps=int(args.warps),
        num_stages=int(args.stages),
        return_prelast=True,
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
    A_s = A[perm_x].contiguous()
    f_hat, g_hat = geomloss_to_ott_potentials(
        f_grad[perm_x].contiguous(), g_grad[perm_y].contiguous(), a_s, b_s, eps=args.eps
    )

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

    buckets = None
    if args.blocksparse_backend == "taskcsr_bucketed":
        buckets = blocksparse_build_taskcsr_buckets(taskcsr_x, taskcsr_y)

    def run_hvp():
        hvp_x_sqeuclid_from_potentials_taskcsr(
            x_s,
            y_s,
            f_hat,
            g_hat,
            A_s,
            eps=float(args.eps),
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            taskcsr_x=taskcsr_x,
            taskcsr_y=taskcsr_y,
            buckets=buckets,
            tau2=float(args.tau2),
            max_cg_iter=int(args.max_cg_iter),
            cg_rtol=float(args.cg_rtol),
            cg_atol=float(args.cg_atol),
            use_preconditioner=bool(use_preconditioner),
            block_m=int(args.block_m),
            block_n=int(args.block_n),
            num_warps=int(args.warps),
            num_stages=int(args.stages),
            use_exp2=bool(use_exp2),
        )

    run_hvp()
    torch.cuda.synchronize()
    ms = testing.do_bench(run_hvp, warmup=int(args.warmup), rep=int(args.rep))

    print(
        f"n={args.n} m={args.m} d={args.d} eps={args.eps} n_iters={args.n_iters} "
        f"truncate={args.truncate} blocksparse_backend={args.blocksparse_backend} "
        f"tau2={args.tau2} cg_max_iter={args.max_cg_iter} precond={'on' if use_preconditioner else 'off'} "
        f"tf32={'on' if allow_tf32 else 'off'} exp2={'on' if use_exp2 else 'off'}"
    )
    print(f"hvp_ms={ms:.3f}")


if __name__ == "__main__":
    main()

