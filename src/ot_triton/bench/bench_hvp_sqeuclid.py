import argparse

import torch
import triton.testing as testing

from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark streaming Sinkhorn Hessian-vector product (x-only, SqEuclid)."
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--tau2", type=float, default=1e-5)
    parser.add_argument("--max-cg-iter", type=int, default=300)
    parser.add_argument("--no-precond", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 in Triton tl.dot (strict FP32 math, slower).",
    )
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable exp2/log2 path (use exp/log).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = torch.randn(args.m, args.d, device=device, dtype=torch.float32)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(args.n, args.d, device=device, dtype=torch.float32)

    eps_list = [float(args.eps)] * int(args.n_iters)
    use_preconditioner = not args.no_precond
    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2

    # 1) Solve for prelast potentials once (compile excluded from timed region).
    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        eps_list=eps_list,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=True,
        return_prelast=True,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.warps,
        num_stages=args.stages,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=args.eps)

    def run_hvp():
        hvp_x_sqeuclid_from_potentials(
            x,
            y,
            f_hat,
            g_hat,
            A,
            eps=args.eps,
            tau2=args.tau2,
            max_cg_iter=args.max_cg_iter,
            cg_rtol=1e-6,
            cg_atol=1e-6,
            use_preconditioner=use_preconditioner,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
            num_stages=args.stages,
        )

    # 2) Warm up / compile HVP kernels.
    run_hvp()
    torch.cuda.synchronize()

    ms = testing.do_bench(run_hvp, warmup=args.warmup, rep=args.rep)

    print(
        f"n={args.n} m={args.m} d={args.d} eps={args.eps} n_iters={args.n_iters} "
        f"tau2={args.tau2} cg_max_iter={args.max_cg_iter} precond={'on' if use_preconditioner else 'off'} "
        f"tf32={'on' if allow_tf32 else 'off'} exp2={'on' if use_exp2 else 'off'}"
    )
    print(f"hvp_ms={ms:.3f}")


if __name__ == "__main__":
    main()
