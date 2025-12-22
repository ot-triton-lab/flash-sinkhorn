import argparse

import torch
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_sqeuclid import apply_lse_kernel_sqeuclid


def _sqeuclid_cost(x, y):
    x_f = x.float()
    y_f = y.float()
    x2 = (x_f * x_f).sum(dim=1, keepdim=True)
    y2 = (y_f * y_f).sum(dim=1, keepdim=True).transpose(0, 1)
    return x2 + y2 - 2.0 * x_f @ y_f.transpose(0, 1)


def _bench_once(fn, warmup=25, rep=100):
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--with-vec", action="store_true")
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable FlashAttention-style exp2/log2 reduction (use exp/log).",
    )
    parser.add_argument(
        "--include-cost",
        action="store_true",
        help="Include materializing the n×m cost matrix inside the timed region.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    try:
        from geomloss.sinkhorn_samples import softmin_tensorized
    except Exception as e:
        raise RuntimeError(
            "geomloss is required (pip install geomloss)."
        ) from e

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")

    torch.manual_seed(0)
    x = torch.randn(args.n, args.d, device=device, dtype=dtype)
    y = torch.randn(args.m, args.d, device=device, dtype=dtype)
    x2 = (x.float() * x.float()).sum(dim=1)
    y2 = (y.float() * y.float()).sum(dim=1)
    f = torch.randn(args.n, device=device, dtype=torch.float32)
    g = torch.randn(args.m, device=device, dtype=torch.float32)

    vec = None
    if args.with_vec:
        vec_len = args.m if args.axis == 1 else args.n
        vec = torch.rand(vec_len, device=device, dtype=torch.float32) + 0.1

    if args.axis == 1:
        pot = g
        vec_len = args.m
    else:
        pot = f
        vec_len = args.n

    if vec is None:
        h = pot / args.eps
    else:
        h = vec.log() + pot / args.eps

    if not args.include_cost:
        C_xy = _sqeuclid_cost(x, y).contiguous()
        C_yx = C_xy.transpose(0, 1).contiguous()
    else:
        C_xy = None
        C_yx = None

    def run_triton():
        apply_lse_kernel_sqeuclid(
            x,
            y,
            f,
            g,
            args.eps,
            axis=args.axis,
            vec=vec,
            x2=x2,
            y2=y2,
            use_exp2=not args.no_exp2,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
        )

    def run_geomloss():
        if args.include_cost:
            cost = _sqeuclid_cost(x, y)
            cost = cost.transpose(0, 1).contiguous() if args.axis == 0 else cost
        else:
            cost = C_yx if args.axis == 0 else C_xy

        out = -softmin_tensorized(args.eps, cost.unsqueeze(0), h.unsqueeze(0))
        return out

    run_triton()
    run_geomloss()
    torch.cuda.synchronize()

    ms_triton = _bench_once(run_triton)
    ms_geomloss = _bench_once(run_geomloss)

    vec_mode = "vec" if args.with_vec else "no-vec"
    cost_mode = "include-cost" if args.include_cost else "precomputed-cost"
    print(
        f"n={args.n} m={args.m} d={args.d} axis={args.axis} eps={args.eps} "
        f"dtype={args.dtype} {vec_mode} {cost_mode}"
    )
    print(f"triton_apply_lse_kernel_ms={ms_triton:.3f}")
    print(f"geomloss_softmin_tensorized_ms={ms_geomloss:.3f}")


if __name__ == "__main__":
    main()
