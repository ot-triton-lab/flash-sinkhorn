import argparse
import time

import torch
import triton
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_sqeuclid import apply_lse_kernel_sqeuclid


def _bench_once(fn, warmup=25, rep=100):
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def run_bench(n, m, d, axis, dtype, configs, with_vec, *, use_exp2: bool):
    device = torch.device("cuda")
    x = torch.randn(n, d, device=device, dtype=dtype)
    y = torch.randn(m, d, device=device, dtype=dtype)
    x2 = (x.float() * x.float()).sum(dim=1)
    y2 = (y.float() * y.float()).sum(dim=1)
    f = torch.randn(n, device=device, dtype=torch.float32)
    g = torch.randn(m, device=device, dtype=torch.float32)
    eps = 0.5
    vec = None
    if with_vec:
        vec_len = m if axis == 1 else n
        vec = torch.randn(vec_len, device=device, dtype=torch.float32)

    vec_mode = "vec" if with_vec else "no-vec"
    print(f"n={n} m={m} d={d} axis={axis} dtype={dtype} {vec_mode}")
    for cfg in configs:
        block_m, block_n, block_k, num_warps = cfg

        def run():
            apply_lse_kernel_sqeuclid(
                x,
                y,
                f,
                g,
                eps,
                axis,
                vec=vec,
                x2=x2,
                y2=y2,
                use_exp2=use_exp2,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
            )

        run()
        torch.cuda.synchronize()
        ms = _bench_once(run)
        print(
            f"BLOCK_M={block_m} BLOCK_N={block_n} BLOCK_K={block_k} "
            f"warps={num_warps} time_ms={ms:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--with-vec", action="store_true")
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable FlashAttention-style exp2/log2 reduction (use exp/log).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    configs = [
        (32, 32, 32, 4),
        (64, 64, 32, 4),
        (128, 128, 32, 8),
        (64, 128, 32, 8),
    ]
    torch.cuda.synchronize()
    start = time.time()
    run_bench(
        args.n,
        args.m,
        args.d,
        args.axis,
        dtype,
        configs,
        args.with_vec,
        use_exp2=not args.no_exp2,
    )
    torch.cuda.synchronize()
    end = time.time()
    print(f"total_s={end - start:.3f}")


if __name__ == "__main__":
    main()
