import argparse
import ctypes
import os

import torch
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_sqeuclid import apply_lse_kernel_sqeuclid


def _preload_cuda_libs():
    os.environ.setdefault(
        "CUDA_HOME", "/cm/shared/apps/cuda12.1/toolkit/12.1.1"
    )
    os.environ.setdefault(
        "CUDA_PATH", "/cm/shared/apps/cuda12.1/toolkit/12.1.1"
    )
    paths = [
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc.so.12",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.1",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libcudart.so.12",
    ]
    for path in paths:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    _preload_cuda_libs()

    try:
        from geomloss.sinkhorn_samples import lse_genred, softmin_online
    except Exception as e:
        raise RuntimeError(
            "geomloss+pykeops are required. Try: pip install geomloss pykeops"
        ) from e

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
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
        h = g / args.eps
        if vec is not None:
            h = h + vec.log()
        x_keops, y_keops = x.float(), y.float()
    else:
        h = f / args.eps
        if vec is not None:
            h = h + vec.log()
        x_keops, y_keops = y.float(), x.float()

    my_lse = lse_genred("SqDist(X,Y)", args.d)

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

    def run_keops():
        out = -softmin_online(args.eps, (x_keops, y_keops), h, log_conv=my_lse)
        return out

    run_triton()
    run_keops()
    torch.cuda.synchronize()

    ms_triton = _bench_once(run_triton)
    ms_keops = _bench_once(run_keops)

    vec_mode = "vec" if args.with_vec else "no-vec"
    keops_note = "" if dtype == torch.float32 else " (keops runs float32)"
    print(
        f"n={args.n} m={args.m} d={args.d} axis={args.axis} eps={args.eps} "
        f"dtype={args.dtype} {vec_mode}{keops_note}"
    )
    print(f"triton_apply_lse_kernel_ms={ms_triton:.3f}")
    print(f"geomloss_keops_softmin_online_ms={ms_keops:.3f}")


if __name__ == "__main__":
    main()
