import argparse
import ctypes
import os

import torch
import triton.testing as testing

from ot_triton import SamplesLoss


def _preload_cuda_libs() -> None:
    os.environ.setdefault("CUDA_HOME", "/cm/shared/apps/cuda12.1/toolkit/12.1.1")
    os.environ.setdefault("CUDA_PATH", "/cm/shared/apps/cuda12.1/toolkit/12.1.1")
    paths = [
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc.so.12",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.1",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libcudart.so.12",
    ]
    for path in paths:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ot_triton multiscale vs GeomLoss multiscale (KeOps) on GPU."
    )
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=100000)
    parser.add_argument("--d", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--blur", type=float, default=0.05)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--truncate", type=float, default=5.0)
    parser.add_argument("--cluster-scale", type=float, default=None)
    parser.add_argument("--max-coarse-levels", type=int, default=1)
    parser.add_argument(
        "--blocksparse-backend",
        type=str,
        default="auto",
        choices=["auto", "padded", "taskcsr", "taskcsr_bucketed", "ranges_atomic"],
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))

    x = torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = torch.randn(args.m, args.d, device=device, dtype=torch.float32)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2

    loss_triton = SamplesLoss(
        "sinkhorn",
        backend="multiscale",
        debias=False,
        blur=float(args.blur),
        scaling=float(args.scaling),
        truncate=float(args.truncate),
        cluster_scale=None if args.cluster_scale is None else float(args.cluster_scale),
        max_coarse_levels=int(args.max_coarse_levels),
        multiscale_blocksparse_backend=str(args.blocksparse_backend),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
    )

    _preload_cuda_libs()
    try:
        from geomloss import SamplesLoss as GeomLossSamplesLoss
        from geomloss.utils import squared_distances as geomloss_squared_distances
    except Exception as e:
        raise RuntimeError(
            "GeomLoss is required. Install: pip install geomloss pykeops"
        ) from e

    loss_geom = GeomLossSamplesLoss(
        "sinkhorn",
        p=2,
        blur=float(args.blur),
        scaling=float(args.scaling),
        truncate=float(args.truncate),
        cluster_scale=None if args.cluster_scale is None else float(args.cluster_scale),
        cost=("SqDist(X,Y)", geomloss_squared_distances),
        debias=False,
        backend="multiscale",
    )

    def run_triton():
        return loss_triton(a, x, b, y)

    def run_geomloss():
        return loss_geom(a, x, b, y)

    # Warmup: exclude Triton + KeOps compilation from timings.
    out_t = run_triton()
    out_g = run_geomloss()
    torch.cuda.synchronize()

    ms_t = testing.do_bench(run_triton, warmup=args.warmup, rep=args.rep)
    ms_g = testing.do_bench(run_geomloss, warmup=args.warmup, rep=args.rep)

    with torch.no_grad():
        val_t = float(out_t.detach().float().cpu().item())
        val_g = float(out_g.detach().float().cpu().item())
    abs_err = abs(val_t - val_g)
    rel_err = abs_err / max(1e-12, abs(val_g))

    print(
        f"n={args.n} m={args.m} d={args.d} blur={args.blur} scaling={args.scaling} "
        f"truncate={args.truncate} blocksparse_backend={args.blocksparse_backend} "
        f"tf32={'on' if allow_tf32 else 'off'} exp2={'on' if use_exp2 else 'off'}"
    )
    if args.cluster_scale is not None:
        print(f"cluster_scale={args.cluster_scale}")
    print(f"triton_multiscale_ms={ms_t:.3f}")
    print(f"geomloss_multiscale_ms={ms_g:.3f}")
    print(f"speedup_geomloss_over_triton={ms_g / ms_t:.3f}x")
    print(f"cost_triton={val_t:.6e} cost_geomloss={val_g:.6e} abs_err={abs_err:.3e} rel_err={rel_err:.3e}")


if __name__ == "__main__":
    main()
