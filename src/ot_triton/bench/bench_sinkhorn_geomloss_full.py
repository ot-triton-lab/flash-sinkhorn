import argparse
import ctypes
import os
from functools import partial

import torch
import triton
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    _default_block_sizes,
    _geomloss_symmetric_step_sqeuclid,
    log_weights as triton_log_weights,
    max_diameter,
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import epsilon_schedule as triton_epsilon_schedule


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


def _sqdist_cost_full(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    y_f = y.float()
    x2 = (x_f * x_f).sum(dim=-1, keepdim=True)
    y2 = (y_f * y_f).sum(dim=-1, keepdim=True).transpose(-2, -1)
    return x2 + y2 - 2.0 * torch.matmul(x_f, y_f.transpose(-2, -1))


def _bench_once(fn, warmup: int, rep: int) -> float:
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full GeomLoss-style (symmetric) Sinkhorn loop."
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--blur", type=float, default=0.05)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--no-eps-scaling", action="store_true")
    parser.add_argument(
        "--no-last-extrapolation",
        action="store_true",
        help="Disable the final alpha=1 update (GeomLoss last extrapolation).",
    )
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument(
        "--compare",
        type=str,
        default="geomloss-online",
        choices=["none", "geomloss-tensorized", "geomloss-online", "both"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--warps", type=int, default=None)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 in Triton tl.dot (strict FP32 math, slower).",
    )
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable FlashAttention-style exp2/log2 reduction (use exp/log).",
    )
    parser.add_argument(
        "--kernel-breakdown",
        action="store_true",
        help="Time init/iter/last kernel launches separately (compile excluded).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[args.dtype]

    torch.manual_seed(0)
    x = torch.randn(args.n, args.d, device=device, dtype=dtype)
    y = torch.randn(args.m, args.d, device=device, dtype=dtype)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    diameter = max_diameter(x, y)
    use_epsilon_scaling = not args.no_eps_scaling
    last_extrapolation = not args.no_last_extrapolation
    use_exp2 = not args.no_exp2

    if use_epsilon_scaling:
        eps_list = list(
            triton_epsilon_schedule(diameter, args.blur, args.scaling, p=2.0)
        )
        if args.n_iters is not None:
            eps_list = eps_list[: int(args.n_iters)]
    else:
        if args.eps is None or args.n_iters is None:
            raise ValueError("Provide --eps and --n-iters when --no-eps-scaling is set.")
        eps_list = [float(args.eps)] * int(args.n_iters)

    def run_triton():
        sinkhorn_geomloss_online_potentials_sqeuclid(
            x,
            y,
            a,
            b,
            blur=args.blur,
            scaling=args.scaling,
            use_epsilon_scaling=use_epsilon_scaling,
            eps_list=eps_list,
            last_extrapolation=last_extrapolation,
            allow_tf32=not args.no_tf32,
            use_exp2=use_exp2,
            diameter=diameter,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
            num_stages=args.stages,
        )

    # Pre-compile / warm up outside the timed region.
    run_triton()
    torch.cuda.synchronize()

    ms_triton = _bench_once(run_triton, warmup=args.warmup, rep=args.rep)

    print(
        f"n={args.n} m={args.m} d={args.d} dtype={args.dtype} "
        f"blur={args.blur} scaling={args.scaling} "
        f"eps_scaling={'on' if use_epsilon_scaling else 'off'}"
    )
    if not use_epsilon_scaling:
        print(f"eps={args.eps} n_iters={args.n_iters}")
    print(f"last_extrapolation={'on' if last_extrapolation else 'off'}")
    n_launches = 1 + len(eps_list) + (1 if last_extrapolation else 0)
    print(f"kernel_launches={n_launches}")
    print(f"triton_geomloss_full_ms={ms_triton:.3f}")

    if args.kernel_breakdown:
        loga = triton_log_weights(a).contiguous()
        logb = triton_log_weights(b).contiguous()
        x2 = (x.float() * x.float()).sum(dim=1).contiguous()
        y2 = (y.float() * y.float()).sum(dim=1).contiguous()

        bm, bn, bk, nw = _default_block_sizes(args.d)
        block_m = args.block_m or bm
        block_n = args.block_n or bn
        block_k = args.block_k or bk
        if block_k < 16:
            block_k = 16
        num_warps = args.warps or nw

        f0 = torch.zeros((args.n,), device=device, dtype=torch.float32)
        g0 = torch.zeros((args.m,), device=device, dtype=torch.float32)
        f1 = torch.empty_like(f0)
        g1 = torch.empty_like(g0)

        blocks_f = triton.cdiv(args.n, block_m)
        blocks_g = triton.cdiv(args.m, block_n)
        grid = (blocks_f + blocks_g,)

        def launch(f_in, g_in, f_out, g_out, step_eps: float, alpha: float):
            _geomloss_symmetric_step_sqeuclid[grid](
                x,
                y,
                f_in,
                g_in,
                loga,
                logb,
                x2,
                y2,
                f_out,
                g_out,
                args.n,
                args.m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f_in.stride(0),
                g_in.stride(0),
                loga.stride(0),
                logb.stride(0),
                x2.stride(0),
                y2.stride(0),
            f_out.stride(0),
            g_out.stride(0),
            float(step_eps),
            float(alpha),
            D=args.d,
            ALLOW_TF32=not args.no_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=args.stages,
            )

        # Pre-compile / warm up outside the timed region.
        launch(f0, g0, f1, g1, eps_list[0], alpha=1.0)
        torch.cuda.synchronize()

        def run_init():
            launch(f0, g0, f1, g1, eps_list[0], alpha=1.0)

        ms_init = _bench_once(run_init, warmup=args.warmup, rep=args.rep)

        launch(f0, g0, f1, g1, eps_list[0], alpha=1.0)
        torch.cuda.synchronize()

        def run_iter():
            launch(f1, g1, f0, g0, eps_list[0], alpha=0.5)

        ms_iter = _bench_once(run_iter, warmup=args.warmup, rep=args.rep)

        if last_extrapolation:
            launch(f1, g1, f0, g0, eps_list[0], alpha=0.5)
            torch.cuda.synchronize()

            def run_last():
                launch(f0, g0, f1, g1, eps_list[-1], alpha=1.0)

            ms_last = _bench_once(run_last, warmup=args.warmup, rep=args.rep)
            print(
                f"kernel_init_ms={ms_init:.3f} "
                f"kernel_iter_ms={ms_iter:.3f} "
                f"kernel_last_ms={ms_last:.3f}"
            )
        else:
            print(f"kernel_init_ms={ms_init:.3f} kernel_iter_ms={ms_iter:.3f}")

    if args.compare in ("none",):
        return

    try:
        if args.compare in ("geomloss-online", "both"):
            _preload_cuda_libs()
        geomloss = __import__("geomloss")  # noqa: F401
        from geomloss.sinkhorn_divergence import (
            log_weights,
            sinkhorn_loop,
        )
        from geomloss.sinkhorn_samples import sinkhorn_tensorized
        from geomloss.sinkhorn_samples import lse_genred, softmin_online
    except Exception as e:
        raise RuntimeError(
            "GeomLoss is required for comparison. Try: pip install geomloss (and pykeops for online)."
        ) from e

    if args.compare in ("geomloss-tensorized", "both"):
        def run_geomloss_tensorized():
            sinkhorn_tensorized(
                a.unsqueeze(0),
                x.unsqueeze(0),
                b.unsqueeze(0),
                y.unsqueeze(0),
                p=2,
                blur=args.blur,
                scaling=args.scaling,
                diameter=diameter,
                cost=_sqdist_cost_full,
                debias=False,
                potentials=True,
            )

        run_geomloss_tensorized()
        torch.cuda.synchronize()
        ms_geomloss_tensorized = _bench_once(
            run_geomloss_tensorized, warmup=args.warmup, rep=args.rep
        )
        print(f"geomloss_tensorized_full_ms={ms_geomloss_tensorized:.3f}")

    if args.compare in ("geomloss-online", "both"):
        # Avoid including KeOps "formula" construction in the timed region.
        x_keops = x.float()
        y_keops = y.float()
        a_log = log_weights(a)
        b_log = log_weights(b)
        eps_list_keops = eps_list
        my_lse = lse_genred("SqDist(X,Y)", args.d)
        softmin = partial(softmin_online, log_conv=my_lse)
        C_xy = (x_keops, y_keops.detach())
        C_yx = (y_keops, x_keops.detach())

        def run_geomloss_online():
            _, _, g_ab, f_ba = sinkhorn_loop(
                softmin,
                a_log,
                b_log,
                None,
                None,
                C_xy,
                C_yx,
                eps_list_keops,
                rho=None,
                debias=False,
                last_extrapolation=last_extrapolation,
            )
            return f_ba, g_ab

        run_geomloss_online()
        torch.cuda.synchronize()
        ms_geomloss_online = _bench_once(
            run_geomloss_online, warmup=args.warmup, rep=args.rep
        )
        print(f"geomloss_keops_online_full_ms={ms_geomloss_online:.3f}")


if __name__ == "__main__":
    main()
