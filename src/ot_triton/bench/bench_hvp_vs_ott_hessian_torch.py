import argparse
import ctypes
import os
import sys
import warnings
from typing import Callable

import torch
import triton.testing as testing

from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)


def _preload_cuda_libs() -> None:
    os.environ.setdefault("CUDA_HOME", "/cm/shared/apps/cuda12.1/toolkit/12.1.1")
    os.environ.setdefault("CUDA_PATH", "/cm/shared/apps/cuda12.1/toolkit/12.1.1")
    paths = [
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc.so.12",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.1",
        "/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib/libcudart.so.12",
    ]
    for path in paths:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def _bench_once(fn: Callable[[], None], warmup: int, rep: int) -> float:
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HVP runtime: ot_triton (streaming Triton) vs OTT-Hessian (PyTorch)."
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument(
        "--potentials",
        type=str,
        default="sinkhorn",
        choices=["sinkhorn", "zeros"],
        help=(
            "How to obtain dual potentials (f,g). 'sinkhorn' runs the fused Sinkhorn solver "
            "once; 'zeros' uses f=g=0 (useful for very large n,m)."
        ),
    )
    parser.add_argument(
        "--data-scale",
        type=float,
        default=None,
        help=(
            "Multiply x,y by this factor. If omitted and --potentials=zeros, uses "
            "sqrt(eps/(2*d)) to keep typical costs ~ eps."
        ),
    )
    parser.add_argument(
        "--compile-n",
        type=int,
        default=1024,
        help="Warmup size for compilation (avoids doing full warmup at large n).",
    )
    parser.add_argument(
        "--compile-m",
        type=int,
        default=1024,
        help="Warmup size for compilation (avoids doing full warmup at large m).",
    )
    parser.add_argument("--tau2", type=float, default=1e-5)
    parser.add_argument("--max-cg-iter", type=int, default=300)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-atol", type=float, default=1e-6)
    parser.add_argument("--no-precond", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 for dot/matmul (affects PyTorch + Triton when applicable).",
    )
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable exp2/log2 reduction in ot_triton kernels (use exp/log).",
    )
    parser.add_argument(
        "--ott-use-keops",
        action="store_true",
        help="Use KeOps transport functions in OTT-Hessian if available.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the (potentially expensive) one-shot correctness check.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    _preload_cuda_libs()
    _set_tf32(args.tf32)

    # Import OTT-Hessian torch implementation from the vendored folder.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ott_hessian_path = os.path.join(repo_root, "3rd-party", "OTT-Hessian")
    if ott_hessian_path not in sys.path:
        sys.path.insert(0, ott_hessian_path)

    from torch_sinkhorn_hessian import (  # type: ignore
        TorchOTResult,
        TorchSinkhornHessian,
        _TorchGeometry,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)

    scale = args.data_scale
    if scale is None and args.potentials == "zeros":
        scale = float((args.eps / (2.0 * args.d)) ** 0.5)
    if scale is None:
        scale = 1.0

    x = scale * torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = scale * torch.randn(args.m, args.d, device=device, dtype=torch.float32)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(args.n, args.d, device=device, dtype=torch.float32)

    eps_list = [float(args.eps)] * int(args.n_iters)

    if args.potentials == "sinkhorn":
        # Precompute potentials (GeomLoss convention) -> convert to OTT convention.
        _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
            x,
            y,
            a,
            b,
            use_epsilon_scaling=False,
            last_extrapolation=True,
            eps_list=eps_list,
            allow_tf32=args.tf32,
            use_exp2=not args.no_exp2,
            autotune=True,
            return_prelast=True,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
            num_stages=args.stages,
        )
        f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=args.eps)
    else:
        f_hat = torch.zeros((args.n,), device=device, dtype=torch.float32)
        g_hat = torch.zeros((args.m,), device=device, dtype=torch.float32)

    # Build a lightweight OTT-Hessian OT result object without materializing the plan.
    geom = _TorchGeometry(x=x, y=y, epsilon=float(args.eps))
    ot = TorchOTResult(
        geom=geom,
        a=a,
        b=b,
        matrix=torch.empty((0, 0), device=device, dtype=torch.float32),
        reg_ot_cost=torch.tensor(0.0, device=device, dtype=torch.float32),
        threshold=0.0,
        iterations=-1,
        f=f_hat,
        g=g_hat,
    )

    ott_solver = TorchSinkhornHessian(
        svd_thr=1e-10,
        device=device,
        dtype=torch.float32,
        use_compile=False,
        use_keops=bool(args.ott_use_keops),
        solver="native",
    )

    use_preconditioner = not args.no_precond

    def run_triton() -> None:
        hvp_x_sqeuclid_from_potentials(
            x,
            y,
            f_hat,
            g_hat,
            A,
            eps=args.eps,
            tau2=args.tau2,
            max_cg_iter=args.max_cg_iter,
            cg_rtol=args.cg_rtol,
            cg_atol=args.cg_atol,
            use_preconditioner=use_preconditioner,
            allow_tf32=args.tf32,
            use_exp2=not args.no_exp2,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
            num_stages=args.stages,
        )

    def run_ott_hessian() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ott_solver.hessian_vector_product(
                ot,
                A,
                tau2=args.tau2,
                max_cg_iter=args.max_cg_iter,
                cg_rtol=args.cg_rtol,
                cg_atol=args.cg_atol,
                use_preconditioner=use_preconditioner,
                return_info=False,
            )

    # Warm up compilation on a smaller problem to avoid doing a full extra run at large n,m.
    compile_n = int(args.compile_n)
    compile_m = int(args.compile_m)
    did_compile_warmup = False
    if compile_n > 0 and compile_m > 0 and (compile_n != args.n or compile_m != args.m):
        x_c = scale * torch.randn(compile_n, args.d, device=device, dtype=torch.float32)
        y_c = scale * torch.randn(compile_m, args.d, device=device, dtype=torch.float32)
        a_c = torch.ones(compile_n, device=device, dtype=torch.float32) / float(compile_n)
        b_c = torch.ones(compile_m, device=device, dtype=torch.float32) / float(compile_m)
        A_c = torch.randn(compile_n, args.d, device=device, dtype=torch.float32)
        f_c = torch.zeros((compile_n,), device=device, dtype=torch.float32)
        g_c = torch.zeros((compile_m,), device=device, dtype=torch.float32)
        if args.potentials == "sinkhorn":
            _, _, f_grad_c, g_grad_c = sinkhorn_geomloss_online_potentials_sqeuclid(
                x_c,
                y_c,
                a_c,
                b_c,
                use_epsilon_scaling=False,
                last_extrapolation=True,
                eps_list=eps_list,
                allow_tf32=args.tf32,
                use_exp2=not args.no_exp2,
                autotune=True,
                return_prelast=True,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                num_warps=args.warps,
                num_stages=args.stages,
            )
            f_c, g_c = geomloss_to_ott_potentials(f_grad_c, g_grad_c, a_c, b_c, eps=args.eps)

        geom_c = _TorchGeometry(x=x_c, y=y_c, epsilon=float(args.eps))
        ot_c = TorchOTResult(
            geom=geom_c,
            a=a_c,
            b=b_c,
            matrix=torch.empty((0, 0), device=device, dtype=torch.float32),
            reg_ot_cost=torch.tensor(0.0, device=device, dtype=torch.float32),
            threshold=0.0,
            iterations=-1,
            f=f_c,
            g=g_c,
        )

        hvp_x_sqeuclid_from_potentials(
            x_c,
            y_c,
            f_c,
            g_c,
            A_c,
            eps=args.eps,
            tau2=args.tau2,
            max_cg_iter=max(1, min(args.max_cg_iter, 2)),
            cg_rtol=args.cg_rtol,
            cg_atol=args.cg_atol,
            use_preconditioner=use_preconditioner,
            allow_tf32=args.tf32,
            use_exp2=not args.no_exp2,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.warps,
            num_stages=args.stages,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ott_solver.hessian_vector_product(
                ot_c,
                A_c,
                tau2=args.tau2,
                max_cg_iter=max(1, min(args.max_cg_iter, 2)),
                cg_rtol=args.cg_rtol,
                cg_atol=args.cg_atol,
                use_preconditioner=use_preconditioner,
                return_info=False,
            )
        did_compile_warmup = True

    # Default warmup on the actual shapes (skip if we already warmed on smaller shapes).
    if not did_compile_warmup:
        run_triton()
        run_ott_hessian()
        torch.cuda.synchronize()

    max_abs = float("nan")
    rel = float("nan")
    info_triton = None
    info_ott = None
    if not args.skip_correctness:
        # Correctness check (single run, not timed).
        with torch.no_grad():
            hvp_triton, info_triton = hvp_x_sqeuclid_from_potentials(
                x,
                y,
                f_hat,
                g_hat,
                A,
                eps=args.eps,
                tau2=args.tau2,
                max_cg_iter=args.max_cg_iter,
                cg_rtol=args.cg_rtol,
                cg_atol=args.cg_atol,
                use_preconditioner=use_preconditioner,
                allow_tf32=args.tf32,
                use_exp2=not args.no_exp2,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                num_warps=args.warps,
                num_stages=args.stages,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hvp_ott, info_ott = ott_solver.hessian_vector_product(
                    ot,
                    A,
                    tau2=args.tau2,
                    max_cg_iter=args.max_cg_iter,
                    cg_rtol=args.cg_rtol,
                    cg_atol=args.cg_atol,
                    use_preconditioner=use_preconditioner,
                    return_info=True,
                )
            max_abs = (hvp_triton - hvp_ott).abs().max().item()
            denom = hvp_ott.abs().max().item()
            rel = max_abs / (denom + 1e-12)

    ms_triton = _bench_once(run_triton, warmup=args.warmup, rep=args.rep)
    ms_ott = _bench_once(run_ott_hessian, warmup=args.warmup, rep=args.rep)

    print(
        f"n={args.n} m={args.m} d={args.d} eps={args.eps} n_iters={args.n_iters} "
        f"potentials={args.potentials} data_scale={scale:g} "
        f"tau2={args.tau2} cg_max_iter={args.max_cg_iter} precond={'on' if use_preconditioner else 'off'} "
        f"tf32={'on' if args.tf32 else 'off'} exp2={'off' if args.no_exp2 else 'on'} "
        f"ott_keops={'on' if args.ott_use_keops else 'off'}"
    )
    if not args.skip_correctness and info_triton is not None:
        print(f"max_abs_diff(hvp)={max_abs:.3e} rel_to_ott_max={rel:.3e}")
        print(
            f"cg_triton(converged={info_triton.cg_converged}, iters={info_triton.cg_iters}, resid={info_triton.cg_residual:.3e})"
        )
        if isinstance(info_ott, dict):
            print(
                f"cg_ott(converged={info_ott.get('cg_converged')}, iters={info_ott.get('cg_iters')}, resid={info_ott.get('cg_residual')})"
            )
    print(f"ot_triton_hvp_ms={ms_triton:.3f}")
    print(f"ott_hessian_torch_hvp_ms={ms_ott:.3f}")
    print(f"speedup(ott/ot_triton)={ms_ott / ms_triton:.3f}x")


if __name__ == "__main__":
    main()
