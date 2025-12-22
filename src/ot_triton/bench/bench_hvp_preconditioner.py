import argparse

import torch
import triton.testing as testing

from ot_triton.hvp import geomloss_to_ott_potentials, hvp_x_sqeuclid_from_potentials
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)


def _parse_ns(ns: str) -> list[int]:
    return [int(x) for x in ns.split(",") if x.strip()]


def bench_size(
    *,
    n: int,
    d: int,
    eps: float,
    n_iters: int,
    allow_tf32: bool,
    use_exp2: bool,
    warmup: int,
    rep: int,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    m = n
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(n, d, device=device, dtype=torch.float32)

    eps_list = [eps] * n_iters

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
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)

    configs = [
        ("none", 0),
        ("jacobi", 0),
        ("neumann", 1),
        ("neumann", 2),
        ("neumann", 3),
        ("neumann", 4),
    ]

    print(
        f"\n=== n={n} d={d} eps={eps} n_iters={n_iters} tf32={'on' if allow_tf32 else 'off'} ===",
        flush=True,
    )

    for mode, terms in configs:
        def run_hvp() -> None:
            hvp_x_sqeuclid_from_potentials(
                x,
                y,
                f_hat,
                g_hat,
                A,
                eps=eps,
                tau2=1e-5,
                max_cg_iter=300,
                cg_rtol=1e-6,
                cg_atol=1e-6,
                cg_stabilise_every=0,
                preconditioner=mode,
                precond_terms=terms,
                use_preconditioner=True,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
            )

        run_hvp()
        torch.cuda.synchronize()

        _, info = hvp_x_sqeuclid_from_potentials(
            x,
            y,
            f_hat,
            g_hat,
            A,
            eps=eps,
            tau2=1e-5,
            max_cg_iter=300,
            cg_rtol=1e-6,
            cg_atol=1e-6,
            cg_stabilise_every=0,
            preconditioner=mode,
            precond_terms=terms,
            use_preconditioner=True,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

        ms = testing.do_bench(run_hvp, warmup=warmup, rep=rep)
        label = mode if mode != "neumann" else f"neumann-{terms}"
        print(
            f"{label:12s} hvp_ms={ms:8.3f}  cg_iters={info.cg_iters:3d}  cg_res={info.cg_residual:.2e}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HVP preconditioners (SqEuclid)."
    )
    parser.add_argument("--ns", type=str, default="4096,8192,16384")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    ns = _parse_ns(args.ns)
    for n in ns:
        bench_size(
            n=n,
            d=args.d,
            eps=args.eps,
            n_iters=args.n_iters,
            allow_tf32=bool(args.allow_tf32),
            use_exp2=not bool(args.no_exp2),
            warmup=int(args.warmup),
            rep=int(args.rep),
        )


if __name__ == "__main__":
    main()
