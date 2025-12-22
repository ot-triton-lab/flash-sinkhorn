import argparse
import csv
import ctypes
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import sinkhorn_geomloss_online_potentials_sqeuclid


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


def _set_strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def _parse_sizes(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


# Kernel runners (Triton/GeomLoss).
def _make_triton_runner(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps_list: Sequence[float],
    *,
    use_exp2: bool,
) -> Callable[[], None]:
    def run() -> None:
        sinkhorn_geomloss_online_potentials_sqeuclid(
            x,
            y,
            a,
            b,
            use_epsilon_scaling=False,
            eps_list=eps_list,
            last_extrapolation=True,
            allow_tf32=False,
            autotune=True,
            use_exp2=use_exp2,
        )

    return run


def _make_geomloss_runner(
    softmin: Callable,
    a_log: torch.Tensor,
    b_log: torch.Tensor,
    c_xy: Sequence[torch.Tensor],
    c_yx: Sequence[torch.Tensor],
    eps_list: Sequence[float],
) -> Callable[[], None]:
    from geomloss.sinkhorn_divergence import sinkhorn_loop

    def run() -> None:
        sinkhorn_loop(
            softmin,
            a_log,
            b_log,
            None,
            None,
            c_xy,
            c_yx,
            eps_list,
            rho=None,
            debias=False,
            last_extrapolation=True,
        )

    return run


# Testing/plot helpers.
def _bench_once(fn: Callable[[], None], warmup: int, rep: int) -> float:
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def _bench_params(n: int) -> Dict[str, int]:
    if n <= 4096:
        return {"rep": 5, "warmup": 2}
    if n <= 16384:
        return {"rep": 3, "warmup": 2}
    return {"rep": 1, "warmup": 1}


def _plot_results(rows: List[Dict[str, float]], plot_path: str, d: int, n_iters: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: int(r["n"]))
    ns = [int(r["n"]) for r in rows]
    triton_ms = [float(r["triton_ms"]) for r in rows if "triton_ms" in r]
    triton_exp2_ms = [float(r["triton_exp2_ms"]) for r in rows if "triton_exp2_ms" in r]
    triton_exp_ms = [float(r["triton_exp_ms"]) for r in rows if "triton_exp_ms" in r]
    geomloss_ms = [float(r["geomloss_ms"]) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if triton_exp2_ms and triton_exp_ms:
        ax.plot(ns, triton_exp2_ms, marker="o", label="Triton (exp2/log2)")
        ax.plot(ns, triton_exp_ms, marker="^", label="Triton (exp/log)")
    else:
        ax.plot(ns, triton_ms, marker="o", label="Triton")
    ax.plot(ns, geomloss_ms, marker="s", label="GeomLoss (KeOps)")
    ax.set_title(f"Sinkhorn timings (eps=0.1, D={d}, fp32, n_iters={n_iters})")
    ax.set_xlabel("N = M")
    ax.set_ylabel("time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark N sweep for eps=0.1 (fp32, strict math)."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1024,2048,4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-iters", type=int, default=16)
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path("output") / "bench_sizes" / "bench_sizes_eps_fp32_eps0p1.csv"),
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=str(Path("output") / "bench_sizes" / "bench_sizes_eps_fp32_eps0p1.png"),
    )
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable FlashAttention-style exp2/log2 reduction (use exp/log).",
    )
    parser.add_argument(
        "--compare-exp2",
        action="store_true",
        help="Benchmark Triton with exp2/log2 and exp/log in the same run.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _preload_cuda_libs()
    _set_strict_fp32()

    from geomloss.sinkhorn_divergence import log_weights
    from geomloss.sinkhorn_samples import lse_genred, softmin_online

    Path(args.csv).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.plot).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    sizes = _parse_sizes(args.sizes)
    eps = 0.1
    n_iters = int(args.n_iters)
    d = int(args.d)

    device = torch.device("cuda")
    torch.manual_seed(0)

    my_lse = lse_genred("SqDist(X,Y)", d)
    softmin = partial(softmin_online, log_conv=my_lse)

    results: List[Dict[str, float]] = []

    for n in sizes:
        m = n
        x = torch.randn(n, d, device=device, dtype=torch.float32)
        y = torch.randn(m, d, device=device, dtype=torch.float32)
        a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
        b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
        a = a / a.sum()
        b = b / b.sum()

        a_log = log_weights(a)
        b_log = log_weights(b)
        c_xy = (x.float(), y.float().detach())
        c_yx = (y.float(), x.float().detach())
        eps_list = [eps] * n_iters

        if args.compare_exp2:
            triton_run_exp2 = _make_triton_runner(x, y, a, b, eps_list, use_exp2=True)
            triton_run_exp = _make_triton_runner(x, y, a, b, eps_list, use_exp2=False)
        else:
            triton_run = _make_triton_runner(
                x, y, a, b, eps_list, use_exp2=not args.no_exp2
            )
        geomloss_run = _make_geomloss_runner(softmin, a_log, b_log, c_xy, c_yx, eps_list)

        params = _bench_params(n)

        if args.compare_exp2:
            triton_run_exp2()
            triton_run_exp()
        else:
            triton_run()
        geomloss_run()
        torch.cuda.synchronize()

        if args.compare_exp2:
            ms_triton_exp2 = _bench_once(
                triton_run_exp2, warmup=params["warmup"], rep=params["rep"]
            )
            ms_triton_exp = _bench_once(
                triton_run_exp, warmup=params["warmup"], rep=params["rep"]
            )
        else:
            ms_triton = _bench_once(
                triton_run, warmup=params["warmup"], rep=params["rep"]
            )
        ms_geomloss = _bench_once(
            geomloss_run, warmup=params["warmup"], rep=params["rep"]
        )

        row: Dict[str, float] = {
            "n": n,
            "m": m,
            "d": d,
            "eps": eps,
            "n_iters": n_iters,
            "geomloss_ms": ms_geomloss,
            "rep": params["rep"],
            "warmup": params["warmup"],
        }
        if args.compare_exp2:
            row["triton_exp2_ms"] = ms_triton_exp2
            row["triton_exp_ms"] = ms_triton_exp
        else:
            row["triton_ms"] = ms_triton
        results.append(row)

        if args.compare_exp2:
            print(
                f"n={n} eps={eps} rep={params['rep']} "
                f"triton_exp2_ms={ms_triton_exp2:.3f} triton_exp_ms={ms_triton_exp:.3f} "
                f"geomloss_ms={ms_geomloss:.3f}"
            )
        else:
            print(
                f"n={n} eps={eps} rep={params['rep']} "
                f"triton_ms={ms_triton:.3f} geomloss_ms={ms_geomloss:.3f}"
            )

        del x, y, a, b, a_log, b_log, c_xy, c_yx
        torch.cuda.empty_cache()

    fieldnames = [
        "n",
        "m",
        "d",
        "eps",
        "n_iters",
        "geomloss_ms",
        "rep",
        "warmup",
    ]
    if args.compare_exp2:
        fieldnames.insert(fieldnames.index("geomloss_ms"), "triton_exp2_ms")
        fieldnames.insert(fieldnames.index("geomloss_ms"), "triton_exp_ms")
    else:
        fieldnames.insert(fieldnames.index("geomloss_ms"), "triton_ms")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    _plot_results(results, args.plot, d, n_iters)
    print(f"Saved CSV: {args.csv}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
