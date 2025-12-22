"""
Benchmark backward (grad x,y) for online Sinkhorn:
  - ot_triton SamplesLoss (analytic gradients)
  - GeomLoss sinkhorn_loop + sinkhorn_cost (analytic gradients)

Fixed eps and n_iters to match steps: steps = n_iters + 2.
Produces small (log-log) and large (linear) plots.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import textwrap
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import triton.testing as testing

from ot_triton import SamplesLoss as TritonSamplesLoss
from ot_triton.kernels import sinkhorn_triton_grad_sqeuclid as grad_kernels


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


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def _bench_once(fn, *, warmup: int, rep: int) -> float:
    return float(testing.do_bench(fn, warmup=warmup, rep=rep))


def _make_weights(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = rng.random((n,), dtype=np.float32) + 0.1
    b = rng.random((n,), dtype=np.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    return a, b


def _run_one_size(
    *,
    n: int,
    d: int,
    eps: float,
    n_iters: int,
    allow_tf32: bool,
    use_exp2: bool,
    autotune_triton: bool,
    warmup_small: int,
    rep_small: int,
    warmup_large: int,
    rep_large: int,
    rng: np.random.Generator,
    device: torch.device,
    geomloss_truncate: int,
    dump_triton_config: bool,
) -> Dict[str, float]:
    x_np = rng.standard_normal((n, d), dtype=np.float32)
    y_np = rng.standard_normal((n, d), dtype=np.float32)
    a_np, b_np = _make_weights(rng, n)

    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
    a_t = torch.from_numpy(a_np).to(device=device, dtype=torch.float32)
    b_t = torch.from_numpy(b_np).to(device=device, dtype=torch.float32)

    eps_list = [float(eps)] * int(n_iters)
    steps = int(n_iters) + 2

    # -------- ot_triton backward --------
    x_ot = x_t.clone().requires_grad_(True)
    y_ot = y_t.clone().requires_grad_(True)

    ot_loss = TritonSamplesLoss(
        "sinkhorn",
        blur=float(np.sqrt(eps)),
        scaling=0.5,
        debias=False,
        potentials=False,
        normalize=False,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune_triton,
        eps=eps,
        n_iters=n_iters,
    )

    val_ot = ot_loss(a_t, x_ot, b_t, y_ot)

    def run_ot_backward() -> None:
        torch.autograd.grad(val_ot, (x_ot, y_ot), retain_graph=True)

    run_ot_backward()
    torch.cuda.synchronize()

    # -------- GeomLoss backward (custom fixed eps_list) --------
    from geomloss.sinkhorn_divergence import sinkhorn_loop as geomloss_sinkhorn_loop
    from geomloss.sinkhorn_divergence import sinkhorn_cost as geomloss_sinkhorn_cost
    from geomloss.sinkhorn_divergence import log_weights as geomloss_log_weights
    from geomloss.sinkhorn_samples import lse_genred, softmin_online

    x_gl = x_t.clone().requires_grad_(True)
    y_gl = y_t.clone().requires_grad_(True)

    loga = geomloss_log_weights(a_t)
    logb = geomloss_log_weights(b_t)

    my_lse = lse_genred("SqDist(X,Y)", d, dtype="float32")
    softmin = partial(softmin_online, log_conv=my_lse)

    C_xy = (x_gl, y_gl.detach())
    C_yx = (y_gl, x_gl.detach())

    f_aa, g_bb, g_ab, f_ba = geomloss_sinkhorn_loop(
        softmin,
        loga,
        logb,
        None,
        None,
        C_xy,
        C_yx,
        eps_list,
        rho=None,
        debias=False,
        last_extrapolation=True,
        truncate=int(geomloss_truncate),
        cost="SqDist(X,Y)",
    )

    val_gl = geomloss_sinkhorn_cost(
        eps,
        None,
        a_t,
        b_t,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=False,
        debias=False,
        potentials=False,
    )

    def run_gl_backward() -> None:
        torch.autograd.grad(val_gl, (x_gl, y_gl), retain_graph=True)

    run_gl_backward()
    torch.cuda.synchronize()

    # -------- timings --------
    is_large = n >= 10_000
    warmup = int(warmup_large if is_large else warmup_small)
    rep = int(rep_large if is_large else rep_small)

    ms_ot = _bench_once(run_ot_backward, warmup=warmup, rep=rep)
    ms_gl = _bench_once(run_gl_backward, warmup=warmup, rep=rep)

    result = {
        "N": float(n),
        "steps": float(steps),
        "ot_triton_bwd_ms": float(ms_ot),
        "geomloss_bwd_ms": float(ms_gl),
    }
    if dump_triton_config and autotune_triton:
        cfg = getattr(grad_kernels, "_geomloss_grad_sqeuclid_autotune", None)
        best = getattr(cfg, "best_config", None) if cfg is not None else None
        if best is not None:
            result["triton_block_m"] = float(best.kwargs.get("BLOCK_M", 0))
            result["triton_block_n"] = float(best.kwargs.get("BLOCK_N", 0))
            result["triton_block_k"] = float(best.kwargs.get("BLOCK_K", 0))
            result["triton_num_warps"] = float(best.num_warps)
            result["triton_num_stages"] = float(best.num_stages)
    return result


def _plot_two(
    xs: np.ndarray,
    ys: Dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    logx: bool,
    logy: bool,
) -> None:
    def _wrap_title(text: str, width: int = 80) -> str:
        return "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, ys["ot_triton_bwd_ms"], "o-", linewidth=2, label="ot_triton")
    plt.plot(xs, ys["geomloss_bwd_ms"], "o-", linewidth=2, label="geomloss")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.title(_wrap_title(title), fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark backward (grad) for online Sinkhorn: ot_triton vs GeomLoss."
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geomloss-truncate", type=int, default=5)
    parser.add_argument(
        "--dump-triton-config",
        action="store_true",
        help="Record the selected Triton autotune config per size (ot_triton backward).",
    )

    parser.add_argument("--warmup-small", type=int, default=20)
    parser.add_argument("--rep-small", type=int, default=100)
    parser.add_argument("--warmup-large", type=int, default=3)
    parser.add_argument("--rep-large", type=int, default=5)

    parser.add_argument(
        "--skip-small",
        action="store_true",
        help="Skip the small-N sweep (128..8192).",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip the large-N sweep (10k..50k).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("output") / "bench_sinkhorn_grad_vs_geomloss_eps0p1_d64"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.skip_small and args.skip_large:
        raise ValueError("At least one of small/large sweeps must be enabled.")

    _set_tf32(bool(args.allow_tf32))
    _preload_cuda_libs()

    device = torch.device("cuda")
    sizes_small = [128, 256, 512, 1024, 2048, 4096, 8192]
    sizes_large = [10000, 20000, 30000, 40000, 50000]
    small_set = {int(x) for x in sizes_small}
    large_set = {int(x) for x in sizes_large}

    sizes: List[int] = []
    if not args.skip_small:
        sizes += sizes_small
    if not args.skip_large:
        sizes += sizes_large

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, float]] = []
    for n in sizes:
        row = _run_one_size(
            n=int(n),
            d=int(args.d),
            eps=float(args.eps),
            n_iters=int(args.n_iters),
            allow_tf32=bool(args.allow_tf32),
            use_exp2=not args.no_exp2,
            autotune_triton=not args.no_autotune,
            warmup_small=int(args.warmup_small),
            rep_small=int(args.rep_small),
            warmup_large=int(args.warmup_large),
            rep_large=int(args.rep_large),
            rng=rng,
            device=device,
            geomloss_truncate=int(args.geomloss_truncate),
            dump_triton_config=bool(args.dump_triton_config),
        )
        rows.append(row)
        msg = (
            f"N={int(row['N'])} steps={int(row['steps'])} "
            f"ot_triton_bwd_ms={row['ot_triton_bwd_ms']:.3f} "
            f"geomloss_bwd_ms={row['geomloss_bwd_ms']:.3f}"
        )
        if args.dump_triton_config and "triton_block_m" in row:
            msg += (
                f" cfg=BM{int(row['triton_block_m'])}"
                f"/BN{int(row['triton_block_n'])}"
                f"/BK{int(row['triton_block_k'])}"
                f" warps{int(row['triton_num_warps'])}"
                f" stages{int(row['triton_num_stages'])}"
            )
        print(msg)

    csv_path = out_dir / "sinkhorn_grad_vs_geomloss.csv"
    header = "N,steps,ot_triton_bwd_ms,geomloss_bwd_ms"
    data = np.array(
        [[r["N"], r["steps"], r["ot_triton_bwd_ms"], r["geomloss_bwd_ms"]] for r in rows],
        dtype=np.float64,
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}")

    if args.dump_triton_config:
        cfg_path = out_dir / "sinkhorn_grad_triton_configs.txt"
        lines = []
        for row in rows:
            if "triton_block_m" not in row:
                continue
            lines.append(
                "N={N} steps={steps} BM={bm} BN={bn} BK={bk} warps={warps} stages={stages}".format(
                    N=int(row["N"]),
                    steps=int(row["steps"]),
                    bm=int(row["triton_block_m"]),
                    bn=int(row["triton_block_n"]),
                    bk=int(row["triton_block_k"]),
                    warps=int(row["triton_num_warps"]),
                    stages=int(row["triton_num_stages"]),
                )
            )
        cfg_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Saved {cfg_path}")

    data_small = (
        np.array([row for row in data if int(row[0]) in small_set], dtype=np.float64)
        if not args.skip_small
        else np.zeros((0, 4), dtype=np.float64)
    )
    data_large = (
        np.array([row for row in data if int(row[0]) in large_set], dtype=np.float64)
        if not args.skip_large
        else np.zeros((0, 4), dtype=np.float64)
    )

    xs_small = data_small[:, 0] if data_small.size else np.zeros((0,), dtype=np.float64)
    ys_small = {
        "ot_triton_bwd_ms": data_small[:, 2] if data_small.size else np.zeros((0,), dtype=np.float64),
        "geomloss_bwd_ms": data_small[:, 3] if data_small.size else np.zeros((0,), dtype=np.float64),
    }
    xs_large = data_large[:, 0] if data_large.size else np.zeros((0,), dtype=np.float64)
    ys_large = {
        "ot_triton_bwd_ms": data_large[:, 2] if data_large.size else np.zeros((0,), dtype=np.float64),
        "geomloss_bwd_ms": data_large[:, 3] if data_large.size else np.zeros((0,), dtype=np.float64),
    }

    steps = int(args.n_iters) + 2
    common = (
        f"eps={args.eps} d={args.d} n_iters={args.n_iters} (steps={steps}) "
        f"tf32={'on' if args.allow_tf32 else 'off'} exp2={'off' if args.no_exp2 else 'on'}"
    )

    if xs_small.size:
        _plot_two(
            xs_small,
            ys_small,
            title="Sinkhorn backward (online) vs N (log-log)\n" + common,
            xlabel="N (=M)",
            ylabel="Backward time (ms)",
            out_png=out_dir / "sinkhorn_grad_small_loglog.png",
            logx=True,
            logy=True,
        )
        print(f"Saved {out_dir / 'sinkhorn_grad_small_loglog.png'}")

    if xs_large.size:
        _plot_two(
            xs_large,
            ys_large,
            title="Sinkhorn backward (online) vs N (linear)\n" + common,
            xlabel="N (=M)",
            ylabel="Backward time (ms)",
            out_png=out_dir / "sinkhorn_grad_large_linear.png",
            logx=False,
            logy=False,
        )
        print(f"Saved {out_dir / 'sinkhorn_grad_large_linear.png'}")


if __name__ == "__main__":
    main()
