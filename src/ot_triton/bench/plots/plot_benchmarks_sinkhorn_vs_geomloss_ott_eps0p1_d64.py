"""
Benchmark online Sinkhorn potentials (eps fixed) for:
  - ot_triton (fused GeomLoss-style symmetric updates in Triton)
  - GeomLoss (KeOps backend="online")
  - OTT/JAX (PointCloud batch_size != None)

Produces two figures:
  1) Small N sweep in log-log scale
  2) Large N sweep in linear scale

IMPORTANT: Sizes are benchmarked in reverse order (large→small) to avoid Triton
autotuning cache pollution. Running small sizes first causes the autotuner to
cache suboptimal kernel configurations that degrade large-N performance by ~50%.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import triton.testing as testing

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
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def _bench_once_torch(fn, *, warmup: int, rep: int) -> float:
    return float(testing.do_bench(fn, warmup=warmup, rep=rep))


def _bench_once_jax(fn, *, warmup: int, rep: int) -> float:
    import jax

    for _ in range(int(warmup)):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(int(rep)):
        jax.block_until_ready(fn())
    end = time.perf_counter()
    return (end - start) * 1000.0 / float(rep)


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
    ott_batch_size: int,
    jax_matmul_precision: str,
    warmup_small: int,
    rep_small: int,
    warmup_large: int,
    rep_large: int,
    rng: np.random.Generator,
    device: torch.device,
    geomloss_truncate: int,
) -> Dict[str, float]:
    x_np = rng.standard_normal((n, d), dtype=np.float32)
    y_np = rng.standard_normal((n, d), dtype=np.float32)
    a_np, b_np = _make_weights(rng, n)

    # Torch tensors (fp32).
    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
    a_t = torch.from_numpy(a_np).to(device=device, dtype=torch.float32)
    b_t = torch.from_numpy(b_np).to(device=device, dtype=torch.float32)

    eps_list = [float(eps)] * int(n_iters)
    n_steps = int(n_iters) + 2  # init + n_iters symmetric + last extrapolation

    # -------- ot_triton (GeomLoss-style fused loop) --------
    def run_ot_triton() -> None:
        sinkhorn_geomloss_online_potentials_sqeuclid(
            x_t,
            y_t,
            a_t,
            b_t,
            use_epsilon_scaling=False,
            last_extrapolation=True,
            eps_list=eps_list,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            autotune=autotune_triton,
        )

    run_ot_triton()
    torch.cuda.synchronize()

    # -------- GeomLoss (online, fixed eps_list) --------
    # Import geomloss after CUDA libs are preloaded.
    from geomloss.sinkhorn_divergence import sinkhorn_loop as geomloss_sinkhorn_loop
    from geomloss.sinkhorn_divergence import log_weights as geomloss_log_weights
    from geomloss.sinkhorn_samples import lse_genred, softmin_online
    from functools import partial

    # Full squared Euclidean cost (not /2).
    my_lse = lse_genred("SqDist(X,Y)", d, dtype="float32")
    softmin = partial(softmin_online, log_conv=my_lse)

    loga = geomloss_log_weights(a_t)
    logb = geomloss_log_weights(b_t)
    C_xy = (x_t, y_t.detach())
    C_yx = (y_t, x_t.detach())

    def run_geomloss() -> None:
        # debias=False matches our current comparisons (fewer problems solved).
        # Use the same fixed-eps iteration count as ot_triton: init + n_iters + last.
        geomloss_sinkhorn_loop(
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

    run_geomloss()
    torch.cuda.synchronize()

    # -------- OTT/JAX (online) --------
    import jax
    from jax import config as jax_config
    import jax.numpy as jnp

    jax_config.update("jax_default_matmul_precision", jax_matmul_precision)
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX GPU backend is required for OTT benchmarks.")

    from ott.geometry import pointcloud

    dev = jax.devices("gpu")[0]
    x_j = jax.device_put(jnp.asarray(x_np, dtype=jnp.float32), device=dev)
    y_j = jax.device_put(jnp.asarray(y_np, dtype=jnp.float32), device=dev)
    loga_j = jax.device_put(jnp.asarray(np.log(a_np), dtype=jnp.float32), device=dev)
    logb_j = jax.device_put(jnp.asarray(np.log(b_np), dtype=jnp.float32), device=dev)

    geom = pointcloud.PointCloud(
        x_j, y_j, batch_size=int(ott_batch_size), epsilon=float(eps)
    )

    def sinkhorn_loop(geom, loga, logb):
        f0 = jnp.zeros((n,), dtype=jnp.float32)
        g0 = jnp.zeros((n,), dtype=jnp.float32)

        def body(_, state):
            f, g = state
            g = geom.update_potential(f, g, logb, axis=0)
            f = geom.update_potential(f, g, loga, axis=1)
            return (f, g)

        # Match the number of GeomLoss-style "steps" (init + n_iters + last).
        return jax.lax.fori_loop(0, int(n_steps), body, (f0, g0))

    sinkhorn_loop_jit = jax.jit(sinkhorn_loop)
    f_j, g_j = sinkhorn_loop_jit(geom, loga_j, logb_j)
    jax.block_until_ready((f_j, g_j))

    run_ott = lambda: sinkhorn_loop_jit(geom, loga_j, logb_j)

    # -------- timings --------
    is_large = n >= 10_000
    warmup = int(warmup_large if is_large else warmup_small)
    rep = int(rep_large if is_large else rep_small)

    ms_ot_triton = _bench_once_torch(run_ot_triton, warmup=warmup, rep=rep)
    ms_geomloss = _bench_once_torch(run_geomloss, warmup=warmup, rep=rep)
    ms_ott = _bench_once_jax(run_ott, warmup=warmup, rep=rep)

    return {
        "N": float(n),
        "steps": float(n_steps),
        "ot_triton_ms": float(ms_ot_triton),
        "geomloss_ms": float(ms_geomloss),
        "ott_ms": float(ms_ott),
    }


def _plot_three(
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
    plt.plot(xs, ys["ot_triton_ms"], "o-", linewidth=2, label="ot_triton")
    plt.plot(xs, ys["geomloss_ms"], "o-", linewidth=2, label="geomloss")
    plt.plot(xs, ys["ott_ms"], "o-", linewidth=2, label="ott (jax)")
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
        description="Benchmark online Sinkhorn potentials: ot_triton vs GeomLoss vs OTT/JAX."
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--ott-batch-size", type=int, default=256)
    parser.add_argument(
        "--jax-matmul-precision",
        type=str,
        default="highest",
        choices=["default", "high", "highest"],
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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

    parser.add_argument("--warmup-small", type=int, default=20)
    parser.add_argument("--rep-small", type=int, default=100)
    parser.add_argument("--warmup-large", type=int, default=3)
    parser.add_argument("--rep-large", type=int, default=5)

    parser.add_argument("--geomloss-truncate", type=int, default=5)

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("output") / "bench_sinkhorn_vs_geomloss_ott_eps0p1_d64"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.ott_batch_size <= 0:
        raise ValueError("--ott-batch-size must be > 0 for online mode.")

    device = torch.device("cuda")
    _set_tf32(bool(args.allow_tf32))
    _preload_cuda_libs()

    sizes_small = [128, 256, 512, 1024, 2048, 4096, 8192]
    sizes_large = [10000, 20000, 30000, 40000, 50000]
    small_set = {int(x) for x in sizes_small}
    large_set = {int(x) for x in sizes_large}

    if args.skip_small and args.skip_large:
        raise ValueError("At least one of small/large sweeps must be enabled.")

    # IMPORTANT: Run sizes in REVERSE order (large→small) to avoid Triton autotuning
    # cache pollution. The autotuner caches kernel configs per input shape; running
    # small sizes first populates the cache with configs optimized for small N, which
    # then get incorrectly reused for large N, causing ~50% performance degradation.
    sizes: List[int] = []
    if not args.skip_large:
        sizes += list(reversed(sizes_large))  # 50k, 40k, 30k, 20k, 10k
    if not args.skip_small:
        sizes += list(reversed(sizes_small))  # 8192, 4096, ..., 128

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
            ott_batch_size=int(args.ott_batch_size),
            jax_matmul_precision=str(args.jax_matmul_precision),
            warmup_small=int(args.warmup_small),
            rep_small=int(args.rep_small),
            warmup_large=int(args.warmup_large),
            rep_large=int(args.rep_large),
            rng=rng,
            device=device,
            geomloss_truncate=int(args.geomloss_truncate),
        )
        rows.append(row)
        print(
            f"N={int(row['N'])} steps={int(row['steps'])} "
            f"ot_triton_ms={row['ot_triton_ms']:.3f} "
            f"geomloss_ms={row['geomloss_ms']:.3f} ott_ms={row['ott_ms']:.3f}"
        )

    # Save CSV.
    csv_path = out_dir / "sinkhorn_vs_geomloss_ott.csv"
    header = "N,steps,ot_triton_ms,geomloss_ms,ott_ms"
    data = np.array(
        [
            [r["N"], r["steps"], r["ot_triton_ms"], r["geomloss_ms"], r["ott_ms"]]
            for r in rows
        ],
        dtype=np.float64,
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}")

    # Split and plot.
    data_small = (
        np.array([row for row in data if int(row[0]) in small_set], dtype=np.float64)
        if not args.skip_small
        else np.zeros((0, 5), dtype=np.float64)
    )
    data_large = (
        np.array([row for row in data if int(row[0]) in large_set], dtype=np.float64)
        if not args.skip_large
        else np.zeros((0, 5), dtype=np.float64)
    )

    xs_small = data_small[:, 0] if data_small.size else np.zeros((0,), dtype=np.float64)
    ys_small = {
        "ot_triton_ms": data_small[:, 2] if data_small.size else np.zeros((0,), dtype=np.float64),
        "geomloss_ms": data_small[:, 3] if data_small.size else np.zeros((0,), dtype=np.float64),
        "ott_ms": data_small[:, 4] if data_small.size else np.zeros((0,), dtype=np.float64),
    }
    xs_large = data_large[:, 0] if data_large.size else np.zeros((0,), dtype=np.float64)
    ys_large = {
        "ot_triton_ms": data_large[:, 2] if data_large.size else np.zeros((0,), dtype=np.float64),
        "geomloss_ms": data_large[:, 3] if data_large.size else np.zeros((0,), dtype=np.float64),
        "ott_ms": data_large[:, 4] if data_large.size else np.zeros((0,), dtype=np.float64),
    }

    n_steps = int(args.n_iters) + 2
    common = (
        f"eps={args.eps} d={args.d} n_iters={args.n_iters} (steps={n_steps}) "
        f"online(ott_batch_size={args.ott_batch_size}) "
        f"tf32={'on' if args.allow_tf32 else 'off'} exp2={'off' if args.no_exp2 else 'on'}"
    )

    if xs_small.size:
        _plot_three(
            xs_small,
            ys_small,
            title="Sinkhorn (online) runtime vs N (log-log)\\n" + common,
            xlabel="N (=M)",
            ylabel="Time (ms)",
            out_png=out_dir / "sinkhorn_small_loglog.png",
            logx=True,
            logy=True,
        )
        print(f"Saved {out_dir / 'sinkhorn_small_loglog.png'}")

    if xs_large.size:
        _plot_three(
            xs_large,
            ys_large,
            title="Sinkhorn (online) runtime vs N (linear)\\n" + common,
            xlabel="N (=M)",
            ylabel="Time (ms)",
            out_png=out_dir / "sinkhorn_large_linear.png",
            logx=False,
            logy=False,
        )
        print(f"Saved {out_dir / 'sinkhorn_large_linear.png'}")


if __name__ == "__main__":
    main()
