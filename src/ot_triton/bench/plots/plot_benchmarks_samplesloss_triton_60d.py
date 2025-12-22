"""
Benchmark Sinkhorn SamplesLoss in 60D (GeomLoss vs ot_triton)
============================================================

This is the same benchmark style as `plot_benchmarks_samplesloss_triton_3d.py`,
but for D=60 to match higher-dimensional regimes where Triton tiling is effective.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from ot_triton import SamplesLoss as TritonSamplesLoss

    _OT_TRITON_AVAILABLE = True
except Exception:
    _OT_TRITON_AVAILABLE = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Allow very long runs to reach N=1e6 (still O(N^2) per Sinkhorn sweep).
MAXTIME = 1000 if use_cuda else 1
REDTIME = 2 if use_cuda else 0.2
D = 60

# Match the reference GeomLoss benchmark grid (we will stop early once too slow).
NS: Sequence[int] = [
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
]

# Sinkhorn settings (match GeomLoss convention but with *full* squared distance).
BLURS: Sequence[float] = [0.05, 0.01]
SCALING = 0.5
DIAMETER = 1.0

# Strict fp32 by default (TF32 can be enabled for speed, but changes numerics).
ALLOW_TF32 = False
DEFAULT_OUT_DIR = Path("output") / "samplesloss"


def _preload_cuda_libs() -> None:
    """Preload CUDA runtime libs so KeOps (GeomLoss online backend) stays on GPU."""

    if not use_cuda:
        return

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


def generate_samples(n: int, *, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Create point clouds sampled non-uniformly on a sphere of diameter ~1."""

    x = torch.randn(n, D, device=device, dtype=torch.float32)
    x[:, 0] += 1
    x = x / (2 * x.norm(dim=1, keepdim=True))

    y = torch.randn(n, D, device=device, dtype=torch.float32)
    y[:, 1] += 2
    y = y / (2 * y.norm(dim=1, keepdim=True))

    # Mirror GeomLoss' original benchmark: gradient w.r.t. x only.
    x.requires_grad_(True)
    y.requires_grad_(False)

    a = torch.randn(n, device=device, dtype=torch.float32).abs()
    b = torch.randn(n, device=device, dtype=torch.float32).abs()
    a = a / a.sum()
    b = b / b.sum()

    return a, x, b, y


def benchmark(loss: torch.nn.Module, *, n: int, loops: int, warmup: bool) -> float:
    """Time a loss computation + backward on an N-by-N problem (seconds / call)."""

    a, x, b, y = generate_samples(n, device=device)

    def run_once() -> None:
        out = loss(a, x, b, y)
        out.backward()

    # Ensure sample generation kernels (randn, norms, etc.) are not counted.
    if use_cuda:
        torch.cuda.synchronize()

    # Warmup is only needed once per (loss, dtype, D) config to exclude JIT/compilation.
    if warmup:
        run_once()
        if use_cuda:
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(int(loops)):
        run_once()
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sec = elapsed / loops
    print(f"{loops:3} NxN loss, N={n:7}: {sec:.6f}s")
    return sec


def bench_config(loss: torch.nn.Module) -> List[float]:
    """Benchmark a single loss object for increasing number of samples."""

    backend = getattr(loss, "backend", "unknown")
    print(f"Backend: {backend}, Device: {device} -------------")

    times: List[float] = []

    def run_bench() -> None:
        nonlocal times
        nloops_choices = [100, 10, 1]
        nloops = nloops_choices.pop(0)

        prev_n: int | None = None
        prev_sec: float | None = None

        for n in NS:
            # Prevent obviously-infeasible runs once we're down to a single loop.
            if nloops == 1 and prev_n is not None and prev_sec is not None:
                est = prev_sec * (float(n) / float(prev_n)) ** 2
                if est > MAXTIME:
                    print("**\nToo slow (estimated)!")
                    break

            sec = benchmark(loss, n=n, loops=nloops, warmup=(prev_n is None))
            times.append(sec)
            prev_n, prev_sec = n, sec

            if (nloops * sec > MAXTIME) or (nloops * sec > REDTIME and len(nloops_choices) > 0):
                nloops = nloops_choices.pop(0)

    try:
        run_bench()
    except IndexError:
        print("**\nToo slow !")
    except RuntimeError as err:
        if use_cuda and str(err).startswith("CUDA"):
            print("**\nMemory overflow !")
        else:
            raise

    return times + (len(NS) - len(times)) * [np.nan]


@dataclass(frozen=True)
class Impl:
    name: str
    make: Callable[[float], torch.nn.Module]


def _impls_for_blur(blur: float, *, use_exp2: bool) -> Sequence[Impl]:
    impls: List[Impl] = []

    try:
        from geomloss import SamplesLoss as GeomLossSamplesLoss

        impls.append(
            Impl(
                name='geomloss backend="online"',
                make=lambda b: GeomLossSamplesLoss(
                    "sinkhorn",
                    p=2,
                    blur=b,
                    scaling=SCALING,
                    diameter=DIAMETER,
                    cost="SqDist(X,Y)",
                    debias=False,
                    potentials=False,
                    backend="online",
                ),
            )
        )
    except Exception:
        pass

    if _OT_TRITON_AVAILABLE and use_cuda:
        impls.append(
            Impl(
                name=f"ot_triton (strict fp32, exp2={'on' if use_exp2 else 'off'})",
                make=lambda b: TritonSamplesLoss(
                    "sinkhorn",
                    p=2,
                    blur=b,
                    scaling=SCALING,
                    debias=False,
                    potentials=False,
                    backend="online",
                    normalize=False,
                    use_epsilon_scaling=True,
                    last_extrapolation=True,
                    allow_tf32=ALLOW_TF32,
                    use_exp2=use_exp2,
                    autotune=False,
                    diameter=DIAMETER,
                ),
            )
        )

    return impls


def _csv_key(name: str) -> str:
    if name.startswith('geomloss'):
        return "geomloss_s"
    if name.startswith("ot_triton"):
        return "ot_triton_s"
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return f"{safe.lower()}_s"


def _write_csv(path: Path, *, npoints: Sequence[int], series: dict[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(series.keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Npoints", *keys])
        for i, n in enumerate(npoints):
            writer.writerow([int(n), *[float(series[k][i]) for k in keys]])


def full_bench_sinkhorn_60d(
    *, blurs: Sequence[float], use_exp2: bool, out_dir: Path
) -> None:
    if not _OT_TRITON_AVAILABLE:
        raise RuntimeError(
            "ot_triton is not importable. Run with `PYTHONPATH=src` or `pip install -e .`."
        )
    if not use_cuda:
        raise RuntimeError("CUDA is required (ot_triton is CUDA-only).")

    _preload_cuda_libs()
    _set_tf32(ALLOW_TF32)

    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_exp2" if use_exp2 else ""

    for blur in blurs:
        impls = list(_impls_for_blur(blur, use_exp2=use_exp2))
        if len(impls) == 0:
            raise RuntimeError("No implementations available to benchmark.")

        print("\nBenchmarking Sinkhorn ===============================")
        print(
            f"D={D} blur={blur} scaling={SCALING} diameter={DIAMETER} "
            f"tf32={'on' if ALLOW_TF32 else 'off'}"
        )

        lines = [list(NS)]
        col_keys: List[str] = []
        for impl in impls:
            loss = impl.make(blur)
            col_keys.append(_csv_key(impl.name))
            lines.append(bench_config(loss))

        benches = np.array(lines).T

        plt.figure()
        linestyles = ["o-", "s-", "^-", "d-"]
        for i, impl in enumerate(impls):
            plt.plot(
                benches[:, 0],
                benches[:, i + 1],
                linestyles[i % len(linestyles)],
                linewidth=2,
                label=impl.name,
            )

        plt.title(f"Runtime for Sinkhorn SamplesLoss in {D}D (blur={blur})")
        plt.xlabel("Number of samples per measure")
        plt.ylabel("Seconds (forward+backward)")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.grid(True, which="major", linestyle="-")
        plt.grid(True, which="minor", linestyle="dotted")
        plt.axis([NS[0], NS[-1], 1e-4, MAXTIME])
        plt.tight_layout()

        csv_path = out_dir / f"benchmark_sinkhorn_{D}D_blur{blur:g}{suffix}.csv"
        png_path = out_dir / f"benchmark_sinkhorn_{D}D_blur{blur:g}{suffix}.png"
        series = {col_keys[i]: benches[:, i + 1].tolist() for i in range(len(col_keys))}
        _write_csv(csv_path, npoints=benches[:, 0].tolist(), series=series)
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"Saved {csv_path}")
        print(f"Saved {png_path}")

        # Speedup plot: GeomLoss / ot_triton
        if benches.shape[1] >= 3:
            speedup = benches[:, 1] / benches[:, 2]

            plt.figure()
            plt.plot(
                benches[:, 0],
                speedup,
                "o-",
                linewidth=2,
                label="GeomLoss / ot_triton",
            )
            plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
            plt.title(f"Speedup for Sinkhorn SamplesLoss in {D}D (blur={blur})")
            plt.xlabel("Number of samples per measure")
            plt.ylabel("Speedup (>1 means ot_triton faster)")
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(True, which="major", linestyle="-")
            plt.grid(True, which="minor", linestyle="dotted")
            plt.tight_layout()

            speedup_csv = out_dir / f"benchmark_sinkhorn_{D}D_blur{blur:g}{suffix}_speedup.csv"
            speedup_png = out_dir / f"benchmark_sinkhorn_{D}D_blur{blur:g}{suffix}_speedup.png"
            _write_csv(
                speedup_csv,
                npoints=benches[:, 0].tolist(),
                series={"speedup_geomloss_over_ot_triton": speedup.tolist()},
            )
            plt.savefig(speedup_png, dpi=200)
            plt.close()
            print(f"Saved {speedup_csv}")
            print(f"Saved {speedup_png}")

    if os.environ.get("DISPLAY"):
        plt.show()
    else:
        print("No DISPLAY detected; skipping plt.show().")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Sinkhorn SamplesLoss in 60D.")
    parser.add_argument(
        "--blur",
        type=float,
        default=None,
        help="If set, run only this blur value (otherwise run the default list).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="If set, restrict the benchmark grid to N <= max-n (for quick smoke tests).",
    )
    parser.add_argument(
        "--exp2",
        action="store_true",
        help="Enable exp2/log2 path in ot_triton kernels (GeomLoss unchanged).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help='Output directory for CSV/PNG files (default: "output/samplesloss").',
    )
    args = parser.parse_args()

    if args.max_n is not None:
        global NS
        NS = [int(n) for n in NS if int(n) <= int(args.max_n)]
        if len(NS) == 0:
            raise ValueError(f"--max-n={args.max_n} filtered out all benchmark sizes.")

    blurs = [float(args.blur)] if args.blur is not None else list(BLURS)
    full_bench_sinkhorn_60d(blurs=blurs, use_exp2=bool(args.exp2), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
