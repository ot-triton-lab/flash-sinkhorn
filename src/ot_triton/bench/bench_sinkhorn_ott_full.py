import argparse
import os
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import triton.testing as testing

from ot_triton.kernels.sinkhorn_triton_sqeuclid import sinkhorn_potentials_sqeuclid


def _bench_once_torch(fn, warmup: int, rep: int) -> float:
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def _bench_once_jax(fn, warmup: int, rep: int) -> float:
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(rep):
        jax.block_until_ready(fn())
    end = time.perf_counter()
    return (end - start) * 1000.0 / rep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark OTT-style alternating Sinkhorn potentials (Torch+Triton vs JAX+OTT)."
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
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Enable guided Triton autotune for apply_lse_kernel (one-time per (D, dtype, tf32, vec, exp2)).",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 for matmul/dot where supported (Torch and Triton).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable FlashAttention-style exp2/log2 reduction in Triton (use exp/log).",
    )
    parser.add_argument(
        "--jax-matmul-precision",
        type=str,
        default="default",
        choices=["default", "high", "highest"],
        help='JAX matmul precision (controls TF32 usage on Ampere GPUs); use "highest" for strict FP32.',
    )
    parser.add_argument(
        "--bench-order",
        type=str,
        default="torch-first",
        choices=["torch-first", "jax-first"],
        help="Run timing in this order (helps avoid GPU clock skew in comparisons).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarks.")

    dtype_torch = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((args.n, args.d), dtype=np.float32)
    y_np = rng.standard_normal((args.m, args.d), dtype=np.float32)
    a_np = rng.random((args.n,), dtype=np.float32) + 0.1
    b_np = rng.random((args.m,), dtype=np.float32) + 0.1
    a_np = a_np / a_np.sum()
    b_np = b_np / b_np.sum()
    loga_np = np.log(a_np).astype(np.float32)
    logb_np = np.log(b_np).astype(np.float32)

    device = torch.device("cuda")
    x_t = torch.from_numpy(x_np).to(device=device, dtype=dtype_torch)
    y_t = torch.from_numpy(y_np).to(device=device, dtype=dtype_torch)
    loga_t = torch.from_numpy(loga_np).to(device=device, dtype=torch.float32)
    logb_t = torch.from_numpy(logb_np).to(device=device, dtype=torch.float32)

    def run_triton():
        sinkhorn_potentials_sqeuclid(
            x_t,
            y_t,
            loga_t,
            logb_t,
            args.eps,
            args.n_iters,
            use_exp2=not args.no_exp2,
            allow_tf32=args.allow_tf32,
            autotune=args.autotune,
        )

    import jax
    from jax import config as jax_config
    import jax.numpy as jnp

    jax_config.update("jax_default_matmul_precision", args.jax_matmul_precision)

    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX GPU backend is required for OTT benchmarks.")

    from ott.geometry import pointcloud

    dev = jax.devices("gpu")[0]
    x_j = jax.device_put(jnp.asarray(x_np, dtype=getattr(jnp, args.dtype)), device=dev)
    y_j = jax.device_put(jnp.asarray(y_np, dtype=getattr(jnp, args.dtype)), device=dev)
    loga_j = jax.device_put(jnp.asarray(loga_np, dtype=jnp.float32), device=dev)
    logb_j = jax.device_put(jnp.asarray(logb_np, dtype=jnp.float32), device=dev)

    batch_arg = None if args.batch_size == 0 else args.batch_size
    geom = pointcloud.PointCloud(x_j, y_j, batch_size=batch_arg, epsilon=args.eps)

    def sinkhorn_loop(geom, loga, logb):
        f0 = jnp.zeros((args.n,), dtype=jnp.float32)
        g0 = jnp.zeros((args.m,), dtype=jnp.float32)

        def body(_, state):
            f, g = state
            g = geom.update_potential(f, g, logb, axis=0)
            f = geom.update_potential(f, g, loga, axis=1)
            return (f, g)

        return jax.lax.fori_loop(0, args.n_iters, body, (f0, g0))

    sinkhorn_loop_jit = jax.jit(sinkhorn_loop)
    f_j, g_j = sinkhorn_loop_jit(geom, loga_j, logb_j)
    jax.block_until_ready((f_j, g_j))

    run_ott = lambda: sinkhorn_loop_jit(geom, loga_j, logb_j)
    if args.bench_order == "jax-first":
        ms_ott = _bench_once_jax(run_ott, warmup=args.warmup, rep=args.rep)
        run_triton()
        torch.cuda.synchronize()
        ms_torch = _bench_once_torch(run_triton, warmup=args.warmup, rep=args.rep)
    else:
        run_triton()
        torch.cuda.synchronize()
        ms_torch = _bench_once_torch(run_triton, warmup=args.warmup, rep=args.rep)
        ms_ott = _bench_once_jax(run_ott, warmup=args.warmup, rep=args.rep)

    mode = "online" if batch_arg is not None else "offline"
    print(
        f"n={args.n} m={args.m} d={args.d} dtype={args.dtype} eps={args.eps} "
        f"n_iters={args.n_iters} ott_batch_size={args.batch_size} ({mode}) "
        f"allow_tf32={'on' if args.allow_tf32 else 'off'} autotune={'on' if args.autotune else 'off'} "
        f"triton_exp2={'off' if args.no_exp2 else 'on'} "
        f"jax_matmul_precision={args.jax_matmul_precision} bench_order={args.bench_order}"
    )
    print(f"torch_sinkhorn_full_ms={ms_torch:.3f}")
    print(f"ott_jax_sinkhorn_full_ms={ms_ott:.3f}")


if __name__ == "__main__":
    main()
