import os
import argparse
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud


def _bench_once(fn, warmup=25, rep=100):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(rep):
        jax.block_until_ready(fn())
    end = time.perf_counter()
    return (end - start) * 1000.0 / rep


def run_bench(n, m, d, axis, dtype, batch_size, eps, with_vec, use_jit):
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX GPU backend is required for OTT benchmarks.")

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    x = jax.random.normal(k1, (n, d), dtype=dtype)
    y = jax.random.normal(k2, (m, d), dtype=dtype)
    f = jax.random.normal(k3, (n,), dtype=jnp.float32)
    g = jax.random.normal(k4, (m,), dtype=jnp.float32)
    vec = None
    if with_vec:
        vec = jax.random.normal(
            k5, (m if axis == 1 else n,), dtype=jnp.float32
        )

    batch_arg = None if batch_size == 0 else batch_size
    geom = pointcloud.PointCloud(x, y, batch_size=batch_arg)

    if with_vec:
        def apply(geom, f, g, vec):
            return geom.apply_lse_kernel(f, g, eps, vec=vec, axis=axis)[0]
    else:
        def apply(geom, f, g):
            return geom.apply_lse_kernel(f, g, eps, axis=axis)[0]

    if use_jit:
        apply = jax.jit(apply)
        if with_vec:
            jax.block_until_ready(apply(geom, f, g, vec))
        else:
            jax.block_until_ready(apply(geom, f, g))
    else:
        if with_vec:
            jax.block_until_ready(apply(geom, f, g, vec))
        else:
            jax.block_until_ready(apply(geom, f, g))

    if with_vec:
        fn = lambda: apply(geom, f, g, vec)
    else:
        fn = lambda: apply(geom, f, g)
    ms = _bench_once(fn)
    mode = "online" if batch_arg is not None else "offline"
    vec_mode = "vec" if with_vec else "no-vec"
    jit_mode = "jit" if use_jit else "eager"
    print(
        f"ott {mode} {jit_mode} {vec_mode} axis={axis} dtype={dtype} "
        f"n={n} m={m} d={d} time_ms={ms:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--with-vec", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = dtype_map[args.dtype]
    run_bench(
        args.n,
        args.m,
        args.d,
        args.axis,
        dtype,
        args.batch_size,
        args.eps,
        args.with_vec,
        not args.no_jit,
    )


if __name__ == "__main__":
    main()
