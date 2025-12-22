import argparse
import ctypes
import os
from typing import Callable, Tuple

import torch
import triton.testing as testing

from ot_triton import SamplesLoss as TritonSamplesLoss


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


def _bench_once(fn: Callable[[], None], warmup: int, rep: int) -> float:
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def _clone_inputs(
    x0: torch.Tensor, y0: torch.Tensor, *, requires_grad: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x0.detach().clone()
    y = y0.detach().clone()
    x.requires_grad_(requires_grad)
    y.requires_grad_(requires_grad)
    return x, y


def _profile_backward(
    name: str, backward_fn: Callable[[], None], *, enabled: bool, steps: int
) -> None:
    if not enabled:
        return

    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        print("torch.profiler not available; skipping profile.")
        return

    print(f"\n== torch.profiler: {name} backward ==")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(int(steps)):
            backward_fn()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark/ profile backward (grad x,y) for ot_triton vs GeomLoss SamplesLoss."
    )
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--blur", type=float, default=0.1)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--no-eps-scaling", action="store_true")
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument(
        "--geomloss-backend",
        type=str,
        default="online",
        choices=["online", "tensorized", "auto"],
        help="GeomLoss SamplesLoss backend.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 for dot/matmul (affects GeomLoss + Triton when applicable).",
    )
    parser.add_argument(
        "--no-exp2",
        action="store_true",
        help="Disable exp2/log2 reduction in ot_triton kernels (use exp/log).",
    )
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Disable Triton autotune (compile a single default config).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler for backward (1 run per implementation).",
    )
    parser.add_argument("--profile-steps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _preload_cuda_libs()
    _set_tf32(args.tf32)

    # Import geomloss after preloading CUDA libs so KeOps stays on GPU.
    from geomloss import SamplesLoss as GeomLossSamplesLoss

    def _sqdist_cost_full(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        y_f = y.float()
        x2 = (x_f * x_f).sum(dim=-1, keepdim=True)
        y2 = (y_f * y_f).sum(dim=-1, keepdim=True).transpose(-2, -1)
        return x2 + y2 - 2.0 * torch.matmul(x_f, y_f.transpose(-2, -1))

    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    torch.manual_seed(0)
    x0 = torch.randn(args.n, args.d, device=device, dtype=dtype)
    y0 = torch.randn(args.m, args.d, device=device, dtype=dtype)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    # Share the same diameter for matching eps schedules.
    diameter = None
    if not args.no_eps_scaling:
        from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import max_diameter

        diameter = max_diameter(x0, y0)

    ot_loss = TritonSamplesLoss(
        "sinkhorn",
        blur=args.blur,
        scaling=args.scaling,
        debias=False,
        potentials=False,
        normalize=False,
        use_epsilon_scaling=not args.no_eps_scaling,
        last_extrapolation=True,
        allow_tf32=args.tf32,
        use_exp2=not args.no_exp2,
        autotune=not args.no_autotune,
        eps=args.eps,
        n_iters=args.n_iters,
        diameter=diameter,
    )

    gl_loss = GeomLossSamplesLoss(
        "sinkhorn",
        p=2,
        blur=args.blur,
        scaling=args.scaling,
        diameter=diameter,
        cost=_sqdist_cost_full if args.geomloss_backend == "tensorized" else "SqDist(X,Y)",
        debias=False,
        potentials=False,
        backend=args.geomloss_backend,
    )

    # ---------- ot_triton ----------
    x_t, y_t = _clone_inputs(x0, y0, requires_grad=True)

    def run_ot_forward() -> None:
        ot_loss(a, x_t, b, y_t)

    val_ot = ot_loss(a, x_t, b, y_t)

    def run_ot_backward() -> None:
        torch.autograd.grad(val_ot, (x_t, y_t), retain_graph=True)

    # Warm up (compile forward + backward outside timed region).
    run_ot_forward()
    run_ot_backward()
    torch.cuda.synchronize()

    ms_ot_fwd = _bench_once(run_ot_forward, warmup=args.warmup, rep=args.rep)
    ms_ot_bwd = _bench_once(run_ot_backward, warmup=args.warmup, rep=args.rep)

    _profile_backward("ot_triton", run_ot_backward, enabled=args.profile, steps=args.profile_steps)

    # ---------- geomloss ----------
    x_g, y_g = _clone_inputs(x0, y0, requires_grad=True)

    def run_gl_forward() -> None:
        gl_loss(a, x_g, b, y_g)

    val_gl = gl_loss(a, x_g, b, y_g)

    def run_gl_backward() -> None:
        torch.autograd.grad(val_gl, (x_g, y_g), retain_graph=True)

    run_gl_forward()
    run_gl_backward()
    torch.cuda.synchronize()

    ms_gl_fwd = _bench_once(run_gl_forward, warmup=args.warmup, rep=args.rep)
    ms_gl_bwd = _bench_once(run_gl_backward, warmup=args.warmup, rep=args.rep)

    _profile_backward("geomloss", run_gl_backward, enabled=args.profile, steps=args.profile_steps)

    # ---------- sanity ----------
    with torch.no_grad():
        out_diff = (val_ot.detach() - val_gl.detach()).abs().item()
        gx_ot, gy_ot = torch.autograd.grad(val_ot, (x_t, y_t), retain_graph=True)
        gx_gl, gy_gl = torch.autograd.grad(val_gl, (x_g, y_g), retain_graph=True)
        gx_diff = (gx_ot - gx_gl).abs().max().item()
        gy_diff = (gy_ot - gy_gl).abs().max().item()

    print(
        f"n={args.n} m={args.m} d={args.d} dtype={args.dtype} "
        f"blur={args.blur} scaling={args.scaling} "
        f"eps_scaling={'off' if args.no_eps_scaling else 'on'} "
        f"tf32={'on' if args.tf32 else 'off'}"
    )
    if args.no_eps_scaling:
        print(f"ot_triton_fixed_eps={args.eps} n_iters={args.n_iters}")
    else:
        print(f"diameter={diameter:.6g} geomloss_backend={args.geomloss_backend}")

    print(f"ot_triton_forward_ms={ms_ot_fwd:.3f} backward_ms={ms_ot_bwd:.3f}")
    print(f"geomloss_forward_ms={ms_gl_fwd:.3f} backward_ms={ms_gl_bwd:.3f}")
    print(
        f"abs_diff(cost)={out_diff:.3e} max_abs_diff(grad_x)={gx_diff:.3e} max_abs_diff(grad_y)={gy_diff:.3e}"
    )


if __name__ == "__main__":
    main()
