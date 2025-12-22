import argparse

import torch

from ot_triton import SamplesLoss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA profile a single ot_triton multiscale Sinkhorn forward pass."
    )
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=100000)
    parser.add_argument("--d", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--blur", type=float, default=0.05)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--truncate", type=float, default=5.0)
    parser.add_argument(
        "--blocksparse-backend",
        type=str,
        default="auto",
        choices=["auto", "padded", "taskcsr", "taskcsr_bucketed", "ranges_atomic"],
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-exp2", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Wrap a single forward pass in torch.cuda.profiler.start/stop. "
            "Use with Nsight Systems: nsys profile --capture-range=cudaProfilerApi "
            "--stop-on-range-end=true ..."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=40)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))

    x = torch.randn(args.n, args.d, device=device, dtype=torch.float32)
    y = torch.randn(args.m, args.d, device=device, dtype=torch.float32)
    a = torch.rand(args.n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(args.m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    allow_tf32 = not args.no_tf32
    use_exp2 = not args.no_exp2

    loss = SamplesLoss(
        "sinkhorn",
        backend="multiscale",
        debias=False,
        blur=float(args.blur),
        scaling=float(args.scaling),
        truncate=float(args.truncate),
        max_coarse_levels=1,
        multiscale_blocksparse_backend=str(args.blocksparse_backend),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
    )

    def run():
        return loss(a, x, b, y)

    # Warmup: exclude JIT compilation and allocator effects.
    out = run()
    torch.cuda.synchronize()

    prof = None
    if args.cuda_profiler_range:
        torch.cuda.profiler.start()
        out = run()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    else:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_:
            out = run()
            torch.cuda.synchronize()
            prof = prof_

    print(
        f"n={args.n} m={args.m} d={args.d} blur={args.blur} scaling={args.scaling} "
        f"truncate={args.truncate} blocksparse_backend={args.blocksparse_backend} "
        f"tf32={'on' if allow_tf32 else 'off'} exp2={'on' if use_exp2 else 'off'} "
        f"loss={float(out.detach().cpu())}"
    )
    if prof is not None:
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=int(args.row_limit)
            )
        )


if __name__ == "__main__":
    main()
