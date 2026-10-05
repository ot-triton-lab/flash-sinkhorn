<p align="center">
  <img src="https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/FlashSinkhorn.png" alt="FlashSinkhorn" width="100%">
</p>

# FlashSinkhorn

[![PyPI](https://img.shields.io/pypi/v/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![Python](https://img.shields.io/pypi/pyversions/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Kernels](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kernels%20Hub-ffb000)](https://huggingface.co/kernels/yexf308/flash-sinkhorn)
[![Kernel build](https://github.com/ot-triton-lab/flash-sinkhorn/actions/workflows/build-kernel.yml/badge.svg?branch=main)](https://github.com/ot-triton-lab/flash-sinkhorn/actions/workflows/build-kernel.yml)

**Streaming Entropic Optimal Transport in PyTorch + Triton**

FlashSinkhorn computes Sinkhorn OT using FlashAttention-style streaming—**never materializing the n×m cost matrix**—enabling **O(nd) memory** instead of O(n²); the multiscale backend also stores its block masks.

## News

- **v0.4.1**: OT and derivative corrections, consistent TF32 coordinate rounding,
  and 15 Jupyter notebooks covering introductory lessons and applications. Available on
  [PyPI](https://pypi.org/project/flash-sinkhorn/0.4.1/) and the
  [Kernels Hub](https://huggingface.co/kernels/yexf308/flash-sinkhorn/tree/v1).
  See the [release notes](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/CHANGELOG.md).
- **2026-10** Released [v0.4.0](https://github.com/ot-triton-lab/flash-sinkhorn/releases/tag/v0.4.0): a multiscale backend for large 3-D point clouds (`SamplesLoss(backend="multiscale", tol=...)`).
- **2026-05** 🎉 FlashSinkhorn accepted to **ICML 2026 as an Oral** (top 0.7%, 168 of ~24k submissions).
- **2026-04** Released [v0.3.3](https://github.com/ot-triton-lab/flash-sinkhorn/releases/tag/v0.3.3).
- **2026-02** FlashSinkhorn (v0.3.0) released; preprint on [arXiv](https://arxiv.org/abs/2602.03067).
- **2025-12** Initial release (v0.1.0).

## Features

- **FlashSinkhorn kernels** — shifted-potential formulation inspired by FlashAttention. On A100 GPUs, achieves up to **32× forward-pass** and **161× end-to-end** speedups over state-of-the-art online baselines on point-cloud OT
- **Fused Triton kernels** for forward, gradient, and HVP
- **GeomLoss-compatible API** (`SamplesLoss`)
- **Analytic gradients** (no backprop through Sinkhorn iterations)
- **Hessian-vector products** via streaming CG solver
- **C-transform / semi-dual OT** — streaming hard-argmin Kantorovich c-transform with analytic Danskin gradients, for WPP and other semi-dual methods (`c_transform_fwd`, `c_transform_cost`)
- **Half-cost support** (`half_cost=True`) to match GeomLoss's squared-Euclidean cost convention
- **Unbalanced/semi-unbalanced OT** via `reach` parameter
- **Large-D support** (d > 1024) with tiled gradient kernel
- **Early stopping** with convergence threshold
- **Multiscale backend** (`SamplesLoss(backend="multiscale", tol=...)`) for balanced, forward-only OT on large 3-D point clouds: Sinkhorn on Morton cells, then block-sparse fine updates as needed, until an empirical sampled marginal check passes `tol` (otherwise it warns and returns its best candidate); see [API.md](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/API.md#multiscale-backend-large-3-d-point-clouds) for its options and measured limits

## Install

Install or upgrade to **v0.4.1** from [PyPI](https://pypi.org/project/flash-sinkhorn/):

```bash
pip install --upgrade flash-sinkhorn
```

To run the tutorials, install the notebook dependencies and clone the example files:

```bash
pip install --upgrade "flash-sinkhorn[notebooks]"
git clone https://github.com/ot-triton-lab/flash-sinkhorn.git
cd flash-sinkhorn
jupyter lab examples/getting_started/first_steps.ipynb
```

For the plotting scripts, use `pip install --upgrade "flash-sinkhorn[examples]"`.
The examples use the installed package, which can be upgraded through pip.
**Requirements:** Python ≥3.9, PyTorch ≥2.5, Triton ≥3.1, CUDA 12.x.

The [Hugging Face Kernels Hub](#hugging-face-kernels-hub) also serves v0.4.1. Its `kernels` client downloads
the package on first use.

## Quick Start

### Basic Usage

```python
import torch
from flash_sinkhorn import SamplesLoss

x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")

loss = SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
cost = loss(x, y)
```

### Gradient Flow

```python
x = torch.randn(4096, 64, device="cuda", requires_grad=True)
y = torch.randn(4096, 64, device="cuda")

loss = SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
cost = loss(x, y)
grad_x = torch.autograd.grad(cost, x)[0]  # Analytic gradient
```

### Matching GeomLoss Conventions

Use `half_cost=True` to match GeomLoss's cost convention:

```python
# FlashSinkhorn with GeomLoss's squared-Euclidean cost convention
flash_loss = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True, debias=True)

# GeomLoss with the same pair-cost and debiasing conventions
# geomloss_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1, debias=True)
```

Iteration schedules and arithmetic precision can still produce numerical differences. See the
[conventions notebook](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/choosing_parameters/plot_conventions.ipynb)
for a controlled comparison.

### Unbalanced OT

For matching with relaxed marginals:

```python
loss = SamplesLoss(
    loss="sinkhorn",
    blur=0.1,
    debias=True,
    reach=1.0,  # Unbalanced OT with KL penalty
)
```

### Semi-Unbalanced OT

Different constraints for source vs target:

```python
loss = SamplesLoss(
    loss="sinkhorn",
    blur=0.1,
    reach_x=1.0,   # Relax source marginal
    reach_y=None,  # Keep target marginal strict (balanced)
)
```

### Early Stopping

```python
loss = SamplesLoss(
    loss="sinkhorn",
    blur=0.1,
    use_epsilon_scaling=False,
    eps=0.01,
    n_iters=100,
    threshold=1e-3,       # Stop when potential change < threshold
    inner_iterations=10,  # Check every N iterations
)
```

### Hessian-Vector Product

```python
x = torch.randn(4096, 64, device="cuda", requires_grad=True)
y = torch.randn(4096, 64, device="cuda")
v = torch.randn_like(x)

loss = SamplesLoss(loss="sinkhorn", blur=0.1, debias=False)
cost = loss(x, y)

# First-order gradient
grad_x = torch.autograd.grad(cost, x, create_graph=True)[0]

# HVP via double backward (uses streaming CG solver)
hvp = torch.autograd.grad((grad_x * v).sum(), x)[0]
```

Double backward currently supports derivatives with respect to `x` with `debias=False`, no gradient on `y`,
and no label cost. See the [HVP notebook](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/advanced/plot_hvp_newton.ipynb)
for convergence checks and an optimization example.

### Large 3-D Point Clouds (Multiscale Backend)

For balanced OT between large three-dimensional point clouds (forward only):

```python
x = torch.rand(2**20, 3, device="cuda")   # (n, 3) point clouds
y = torch.rand(2**20, 3, device="cuda")

loss = SamplesLoss(loss="sinkhorn", blur=0.05, backend="multiscale", tol=5e-3)
cost = loss(x, y)
info = loss.last_multiscale_info   # info.accepted, info.stop, info.detail
```

The solve returns once an empirical sampled check finds the marginal residual below `tol`
(a stopping rule, not a bound); otherwise it warns and returns its best candidate with
`info.accepted == False`.
See [API.md](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/API.md#multiscale-backend-large-3-d-point-clouds) for its options and measured limits.

## Examples

[![Gradient flow of 30,000 particles](https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/examples/getting_started/images/gradient_flow.gif)](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/README.md)

Start with [From two point clouds to a differentiable loss](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/getting_started/first_steps.ipynb),
then [Choose parameters and check the result](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/choosing_parameters/guide.ipynb).
The gallery contains **15 notebooks**: one executable introductory lesson, one parameter guide, and thirteen
applications. They combine explanations, code and labelled reference figures. Install
`pip install --upgrade "flash-sinkhorn[notebooks]"`, clone the example files, and open `jupyter lab`
to run the code cells on a CUDA GPU. The
[case gallery](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/README.md) contains thirteen runnable
examples, from 10,000 points to millions: gradient flows, transport and attribute transfer, relaxed marginals,
labelled datasets, embeddings, batches, multiscale solves, Hessian-vector products and semi-discrete transport.

Reference figures come from previous script runs. Running the code cells produces fresh output and figures.
Each application also provides a `.py` script; notebooks are generated from the same sources.

## Kernel Design

FlashSinkhorn is a reformulated Sinkhorn kernel that uses **shifted potentials** inspired by FlashAttention. It reduces bias vector loads by 67% and elementwise operations by 78% per tile, and improves scalability on OT-based downstream tasks.

### How It Works

Standard Sinkhorn loads 3 bias vectors per tile (g, log_b, y²). FlashSinkhorn precomputes a single fused bias `u = (g_shifted + eps*log(b)) / eps` and uses raw coordinates with an inline scale factor, matching FlashAttention's score interface exactly.

### Earlier Benchmarks (v0.3.0, d=64, A100-80GB, 100 iterations)

These measurements compare v0.3.0 with v0.2.0. To measure the current version on your hardware, use the
[benchmark commands](#benchmarks).

**Symmetric solver (vs v0.2.0 GeomLoss-style kernel):**

| n | v0.2.0 | v0.3.0 | Speedup |
|---|--------|--------|---------|
| 50,000 | 1730 ms | 1450 ms | **1.19x** |
| 10,000 | 88 ms | 61 ms | **1.43x** |
| 5,000 | 25 ms | 24 ms | 1.04x |

**Alternating solver (vs v0.2.0 OTT-style kernel, 10 iterations):**

| n | v0.2.0 | v0.3.0 | Speedup |
|---|--------|--------|---------|
| 50,000 | 137.9 ms | 102.6 ms | **1.34x** |
| 20,000 | 25.7 ms | 21.7 ms | **1.19x** |
| 10,000 | 8.9 ms | 8.3 ms | **1.07x** |

### Low-Level API

Low-level FlashSinkhorn API:

```python
from flash_sinkhorn.kernels import (
    sinkhorn_flashstyle_symmetric,     # Full symmetric solver
    sinkhorn_flashstyle_alternating,   # Full alternating solver
    apply_plan_vec_flashstyle,         # Transport plan @ vector (shifted potentials)
    apply_plan_mat_flashstyle,         # Transport plan @ matrix (shifted potentials)
)
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    flashsinkhorn_symmetric_step,      # Single fused iteration
)
```

## API Reference

### SamplesLoss

```python
SamplesLoss(
    loss="sinkhorn",
    p=2,                      # Only p=2 supported (squared Euclidean)
    blur=0.05,                # Regularization: eps = blur^2
    debias=False,             # True: debiased Sinkhorn divergence
    half_cost=False,          # Use ||x-y||²/2 to match GeomLoss
    reach=None,               # Unbalanced OT (None = balanced)
    reach_x=None,             # Semi-unbalanced: source marginal
    reach_y=None,             # Semi-unbalanced: target marginal
    scaling=0.5,              # Epsilon annealing factor
    n_iters=None,             # Max iterations (None = use scaling)
    threshold=None,           # Early stopping threshold
    inner_iterations=10,      # Check convergence every N iters
    backend="symmetric",      # "symmetric", "alternating" or "multiscale"
    tol=None,                 # Multiscale only: target marginal residual (default 5e-3)
    allow_tf32=True,           # TF32 cost dot products
    truncate_tf32=None,        # Round centred coordinates; None follows allow_tf32
    autotune=True,             # Tune dense forward/first-derivative kernels on first use
)
```

With `backend="multiscale"`, `n_iters` and `threshold` are rejected and `scaling` and `inner_iterations` do not
apply; `tol` sets the stopping point. `backend="alternating"` requires `use_epsilon_scaling=False`, `eps` and `n_iters`.

### Low-Level API

```python
# FlashSinkhorn (recommended)
from flash_sinkhorn.kernels import (
    sinkhorn_flashstyle_symmetric,
    sinkhorn_flashstyle_alternating,
    apply_plan_vec_flashstyle,
    apply_plan_mat_flashstyle,
)

# Legacy kernels (still available)
from flash_sinkhorn.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from flash_sinkhorn.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials

# Multiscale backend (large 3-D point clouds)
from flash_sinkhorn.multiscale import preprocess_coordinates, check_limits, solve_multiscale
```

### C-Transform / Semi-Dual OT

The non-entropic Kantorovich **c-transform** (hard argmin) and its differentiable
semi-dual objective — building blocks for the Wasserstein Patch Prior (WPP) and other
semi-dual OT methods. Like the Sinkhorn kernels, the `n×m` cost matrix is never
materialized (O(nd) memory), and gradients are analytic (no entropic smoothing).

```python
import torch
from flash_sinkhorn import c_transform_fwd, c_transform_cost

x = torch.randn(4096, 64, device="cuda")     # source points [n, d]
y = torch.randn(2048, 64, device="cuda")     # target points [m, d]
psi = torch.randn(2048, device="cuda")        # dual potential on y [m]

# (1) Raw c-transform: values + argmin indices (non-differentiable)
#     c_i = min_j [ cost_scale * ||x_i - y_j||² - psi_j ]
c_values, argmin_idx = c_transform_fwd(x, y, psi, cost_scale=1.0)

# (2) Differentiable semi-dual objective (Danskin's theorem):
#     L(x, psi) = Σ_i a_i * c^psi(x_i) + Σ_j b_j * psi_j   (a, b default uniform)
x = x.requires_grad_(True)
psi = psi.requires_grad_(True)
loss = c_transform_cost(x, y, psi, cost_scale=1.0)
grad_x, grad_psi = torch.autograd.grad(loss, [x, psi])
```

Differentiable w.r.t. `x` and `psi` (not `y`). Use `cost_scale=0.5` for the
`||x-y||²/2` (GeomLoss) convention. With TF32 (the default) the coordinates are first
rounded to TF32; pass `allow_tf32=False` for the cells of the points as given. For the raw
streaming kernel, see `flash_sinkhorn.kernels.c_transform_kernel`.

## Key Concepts

### Cost Convention

- **FlashSinkhorn default**: `C(x,y) = ||x-y||²`
- **GeomLoss p=2 default**: `C(x,y) = ||x-y||²/2`
- Use `half_cost=True` to match GeomLoss

### Memory Efficiency

FlashSinkhorn streams tiles of (x,y) and computes costs on-the-fly:
- **Forward**: O(nd) memory (no n×m cost matrix); the multiscale backend also stores its block masks
- **Gradient**: O(nd) memory (streaming accumulation)
- **HVP**: O(nd) memory (CG solver with streaming matvec)

### Numerical Stability

- Online log-sum-exp with a running maximum, in exp2/log2
- Safe log/division guards against underflow
- TF32 is enabled by default for cost dot products. `truncate_tf32=None` follows `allow_tf32`: dense backends
  round centred FP32/FP64 input coordinates to TF32 before padding, with an identity derivative for gradients and
  supported HVPs; fp16/bf16 inputs receive no additional rounding. Norms and dot products then use the same rounded
  cloud, with FP32 arithmetic error remaining. Set `truncate_tf32=False` to keep unrounded centred coordinates,
  or `allow_tf32=False` for FP32 dot products and no default rounding. Multiscale always rounds and rejects either
  option set to `False`.
- HVP (double backward) uses strict FP32 internally for numerical stability

## Benchmarks

Compare FlashSinkhorn against GeomLoss (KeOps) and OTT-JAX.

**Install benchmark dependencies:**
```bash
pip install geomloss pykeops ott-jax jax[cuda12]
```

**Run benchmarks:**
```bash
# Forward pass benchmark
python -m flash_sinkhorn.bench.bench_forward --sizes 5000,10000,20000 --dims 64 --verify

# Backward pass benchmark
python -m flash_sinkhorn.bench.bench_backward --sizes 5000,10000,20000 --dims 64 --verify

# Quick test (small size)
python -m flash_sinkhorn.bench.bench_forward --sizes 5000 --dims 4 --verify

# Run only FlashSinkhorn (skip GeomLoss/OTT-JAX)
python -m flash_sinkhorn.bench.bench_forward --sizes 10000 --dims 64 --no-geomloss --no-ott
```

Results are saved to `output/paper_benchmarks/forward/` and `output/paper_benchmarks/backward/`.

## Hugging Face Kernels Hub

The [Hub v1 package](https://huggingface.co/kernels/yexf308/flash-sinkhorn/tree/v1) contains FlashSinkhorn
**0.4.1**. Install the client, then load the kernel:

```bash
pip install kernels
```

```python
from kernels import get_kernel

flash_sinkhorn = get_kernel("yexf308/flash-sinkhorn", version=1)
loss = flash_sinkhorn.SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
```

`version=1` selects the Hub kernel API version; it is separate from the Python package version `0.4.1`.
The kernel runs on CUDA tensors and needs the same PyTorch and Triton dependencies as the source package.

Builds and uploads use the [GitHub Actions workflow](https://github.com/ot-triton-lab/flash-sinkhorn/actions/workflows/build-kernel.yml).
For local builds, use the builder revision pinned in
[`flake.nix`](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/flake.nix) and
[the workflow](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/.github/workflows/build-kernel.yml),
following the [kernel-builder instructions](https://huggingface.co/docs/kernels/builder/build).

## Citation

If you find FlashSinkhorn useful in your research, please cite our paper:

```bibtex
@inproceedings{ye2026flashsinkhorn,
  title={FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU},
  author={Ye, Felix X.-F. and Li, Xingjie and Yu, An and Chang, Ming-Ching and Chu, Linsong and Wertheimer, Davis},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  note={Oral Presentation},
  year={2026},
  url={https://arxiv.org/abs/2602.03067}
}
```

## License

MIT
