<p align="center">
  <img src="https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/FlashSinkhorn.png" alt="FlashSinkhorn" width="100%">
</p>

# FlashSinkhorn

[![PyPI](https://img.shields.io/pypi/v/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![Python](https://img.shields.io/pypi/pyversions/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Kernels](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kernels%20Hub-ffb000)](https://huggingface.co/kernels/yexf308/flash-sinkhorn)

**Streaming Entropic Optimal Transport in PyTorch + Triton**

FlashSinkhorn computes Sinkhorn OT using FlashAttention-style streaming—**never materializing the n×m cost matrix**—enabling **O(nd) memory** instead of O(n²); the multiscale backend also stores its block masks.

## News

- **v0.4.1 (unreleased)**: OT and derivative corrections, consistent TF32 coordinate rounding, and a gallery of
  runnable examples. See the [release notes](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/CHANGELOG.md).
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

```bash
pip install flash-sinkhorn

# From source (development)
pip install -e ".[dev]"
```

**Requirements:** PyTorch ≥2.5, Triton ≥3.1, CUDA 12.x

Prefer not to install the package? FlashSinkhorn is also published on the
[Hugging Face Kernels Hub](https://huggingface.co/kernels/yexf308/flash-sinkhorn)
and can be loaded on the fly with `kernels.get_kernel` — no cloning or
`pip install` required. See [Hugging Face Kernels Hub](#hugging-face-kernels-hub) below.

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

### GeomLoss Parity

Use `half_cost=True` to match GeomLoss's cost convention:

```python
# FlashSinkhorn with GeomLoss's squared-Euclidean cost convention
flash_loss = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True, debias=True)

# Equivalent GeomLoss call
# geomloss_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1, debias=True)
```

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

loss = SamplesLoss(loss="sinkhorn", blur=0.1)
cost = loss(x, y)

# First-order gradient
grad_x = torch.autograd.grad(cost, x, create_graph=True)[0]

# HVP via double backward (uses streaming CG solver)
hvp = torch.autograd.grad((grad_x * v).sum(), x)[0]
```

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
These Jupyter notebooks explain the concepts with code and reference figures together. Install
`pip install -e ".[notebooks]"` from a clone and open `jupyter lab` to run the code cells on a CUDA GPU. The
[case gallery](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/README.md) contains thirteen runnable
examples, from 10,000 points to millions: gradient flows, transport and attribute transfer, relaxed marginals,
labelled datasets, embeddings, batches, multiscale solves, Hessian-vector products and semi-discrete transport.

## FlashSinkhorn (v0.3.0)

FlashSinkhorn is a reformulated Sinkhorn kernel that uses **shifted potentials** inspired by FlashAttention. It reduces bias vector loads by 67% and elementwise operations by 78% per tile, and improves scalability on OT-based downstream tasks.

### How It Works

Standard Sinkhorn loads 3 bias vectors per tile (g, log_b, y²). FlashSinkhorn precomputes a single fused bias `u = (g_shifted + eps*log(b)) / eps` and uses raw coordinates with an inline scale factor, matching FlashAttention's score interface exactly.

### Performance (d=64, A100-80GB, 100 iterations)

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

FlashSinkhorn can be loaded as a JIT-compiled Triton kernel from the Hugging
Face Kernels Hub. To build and publish the configured kernel project, install
the [kernel builder](https://huggingface.co/docs/kernels/builder/build) and run:

```bash
kernel-builder check-config
kernel-builder build-and-copy -L
kernel-builder build-and-upload
```

The build configuration publishes version 1 to
[`yexf308/flash-sinkhorn`](https://huggingface.co/kernels/yexf308/flash-sinkhorn).
Consumers can load it without cloning this repository:

```python
from kernels import get_kernel

flash_sinkhorn = get_kernel("yexf308/flash-sinkhorn", version=1)
loss = flash_sinkhorn.SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
```

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
