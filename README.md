<p align="center">
  <img src="FlashSinkhorn.png" alt="FlashSinkhorn" width="100%">
</p>

# FlashSinkhorn

**Streaming Entropic Optimal Transport in PyTorch + Triton**

FlashSinkhorn computes Sinkhorn OT using FlashAttention-style streaming—**never materializing the n×m cost matrix**—enabling **O(nd) memory** instead of O(n²).

## Features

- **FlashSinkhorn kernels** — shifted-potential formulation inspired by FlashAttention, 10-40% faster than previous Triton kernels at n >= 10k
- **Fused Triton kernels** for forward, gradient, and HVP
- **GeomLoss-compatible API** (`SamplesLoss`)
- **Analytic gradients** (no backprop through Sinkhorn iterations)
- **Hessian-vector products** via streaming CG solver
- **Half-cost support** (`half_cost=True`) for exact GeomLoss parity
- **Unbalanced/semi-unbalanced OT** via `reach` parameter
- **Large-D support** (d > 1024) with tiled gradient kernel
- **Early stopping** with convergence threshold

## Install

```bash
pip install -e .
pip install -e ".[dev]"  # with dev dependencies
```

**Requirements:** PyTorch ≥2.5, Triton ≥3.1, CUDA 12.x

## Quick Start

### Basic Usage

```python
import torch
from flash_sinkhorn import SamplesLoss

x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")

# FlashSinkhorn is the default backend (use_flashstyle=True)
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
# FlashSinkhorn with half_cost matches GeomLoss exactly
flash_loss = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True, debias=True)

# Equivalent GeomLoss call
# geomloss_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1, debias="positive")
```

### Unbalanced OT

For distributions with different total mass or outliers:

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

## FlashSinkhorn (v0.3.0)

FlashSinkhorn is a reformulated Sinkhorn kernel that uses **shifted potentials** inspired by FlashAttention. It reduces bias vector loads by 67% and elementwise operations by 78% per tile, yielding 10-40% speedups for n >= 10,000.

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

### Usage

FlashSinkhorn is enabled by default (`use_flashstyle=True`):

```python
# Default: uses FlashSinkhorn (fastest for n >= 5000)
loss = SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)

# Explicitly disable to use previous kernels
loss = SamplesLoss(loss="sinkhorn", blur=0.1, debias=True, use_flashstyle=False)
```

Low-level FlashSinkhorn API:

```python
from flash_sinkhorn.kernels import (
    sinkhorn_flashstyle_symmetric,     # Full symmetric solver
    sinkhorn_flashstyle_alternating,   # Full alternating solver
    flashsinkhorn_symmetric_step,      # Single fused iteration
    apply_plan_vec_flashstyle,         # Transport plan @ vector (shifted potentials)
    apply_plan_mat_flashstyle,         # Transport plan @ matrix (shifted potentials)
)
```

## API Reference

### SamplesLoss

```python
SamplesLoss(
    loss="sinkhorn",
    p=2,                      # Only p=2 supported (squared Euclidean)
    blur=0.05,                # Regularization: eps = blur^2
    debias=True,              # Debiased Sinkhorn divergence
    half_cost=False,          # Use ||x-y||²/2 to match GeomLoss
    reach=None,               # Unbalanced OT (None = balanced)
    reach_x=None,             # Semi-unbalanced: source marginal
    reach_y=None,             # Semi-unbalanced: target marginal
    scaling=0.5,              # Epsilon annealing factor
    n_iters=None,             # Max iterations (None = use scaling)
    threshold=None,           # Early stopping threshold
    inner_iterations=10,      # Check convergence every N iters
    use_flashstyle=True,      # Use FlashSinkhorn shifted-potential kernels
)
```

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
```

## Key Concepts

### Cost Convention

- **FlashSinkhorn default**: `C(x,y) = ||x-y||²`
- **GeomLoss p=2 default**: `C(x,y) = ||x-y||²/2`
- Use `half_cost=True` to match GeomLoss

### Memory Efficiency

FlashSinkhorn streams tiles of (x,y) and computes costs on-the-fly:
- **Forward**: O(nd) memory (no n×m cost matrix)
- **Gradient**: O(nd) memory (streaming accumulation)
- **HVP**: O(nd) memory (CG solver with streaming matvec)

### Numerical Stability

- Uses `exp2/log2` for stable LSE computation
- Safe log/division guards against underflow
- TF32 enabled by default for ~2x speedup on A100/H100 (set `allow_tf32=False` for strict FP32)
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

## Citation

If you find FlashSinkhorn useful in your research, please cite our paper:

```bibtex
@article{ye2026flashsinkhorn,
  title={FlashSinkhorn: IO-Aware Entropic Optimal Transport},
  author={Ye, Felix X.-F. and Li, Xingjie and Yu, An and Chang, Ming-Ching and Chu, Linsong and Wertheimer, Davis},
  journal={arXiv preprint arXiv:2602.03067},
  year={2026},
  url={https://arxiv.org/abs/2602.03067}
}
```

## License

MIT
