# FlashSinkhorn

**Streaming Entropic Optimal Transport in PyTorch + Triton**

FlashSinkhorn computes Sinkhorn OT using FlashAttention-style streaming—**never materializing the n×m cost matrix**—enabling **O(nd) memory** instead of O(n²).

## Features

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
from ot_triton import SamplesLoss

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
)
```

### Low-Level API

```python
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
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
- TF32 disabled by default for reproducibility

## Benchmarks

Compare FlashSinkhorn against GeomLoss (KeOps) and OTT-JAX.

**Install benchmark dependencies:**
```bash
pip install geomloss pykeops ott-jax jax[cuda12]
```

**Run benchmarks:**
```bash
# Forward pass benchmark
python -m ot_triton.bench.bench_forward --sizes 5000,10000,20000 --dims 64 --verify

# Backward pass benchmark
python -m ot_triton.bench.bench_backward --sizes 5000,10000,20000 --dims 64 --verify

# Quick test (small size)
python -m ot_triton.bench.bench_forward --sizes 5000 --dims 4 --verify

# Run only FlashSinkhorn (skip GeomLoss/OTT-JAX)
python -m ot_triton.bench.bench_forward --sizes 10000 --dims 64 --no-geomloss --no-ott
```

Results are saved to `output/paper_benchmarks/forward/` and `output/paper_benchmarks/backward/`.

## License

MIT
