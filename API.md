# OT Triton API Reference

This document describes the `SamplesLoss` API, which provides a GeomLoss-compatible interface for computing entropic optimal transport using Triton kernels.

## SamplesLoss

```python
from ot_triton import SamplesLoss
```

The main entry point for computing Sinkhorn distances and potentials. Drop-in compatible with GeomLoss's `SamplesLoss` for common use cases.

### Basic Usage

```python
import torch
from ot_triton import SamplesLoss

# Create loss function
loss = SamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False)

# Compute OT cost between point clouds
x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")
cost = loss(x, y)  # scalar tensor

# With explicit weights
a = torch.ones(4096, device="cuda") / 4096
b = torch.ones(4096, device="cuda") / 4096
cost = loss(a, x, b, y)
```

### Constructor Parameters

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | `str` | `"sinkhorn"` | Loss type. Only `"sinkhorn"` is supported. |
| `p` | `int` | `2` | Cost exponent. Only `p=2` (squared Euclidean) is supported. |
| `blur` | `float` | `0.05` | Blur parameter (epsilon = blur²). Controls regularization strength. |
| `scaling` | `float` | `0.5` | Epsilon-scaling factor ∈ (0,1). Smaller = more iterations, better accuracy. |
| `debias` | `bool` | `False` | Debiased Sinkhorn divergence. Computes S(x,y) - S(x,x)/2 - S(y,y)/2. |
| `potentials` | `bool` | `False` | If `True`, return `(f, g)` potentials instead of cost. |
| `normalize` | `bool` | `True` | Normalize weights to sum to 1. |

#### Unbalanced / Semi-Unbalanced OT

> **Unique Feature**: Semi-unbalanced OT (`reach_x ≠ reach_y`) is **not available in GeomLoss**.

Control marginal relaxation via `reach`, `reach_x`, and `reach_y` parameters. The marginal penalty strength is `rho = reach²`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reach` | `float` or `None` | `None` | Relaxation for both marginals (symmetric unbalanced). |
| `reach_x` | `float` or `None` | `None` | Relaxation for source marginal only. |
| `reach_y` | `float` or `None` | `None` | Relaxation for target marginal only. |

**Marginal Control Modes:**

| Configuration | Behavior |
|--------------|----------|
| `reach=None` (default) | Balanced OT (strict marginal constraints) |
| `reach=r` | Symmetric unbalanced OT (both marginals relaxed equally) |
| `reach_x=r, reach_y=None` | Semi-unbalanced: relax source, strict target |
| `reach_x=None, reach_y=r` | Semi-unbalanced: strict source, relax target |
| `reach_x=r1, reach_y=r2` | Asymmetric unbalanced: different relaxation per marginal |

#### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"symmetric"` | Backend: `"symmetric"` (GeomLoss-style) or `"alternating"` (OTT-JAX-style). |
| `autotune` | `bool` | `True` | Enable Triton autotuning for kernel configs. |

**Backend comparison:**

| Backend | Iteration Style | Kernel Launches/Iter | Matches |
|---------|-----------------|---------------------|---------|
| `"symmetric"` (default) | Symmetric | 1 | GeomLoss |
| `"alternating"` | Alternating | 2 | OTT-JAX |

> **Important**: These backends implement **mathematically different algorithms** that converge to **different potentials**. Use `backend="symmetric"` for GeomLoss comparisons and `backend="alternating"` for OTT-JAX comparisons.

**Alternating backend restrictions** (required for OTT-JAX parity):
- `use_epsilon_scaling=False` (fixed epsilon only)
- `eps` and `n_iters` must be specified
- `debias=False` (debiased Sinkhorn not supported)
- `reach`/`reach_x`/`reach_y` must be `None` (unbalanced not supported)

#### Epsilon Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_epsilon_scaling` | `bool` | `True` | Use epsilon-scaling schedule (recommended). |
| `eps` | `float` or `None` | `None` | Fixed epsilon (only if `use_epsilon_scaling=False`). |
| `n_iters` | `int` or `None` | `None` | Number of iterations (only if `use_epsilon_scaling=False`). |
| `diameter` | `float` or `None` | `None` | Point cloud diameter. Auto-computed if `None`. |
| `eps_list` | `list[float]` or `None` | `None` | Custom epsilon schedule (overrides other epsilon params). |
| `last_extrapolation` | `bool` | `True` | Final full-step extrapolation (GeomLoss convention). |

#### Early Stopping (like OTT-JAX)

> **New Feature**: Threshold-based early stopping for faster convergence (2-4x speedup).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` or `None` | `None` | Convergence threshold. `None` = run all iterations. |
| `inner_iterations` | `int` | `10` | Check convergence every N iterations. |

**How it works:**
- Tracks potential change: `max(|f_new - f_old|, |g_new - g_old|)` every `inner_iterations`
- Stops when change < `threshold`
- Uses cheap max-reduction (no extra kernel launch)

**Recommended values:**
- `threshold=1e-3`: Good balance of speed (3-4x faster) and accuracy
- `threshold=1e-6`: High precision with moderate speedup (2x faster)
- `threshold=None`: Run all iterations (original behavior)
- `inner_iterations=5`: Optimal check frequency (15% faster than default 10)

#### Numerical Precision

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_tf32` | `bool` | `True` | Allow TF32 for matmuls. Set `False` for strict FP32. |
| `use_exp2` | `bool` | `True` | Use exp2/log2 (FlashAttention-like, more stable). |

#### Kernel Tuning (Advanced)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_m` | `int` or `None` | `None` | Block size for M dimension. |
| `block_n` | `int` or `None` | `None` | Block size for N dimension. |
| `block_k` | `int` or `None` | `None` | Block size for K (feature) dimension. |
| `num_warps` | `int` or `None` | `None` | Number of warps per block. |
| `num_stages` | `int` | `2` | Pipeline stages for memory latency hiding. |

#### HVP / Double Backward (Advanced)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hvp_tau2` | `float` | `1e-5` | Tikhonov regularization for HVP stability. |
| `hvp_max_cg_iter` | `int` | `300` | Max CG iterations for HVP solve. |
| `hvp_cg_rtol` | `float` | `1e-6` | Relative tolerance for CG. |
| `hvp_cg_atol` | `float` | `1e-6` | Absolute tolerance for CG. |
| `hvp_preconditioner` | `str` | `"none"` | Preconditioner: `"none"`, `"jacobi"`, `"neumann"`. |

---

## Examples

### Balanced OT (Default)

```python
from ot_triton import SamplesLoss
import torch

loss = SamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False)

x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")

cost = loss(x, y)
print(f"OT cost: {cost.item():.6f}")
```

### Unbalanced OT (Symmetric)

```python
# Relax both marginals equally
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    reach=1.0,  # rho = reach² = 1.0
    debias=False,
)
cost = loss(x, y)
```

### Semi-Unbalanced OT

```python
# Relax only source marginal (target is strict)
loss_relax_source = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    reach_x=1.0,   # Relax source
    reach_y=None,  # Strict target
    debias=False,
)

# Relax only target marginal (source is strict)
loss_relax_target = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    reach_x=None,  # Strict source
    reach_y=1.0,   # Relax target
    debias=False,
)

# Asymmetric: different relaxation for each
loss_asymmetric = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    reach_x=0.5,  # Strong source relaxation
    reach_y=5.0,  # Weak target relaxation
    debias=False,
)
```

### Get Potentials

```python
loss = SamplesLoss("sinkhorn", blur=0.1, potentials=True)

a = torch.ones(4096, device="cuda") / 4096
b = torch.ones(4096, device="cuda") / 4096

f, g = loss(a, x, b, y)  # Returns potentials instead of cost
print(f"f shape: {f.shape}, g shape: {g.shape}")
```

### Gradient Computation

```python
x = torch.randn(4096, 64, device="cuda", requires_grad=True)
y = torch.randn(4096, 64, device="cuda")

loss = SamplesLoss("sinkhorn", blur=0.1)
cost = loss(x, y)

# Analytic gradient (no backprop through Sinkhorn iterations)
grad_x = torch.autograd.grad(cost, x)[0]
```

### Hessian-Vector Product (HVP)

```python
x = torch.randn(4096, 64, device="cuda", requires_grad=True)
y = torch.randn(4096, 64, device="cuda")
v = torch.randn_like(x)

loss = SamplesLoss("sinkhorn", blur=0.1)
cost = loss(x, y)

# Double backward for HVP
grad_x = torch.autograd.grad(cost, x, create_graph=True)[0]
hvp = torch.autograd.grad((grad_x * v).sum(), x)[0]
```

### Batched Inputs

```python
# Batched point clouds (B, N, D)
x_batch = torch.randn(8, 1024, 64, device="cuda")
y_batch = torch.randn(8, 1024, 64, device="cuda")

loss = SamplesLoss("sinkhorn", blur=0.1)
costs = loss(x_batch, y_batch)  # Returns (B,) tensor
```

### Strict FP32 (No TF32)

```python
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    allow_tf32=False,  # Disable TF32 for strict FP32
    use_exp2=True,     # Keep exp2 for stability
)
```

### Fixed Epsilon (No Scaling)

```python
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    use_epsilon_scaling=False,
    eps=0.1,
    n_iters=50,
)
```

### OTT-JAX Backend (Alternating Updates)

```python
# Use OTT-style alternating updates for OTT-JAX parity
loss = SamplesLoss(
    "sinkhorn",
    backend="alternating",              # Use OTT-JAX-style kernel
    use_epsilon_scaling=False,  # Required for Alternating backend
    eps=0.1,                    # Fixed epsilon
    n_iters=10,                 # Fixed iterations
    allow_tf32=False,           # Strict fp32 for parity
)

x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")
cost = loss(x, y)  # Matches OTT-JAX's sinkhorn() output
```

**When to use Alternating backend:**
- Benchmarking against OTT-JAX
- Reproducing OTT-JAX results exactly
- Research comparing iteration styles

**When to use symmetric backend (default):**
- Benchmarking against GeomLoss
- Debiased Sinkhorn divergence
- Unbalanced/semi-unbalanced OT
- Epsilon-scaling schedules

### Custom Epsilon Schedule

```python
# Geometric schedule from large to small epsilon
eps_schedule = [1.0, 0.5, 0.25, 0.125, 0.1, 0.1, 0.1]

loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    eps_list=eps_schedule,
)
```

### Early Stopping (OTT-JAX Style)

```python
# Enable early stopping for 2-4x speedup
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    use_epsilon_scaling=False,
    eps=0.1,
    n_iters=100,              # Max iterations
    threshold=1e-3,           # Stop when potential change < 1e-3
    inner_iterations=10,      # Check every 10 iterations
)

# Typical behavior: converges in ~20-30 iterations instead of 100
cost = loss(x, y)
```

**Performance (n=1000, d=784):**

| Threshold | Time | Speedup | Loss Parity |
|-----------|------|---------|-------------|
| None (all 100 iters) | 54 ms | 1.0x | — |
| 1e-3 | 16 ms | **3.4x** | 0.00% |
| 1e-6 | 26 ms | **2.1x** | 0.00% |

---

## Low-Level Kernel API

For direct access to the Triton kernels:

### GeomLoss-Style (Symmetric Updates)

```python
from ot_triton.kernels import sinkhorn_geomloss_symmetric_potentials_sqeuclid

f, g = sinkhorn_geomloss_symmetric_potentials_sqeuclid(
    x, y, a, b,
    blur=0.1,
    scaling=0.5,
    use_epsilon_scaling=True,
    last_extrapolation=True,
    allow_tf32=False,
    use_exp2=True,
    # Semi-unbalanced parameters
    reach_x=1.0,
    reach_y=None,
    # Kernel tuning
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    autotune=False,
)
```

### OTT-Style (Alternating Updates)

```python
from ot_triton.kernels.sinkhorn_triton_ott_sqeuclid import sinkhorn_potentials_sqeuclid

# OTT-style uses log-weights (loga, logb) instead of weights (a, b)
import torch
loga = torch.log(a)  # a = uniform weights
logb = torch.log(b)

f, g = sinkhorn_potentials_sqeuclid(
    x, y, loga, logb,
    eps=0.1,       # Fixed epsilon
    n_iters=10,    # Fixed iterations
    autotune=True,
    allow_tf32=False,
)
```

---

## Comparison with GeomLoss

| Feature | OT Triton | GeomLoss |
|---------|-----------|----------|
| Cost function | Squared Euclidean only | Multiple (Euclidean, Laplacian, etc.) |
| Backend | Triton (symmetric, O(nd) memory) | PyTorch (tensorized, symmetric, multiscale) |
| Unbalanced OT | Yes (`reach`, `reach_x`, `reach_y`) | Yes (`reach`) |
| Semi-unbalanced OT | Yes (`reach_x` ≠ `reach_y`) | No |
| Debiased Sinkhorn | Yes (`debias=True`) | Yes |
| Early stopping | Yes (`threshold`, 2-4x speedup) | No (epsilon-scaling only) |
| Gradient | Analytic (no backprop) | Analytic (no backprop) |
| HVP | Yes (CG solver) | No |
| Memory | O(nd) streaming | O(n + m) symmetric or O(nm) tensorized |

### Migration from GeomLoss

```python
# GeomLoss
from geomloss import SamplesLoss as GeomLossSamplesLoss
loss_geo = GeomLossSamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False)

# OT Triton (drop-in replacement for balanced OT)
from ot_triton import SamplesLoss
loss_tri = SamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False)

# Both work the same way
cost_geo = loss_geo(x, y)
cost_tri = loss_tri(x, y)
```

---

## Notes

### Cost and Epsilon Convention

OT Triton uses **full squared Euclidean cost**: `C(x,y) = ||x - y||²`

The epsilon schedule uses `eps = blur^p` (same as GeomLoss), providing exact potential parity when using matching cost functions:

```python
# For potential parity with OT Triton, use full squared cost
def full_sqdist_cost(x, y):
    return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)

loss_geo = GeomLossSamplesLoss("sinkhorn", cost=full_sqdist_cost, ...)
loss_tri = SamplesLoss("sinkhorn", ...)

# Potentials will now match exactly (rtol=1e-4)
```

### Potential Conventions

**GeomLoss convention:**
```
P = diag(a) * exp((f + g - C) / eps) * diag(b)
```

**OTT convention:**
```
P = exp((f_hat + g_hat - C) / eps)
where f_hat = f + eps * log(a), g_hat = g + eps * log(b)
```

Convert between conventions:
```python
from ot_triton.hvp import geomloss_to_ott_potentials
f_hat, g_hat = geomloss_to_ott_potentials(f, g, a, b, eps=0.1)
```

### Numerical Stability

- Use `use_exp2=True` (default) for FlashAttention-like exp2/log2 stability
- Set `allow_tf32=False` for strict FP32 when comparing against CPU references
- For very small `blur` values, increase `n_iters` or use epsilon-scaling
