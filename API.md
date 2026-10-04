# FlashSinkhorn API Reference

This document describes the `SamplesLoss` API, which provides a GeomLoss-compatible interface for computing entropic optimal transport using Triton kernels.

## SamplesLoss

```python
from flash_sinkhorn import SamplesLoss
```

The main entry point for computing Sinkhorn distances and potentials. Its calling interface follows GeomLoss's
`SamplesLoss`, but the defaults differ (full cost `‖x−y‖²`, `debias=False`): match the cost, debiasing, weights and
solver settings when comparing the two.

### Basic Usage

```python
import torch
from flash_sinkhorn import SamplesLoss

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
| `scaling` | `float` | `0.5` | Epsilon-scaling factor ∈ (0,1). Closer to 1 = more annealing steps. At small `blur` the default 0.5 can leave the solve far from converged, as in GeomLoss; no value guarantees convergence. |
| `debias` | `bool` | `False` | Debiased cost OT(x,y) - OT(x,x)/2 - OT(y,y)/2. The symmetric and alternating backends run the three problems on the schedule of OT(x,y), as GeomLoss does; the multiscale backend solves each on its own. With `reach_x != reach_y` it is not a divergence: it need not be symmetric, minimal at x = y, or nonnegative. |
| `potentials` | `bool` | `False` | If `True`, return the potentials `(f, g)` of OT(x,y) instead of the cost, whatever `debias` says (GeomLoss subtracts the self potentials when `debias=True`). |
| `normalize` | `bool` | `True` | Normalize weights to sum to 1. With `False`, supply probability weights that already sum to 1 on each side: dense costs and weight gradients do not support arbitrary input mass totals. |

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

With probability weights, one common reach, the same cost and converged solves, values and potentials agree with
GeomLoss. GeomLoss's coordinate gradients (checked with 0.3.1) are (rho + eps/2) / (rho + eps) times the OT
gradient, because autograd never calls the backward its `UnbalancedWeight` defines; flash-sinkhorn's carry no such
factor.

#### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"symmetric"` | Backend: `"symmetric"` (GeomLoss-style), `"alternating"` (OTT-JAX-style) or `"multiscale"` (large 3-D point clouds, see below). |
| `tol` | `float` or `None` | `None` | Target marginal residual `max(‖P1−a‖₁, ‖Pᵀ1−b‖₁)` for `backend="multiscale"` (default `5e-3`); rejected by the other backends. |
| `autotune` | `bool` | `True` | Enable Triton autotuning for dense forward and first-derivative kernels. The `SamplesLoss` HVP path currently uses its own default tuning. |

**Backend comparison:**

| Backend | Iteration Style | LSE kernel launches/iter | Matches |
|---------|-----------------|--------------------------|---------|
| `"symmetric"` (default) | Symmetric | 1 (fused), or 2 for large n without a label cost | GeomLoss |
| `"alternating"` | Alternating | 2 | OTT-JAX |

> **Important**: Both backends solve the same OT problem with different update orders. At convergence they give the same plan (balanced potentials up to an additive constant); unconverged iterates differ. Use `backend="symmetric"` for GeomLoss comparisons and `backend="alternating"` for OTT-JAX comparisons.

**Alternating backend restrictions** (required for OTT-JAX parity):
- `use_epsilon_scaling=False` (fixed epsilon only)
- `eps` and `n_iters` must be specified
- no label cost (`label_cost_matrix`)

Debiasing and relaxed marginals are supported.

#### Epsilon Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_epsilon_scaling` | `bool` | `True` | Use epsilon-scaling schedule (recommended). |
| `eps` | `float` or `None` | `None` | Fixed epsilon (only if `use_epsilon_scaling=False`). |
| `n_iters` | `int` or `None` | `None` | Length of the fixed-`eps` schedule (`use_epsilon_scaling=False`). An automatic or explicit schedule is truncated to it, never extended. The symmetric backend adds an initialization and, if enabled, the final extrapolation; early stopping can end the run sooner. |
| `diameter` | `float` or `None` | `None` | Point cloud diameter. Auto-computed if `None`. |
| `eps_list` | `list[float]` or `None` | `None` | Custom epsilon schedule, replacing the automatic one (`n_iters` truncates it). The alternating backend runs at most as many iterations as the truncated list has entries, all at its last epsilon. |
| `last_extrapolation` | `bool` | `True` | Final full-step extrapolation (GeomLoss convention); symmetric backend only. |

#### Early Stopping (like OTT-JAX)

A potential-change threshold can stop the iterations early; the time it saves depends on the problem and on the
cost of the checks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` or `None` | `None` | Convergence threshold. `None` = run all iterations. |
| `inner_iterations` | `int` | `10` | Check convergence every N iterations. |

**How it works:**
- Tracks the potential change `max(|f_new - f_old|, |g_new - g_old|)` between checks every `inner_iterations`
  updates, and stops when it falls below `threshold`
- The symmetric backend checks only at the target epsilon and needs two checks there, so under epsilon scaling the
  threshold acts only where the schedule repeats the target epsilon (`eps_list`); the alternating backend checks
  from the start
- Each check costs a max-reduction and a host synchronization
- A threshold that is never met gives a `RuntimeWarning`. The change test is not a marginal-residual certificate.
- `threshold=None` runs all iterations

Some problems stay unconverged at large iteration counts. On clustered 2-D clouds of 400 points at a fixed
`eps=0.09` with `n_iters=50000`, the symmetric backend's `OT(a, b)` solve still had a marginal residual of 2.1e-4 and
the debiased gradient was 1.7% off, against 0.07% for the alternating backend. Raising `n_iters` alone need not fix
this.

#### Numerical Precision

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_tf32` | `bool` | `True` | Allow TF32 for the dot products in the cost. By default this also enables coordinate rounding through `truncate_tf32`. Set `False` for FP32 dot products and, with `truncate_tf32=None`, unrounded centred coordinates. Multiscale rejects `False`. The error of either setting depends on the geometry, `eps`, conditioning and iteration count. |
| `truncate_tf32` | `bool` or `None` | `None` | `None` follows `allow_tf32`. The dense backends round centred FP32/FP64 input coordinates to nearest-even TF32 values before padding; fp16/bf16 inputs receive no additional rounding. Gradients and supported HVPs use an identity derivative through rounding. `False` retains unrounded centred coordinates; `True` rounds even with `allow_tf32=False`. Multiscale always rounds, accepts `None`/`True`, and rejects `False`. |
| `use_exp2` | `bool` | `True` | Compute the log-sum-exp with exp2/log2, as FlashAttention does. |

The symmetric and alternating backends translate both clouds by their joint mass-weighted centroid, in FP32,
before solving; the multiscale backend preprocesses on its own and the low-level solvers do not. The cost is
translation invariant, so nothing changes in exact arithmetic, and a common offset no longer enters the FP32 rounding
of `|x|^2`, which the solvers subtract explicitly. That rounding is relative to `|x|^2`, so clouds spread far from
their centroid can still lose accuracy at small `eps`. With coordinate rounding enabled, the FP32 norms and TF32
dot products use the same rounded cloud. Rounding must follow centring because the subtraction produces new FP32
values. This changes the solved coordinates; FP32 arithmetic error remains.

#### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `half_cost` | `bool` | `False` | Use halved cost C(x,y) = ‖x−y‖²/2 (matches GeomLoss p=2 default). |
| `pad_to_multiple` | `int` or `None` | `None` | Pad point clouds with zero-weight points to the next multiple of this value, so that problems of varying (n, m) share kernel shapes and can trigger fewer compilations and autotuning runs. Must be a positive multiple of 32. |

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

Hessian-vector products (a double backward) are supported with respect to `x` for the raw OT cost, with `y` and the
weights fixed, `debias=False` and no active label cost; the loss may enter a nonlinear function (`loss ** 2`, ...).
A `y` that requires a gradient, `debias=True` or an active label cost raises `NotImplementedError`; second
derivatives involving the weights, and third derivatives, are not supported and not all of them are caught. The
product assumes converged potentials. For balanced problems `hvp_tau2` regularizes the linear solve and biases the
product; relaxed marginals ignore it. If the conjugate gradients do not meet their tolerance on the true residual
within `hvp_max_cg_iter`, a `RuntimeWarning` says so. Meeting it does not certify the product: ill-conditioned
balanced problems (weakly coupled clusters, wide clouds at small `eps`) can have large errors, and lowering
`hvp_tau2` can make them worse.

---

## Examples

### Balanced OT (Default)

```python
from flash_sinkhorn import SamplesLoss
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

At convergence this is the gradient of the OT cost. Before convergence, with the symmetric backend's final
extrapolation, it is the gradient through the stopped final update, as in GeomLoss (without its relaxed-marginal
factor), not the total derivative of the finite run.

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
    use_exp2=True,     # Compute LSE with exp2/log2
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
cost = loss(x, y)  # OTT-JAX's alternating updates
```

**When to use Alternating backend:**
- Benchmarking against OTT-JAX
- Comparing with OTT-JAX (match cost, eps, weights and potential conventions; unconverged iterates need not match)
- Research comparing iteration styles

**When to use symmetric backend (default):**
- Benchmarking against GeomLoss
- Debiased Sinkhorn divergence
- Unbalanced/semi-unbalanced OT
- Epsilon-scaling schedules

### Multiscale Backend (Large 3-D Point Clouds)

```python
loss = SamplesLoss("sinkhorn", blur=0.03, backend="multiscale", tol=5e-3)
cost = loss(a, x, b, y)                 # x: (n, 3), y: (m, 3), CUDA
info = loss.last_multiscale_info        # accepted, stop, detail, envelope, fine_work, B, t, fine_updates, ...
```

The multiscale backend sorts both clouds along a Morton curve, anneals Sinkhorn on
cell centroids, lifts the potentials to the points and finishes with block-sparse
symmetric updates (dense ones when the blur spans most of the cloud). The mask is
rebuilt at every update; in exact arithmetic, omitting a tile removes at most `δ a_i`
from each of its rows and `δ b_j` from each of its columns, with `δ = tol / (4 max(N_B, M_B))`,
so no row or column loses more than `tol / 4` of its mass. The solve returns once a sampled
check (1024 rows and columns, 8192 when inconclusive, mean plus three standard errors; an
empirical stopping rule, not a bound) finds `max(‖P1−a‖₁, ‖Pᵀ1−b‖₁) ≤ tol`; otherwise it
warns and returns its best candidate with `accepted=False`.

- It solves the **centered, TF32-rounded** clouds at `eps = blur**2`, with its own
  annealing schedule. `allow_tf32=False`, `truncate_tf32=False`, `use_epsilon_scaling=False`, `eps`, `eps_list`,
  `n_iters`, `diameter`, `threshold`, `pad_to_multiple`, `reach*` and label costs are
  rejected; `scaling`, `last_extrapolation`, `inner_iterations`, `use_exp2`, the kernel
  launch options and the `hvp_*` options do not apply.
- Balanced OT only; `debias` and `half_cost` are supported, and each of the three debiased
  problems is checked against `tol` separately (the warning names any that is not confirmed).
- Forward only: `potentials=True` returns potentials without autograd (those of OT(a, b),
  not debiased, as with the symmetric backend), and the cost path raises for inputs
  that require gradients.
- The coarse cells are kept at most 9 blurs wide where the cell-count limits allow it, even
  beyond the L2 budget but never beyond twice it: wider cells can lift potentials that leave rows
  so short of mass that the fine masks prune them whole, and the dense coarse updates cost the
  square of the cell count. When no level within twice the budget is narrow enough, the finest
  level within the L2 budget is kept. Clouds smaller than about 8192 points per side are annealed
  on the points themselves; a GPU whose L2 budget holds no cell level is refused.
- The fine stage stops once its sampled score has improved by less than 1% over three consecutive windows of 64
  updates, or after 2048 updates. It stops as unstable when densely repairing a mask's empty block rows of
  positive mass would cost more than one of the update's sparse half-steps, and when two checks in a row give no
  finite score. It stops on its budget before any refinement or update that would take its charged work past 4·10⁸
  point pairs per point, (n + m)/2 points, previews and retries included (`info.fine_work` is the work charged;
  screening and checks are not charged). It stops as too large when a mask or an update would exceed the int32
  indices or the device memory; candidates that do not fit one CSR, or whose refinement runs out of memory, are
  screened and refined in smaller chunks of block rows first. Before stopping as stalled, at the ceiling or on the
  budget, the best candidate is checked once more on the same samples with the same rule, in fp64, and accepted if
  it passes there (`info.detail` then says so, and `info.checks` ends with an entry of kind `"fp64"`); on an A100
  at 2^26 points its first stage took about 15 s and the second, when drawn, costs about eight times more; it is
  not part of `info.fine_work`. Otherwise the solve warns and returns its best candidate with `accepted=False`
  (`info.stop` is `"stalled"`, `"unstable"`, `"ceiling"`, `"budget"` or `"too large"`, and `info.detail` says
  why).
- In sparse updates, empty block rows of zero mass keep their incoming potentials and are never repaired; at the
  end every point of zero weight gets its c-transform against the returned potentials of the other side.
- Limits, measured on A100-SXM4-80GB at tol 1e-2 (looser than the default 5e-3): tests of uniform, clustered (32
  Gaussian clusters), surface and heterogeneous clouds, lognormal weights, LiDAR and N-body data, 2^22 to 2^27
  points, at blurs of 0.35 to 100 median nearest-neighbour spacings.
  `flash_sinkhorn.multiscale.check_limits(x, y, a, b, eps)` reads two warning thresholds from the geometry alone, on
  clouds and weights prepared as for `solve_multiscale` (below), and took seconds in these tests; the solver warns
  above either, then solves anyway.
  - Coarse cells wider than 9 blurs (`envelope.cell_blurs`). Cells that are too wide lose the support of whole block
    rows within the first fine updates, and the width at which that happens depends on the data: uniform clouds kept
    it at 18 blurs, N-body data lost it at 17.8 at a blur where 8.9-blur cells converge (and at 12.8 with a smaller
    blur). Within twice the L2 budget, a cloud filling a cube gets cells of at most 9 blurs up to a side of about
    576 blurs (64 cells per axis), once it has enough points for the per-side cell limit to admit 64³ cells (about
    2^24).
  - ulp32(max |x|²)/eps above 0.05 (`envelope.ulp_over_eps`): no tested problem above it was accepted. Below it the
    sampled check, computed in fp32, can still read high: at 0.019 it read a uniform 2^26 solution at 0.014 whose
    fp64 residual was 0.007 on the same samples; the fp64 re-check above exists for that case.
- Two limits show only while solving, and the solver reports them by its stop:
  - Mask capacity. At 30 median spacings, transposing 1.1e9 to 1.4e9 admitted tiles ran out of memory on 80 GB
    (heterogeneous from 2^24 points, clustered from 2^25, surface at 2^26), and larger masks exceed the int32
    indices (2^31 tiles). In these tests the fine stage stopped as too large within minutes and returned its
    best candidate.
  - Convergence. The clustered clouds stalled at 2^22 points with 3 and 10 median spacings and stopped on the work
    budget at 2^24 with 30 and at 2^26 with 10 (after 4.8 hours on this GPU); a LiDAR pair of 2.8e7 and 8.5e7
    points stalled at 30.
- An unconfirmed solve records its stop in `info.stop` and explains it in `info.detail`; `info.envelope` holds both
  warning numbers.
- `flash_sinkhorn.multiscale.solve_multiscale(x, y, a, b, eps, tol)` is the solver itself.
  It returns `(f, g, info)` for the full cost `||x − y||²` and expects clouds already passed
  through `flash_sinkhorn.multiscale.preprocess_coordinates`; unrounded coordinates are rejected,
  the weights must sum to one within `min(1e-3, tol / 4)` (so must `SamplesLoss` weights with
  `normalize=False`), and each cloud may hold at most 2^27 points.

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
# Stop once the potential change between checks falls below the threshold
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    use_epsilon_scaling=False,
    eps=0.1,
    n_iters=100,              # Max iterations
    threshold=1e-3,           # Stop when potential change < 1e-3
    inner_iterations=10,      # Check every 10 iterations
)

# How many iterations this saves depends on the problem and the threshold
cost = loss(x, y)
```

### Adaptive Padding (Variable-Size OT)

```python
# When solving OT many times with different (n, m), Triton JIT
# recompilation can dominate runtime. Pad to fixed multiples to
# share cached kernels across different problem sizes.
loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    half_cost=True,
    pad_to_multiple=128,  # Pad n and m to next multiple of 128
)

# Sizes that pad to the same (n, m) reuse the compiled kernels
for x_i, y_i, a_i, b_i in variable_size_problems:
    cost = loss(a_i, x_i, b_i, y_i)
```

**How it works:**
- Appends zero-weight phantom points to reach the next multiple
- Mathematically exact: zero-mass points do not affect the transport plan
- Gradients flow correctly through `torch.cat` (autograd handles trimming)
- Potentials (`potentials=True`) are automatically trimmed to original size
- Early stopping convergence checks are masked to unpadded entries

**Choosing the multiple:** a larger multiple gives fewer distinct shapes but pads more points; `None` (the default)
pads nothing.

---

## Low-Level Kernel API

For direct access to the Triton kernels:

### FlashSinkhorn (Shifted Potentials) — Recommended

```python
from flash_sinkhorn.kernels import (
    sinkhorn_flashstyle_symmetric,
    sinkhorn_flashstyle_alternating,
)

# Symmetric solver (matches GeomLoss interface)
f, g = sinkhorn_flashstyle_symmetric(
    x, y, a, b,
    blur=0.05,
    scaling=0.9,
    cost_scale=0.5,  # half_cost for GeomLoss parity
)

# Alternating solver (matches OTT-JAX interface)
f, g = sinkhorn_flashstyle_alternating(
    x, y, a, b,
    eps=0.1,
    n_iters=100,
)
```

#### Shifted Potential Transport Plan Application

```python
from flash_sinkhorn.kernels import apply_plan_vec_flashstyle, apply_plan_mat_flashstyle

# Apply transport plan to a vector: result_i = sum_j P(i,j) * v_j
result = apply_plan_vec_flashstyle(
    x, y, f_shift, g_shift, log_a, log_b, v,
    eps=0.1, axis=1, cost_scale=0.5,
)

# Apply transport plan to a matrix: result_i = sum_j P(i,j) * M_j
result = apply_plan_mat_flashstyle(
    x, y, f_shift, g_shift, log_a, log_b, M,
    eps=0.1, axis=1, cost_scale=0.5,
)
```

#### Potential Conversion

```python
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    standard_to_shifted_potentials,
    shifted_to_standard_potentials,
)

# The conversion helpers take the precomputed (scaled) squared norms, not x/y:
cost_scale = 0.5  # 0.5 for half_cost (GeomLoss parity), else 1.0
alpha = cost_scale * (x ** 2).sum(dim=1)  # [n]
beta = cost_scale * (y ** 2).sum(dim=1)   # [m]

# Convert GeomLoss potentials to shifted (for flashstyle apply kernels)
f_shift, g_shift = standard_to_shifted_potentials(f, g, alpha, beta)

# Convert shifted back to GeomLoss standard
f, g = shifted_to_standard_potentials(f_shift, g_shift, alpha, beta)
```

### GeomLoss-Style (Symmetric Updates) — Legacy

```python
from flash_sinkhorn.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)

f, g = sinkhorn_geomloss_online_potentials_sqeuclid(
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
from flash_sinkhorn.kernels.sinkhorn_triton_ott_sqeuclid import sinkhorn_potentials_sqeuclid

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

### C-Transform / Semi-Dual OT (Non-Entropic)

The hard (non-entropic) Kantorovich c-transform and its differentiable semi-dual
objective — building blocks for the Wasserstein Patch Prior (WPP) and other semi-dual
OT methods. The `n×m` cost matrix is never materialized (O(nd) memory), and gradients
are analytic via Danskin's theorem (no entropic smoothing).

```python
import torch
from flash_sinkhorn import c_transform_fwd, c_transform_cost

x   = torch.randn(4096, 64, device="cuda")   # source points  [n, d]
y   = torch.randn(2048, 64, device="cuda")   # target points  [m, d]
psi = torch.randn(2048,     device="cuda")   # dual potential [m]

# --- (1) c_transform_fwd: raw values + argmin (non-differentiable) ---
#   c_i   = min_j    [ cost_scale * ||x_i - y_j||² - psi_j ]
#   idx_i = argmin_j [ cost_scale * ||x_i - y_j||² - psi_j ]
c_values, argmin_idx = c_transform_fwd(
    x, y, psi,
    cost_scale=1.0,     # 1.0 -> ||x-y||²,  0.5 -> ||x-y||²/2 (GeomLoss convention)
    allow_tf32=True,
    autotune=True,
)
# c_values: [n] float32   argmin_idx: [n] int64

# --- (2) c_transform_cost: differentiable semi-dual objective ---
#   L(x, psi) = Σ_i a_i * c^psi(x_i) + Σ_j b_j * psi_j
x   = x.requires_grad_(True)
psi = psi.requires_grad_(True)
loss = c_transform_cost(
    x, y, psi,
    cost_scale=1.0,
    a=None,             # source weights [n], default uniform 1/n
    b=None,             # target weights [m], default uniform 1/m
)
grad_x, grad_psi = torch.autograd.grad(loss, [x, psi])
#   grad_x_i   = a_i * 2 * cost_scale * (x_i - y_{j*(i)})
#   grad_psi_j = b_j - Σ_{i: j*(i)=j} a_i
```

**Notes**
- Differentiable w.r.t. `x` and `psi` only; `y` must not require grad (raises `NotImplementedError`).
- Weights `a`, `b` must not require grad.
- Tie-breaking: smallest-`j` wins across tiles.
- Precision: with `allow_tf32=True` (the default) the coordinates are rounded to TF32 before
  both the dot products and the squared norms are computed, so the two describe the same points.
  FP32 arithmetic error remains and can change the cell of a point whose two best sites are nearly
  tied. Pass `allow_tf32=False` to keep the coordinates as given. The x gradient uses the original
  coordinates at the selected cells.
- Raw kernel: `from flash_sinkhorn.kernels import c_transform_kernel` computes the inner
  `min_j[-2*cost_scale*⟨x_i, y_j⟩ + bias_j]` over a precomputed `bias = cost_scale*||y||² - psi`;
  `c_transform_fwd` wraps it, rounds the coordinates to TF32 when TF32 is on, and adds back
  `cost_scale*||x_i||²`. Called directly, the kernel uses the bias as given and does not round the
  coordinates first; TF32 still applies to its dot products.

---

## Comparison with GeomLoss

| Feature | FlashSinkhorn | GeomLoss |
|---------|-----------|----------|
| Cost function | Squared Euclidean only | Multiple (Euclidean, Laplacian, etc.) |
| Backend | Triton (symmetric, alternating, multiscale) | PyTorch / KeOps (tensorized, online, multiscale) |
| Unbalanced OT | Yes (`reach`, `reach_x`, `reach_y`) | Yes (`reach`) |
| Semi-unbalanced OT | Yes (`reach_x` ≠ `reach_y`) | No |
| Debiased Sinkhorn | Yes (`debias=True`) | Yes |
| Early stopping | Yes (`threshold`) | No (epsilon-scaling only) |
| Adaptive padding | Yes (`pad_to_multiple`, variable-size OT) | No |
| Gradient | Analytic (no backprop; not for multiscale) | Analytic (no backprop) |
| HVP | Yes (CG solver; not for multiscale) | No |
| Memory | O(nd) streaming (multiscale also stores its block masks) | O(n + m) symmetric or O(nm) tensorized |

### Migration from GeomLoss

```python
# GeomLoss
from geomloss import SamplesLoss as GeomLossSamplesLoss
loss_geo = GeomLossSamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False)

# FlashSinkhorn with the same settings
from flash_sinkhorn import SamplesLoss
loss_tri = SamplesLoss("sinkhorn", blur=0.1, scaling=0.5, debias=False, half_cost=True, allow_tf32=False)

# Same cost convention (GeomLoss p=2 is |x-y|^2/2); values agree to numerical tolerance
cost_geo = loss_geo(x, y)
cost_tri = loss_tri(x, y)
```

---

## Notes

### Cost and Epsilon Convention

FlashSinkhorn uses **full squared Euclidean cost**: `C(x,y) = ||x - y||²`

The epsilon schedule uses `eps = blur^p`, as in GeomLoss. To compare raw OT potentials, set `potentials=True` and `debias=False` in both libraries and match the cost, weights, schedule and precision:

```python
# For potential parity with FlashSinkhorn, use full squared cost
def full_sqdist_cost(x, y):
    return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)

loss_geo = GeomLossSamplesLoss("sinkhorn", cost=full_sqdist_cost, ...)
loss_tri = SamplesLoss("sinkhorn", ...)

# Raw potentials agree to numerical tolerance (balanced ones up to an additive constant)
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

**Shifted convention (FlashSinkhorn):**
```
f_shift = f - cost_scale * ||x||²
g_shift = g - cost_scale * ||y||²
```
The shifted formulation factors out the quadratic norm from the LSE, reducing per-tile operations. The `apply_plan_*_flashstyle` kernels expect shifted potentials.

Convert between conventions:
```python
from flash_sinkhorn.hvp import geomloss_to_ott_potentials
f_hat, g_hat = geomloss_to_ott_potentials(f, g, a, b, eps=0.1)

from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import standard_to_shifted_potentials, shifted_to_standard_potentials
alpha = 0.5 * (x ** 2).sum(dim=1)  # cost_scale * ||x||²  (cost_scale=0.5 here for half_cost)
beta = 0.5 * (y ** 2).sum(dim=1)   # cost_scale * ||y||²
f_shift, g_shift = standard_to_shifted_potentials(f, g, alpha, beta)
f, g = shifted_to_standard_potentials(f_shift, g_shift, alpha, beta)
```

### Numerical Stability

- Set `allow_tf32=False` for strict FP32 when comparing against CPU references
- For very small `blur`, use a slower schedule (`scaling` closer to 1) followed by more iterations at the target epsilon (`eps_list`), and check convergence; `n_iters` truncates an annealing schedule and never extends it
