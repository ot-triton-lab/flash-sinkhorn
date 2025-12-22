# Gradient and HVP Guide

This note summarizes how to compute **gradients** and **Hessian‑vector products (HVP)** with `ot_triton` (squared Euclidean cost, CUDA only).

## Scope and constraints

- Cost: full `||x - y||^2`.
- CUDA required (Triton kernels).
- HVP is implemented **w.r.t. x only**.
- Multiscale HVP only supports `d in {1,2,3}`.
- Double backward w.r.t. `y` is not implemented.

## 1) First‑order gradient (recommended path)

Use the `SamplesLoss` API (analytic gradients, no backprop through iterations):

```python
import torch
from ot_triton import SamplesLoss

x = torch.randn(4096, 64, device="cuda", dtype=torch.float32, requires_grad=True)
y = torch.randn(4096, 64, device="cuda", dtype=torch.float32)

loss = SamplesLoss(
    "sinkhorn",
    blur=0.1,
    scaling=0.5,
    debias=False,
    normalize=False,
    use_epsilon_scaling=False,
    eps=0.1,
    n_iters=16,
)

val = loss(x, y)
grad_x = torch.autograd.grad(val, x)[0]
```

Notes:
- `grad_x` uses the **analytic gradient** kernels.
- If you want gradients w.r.t. `a,b`, set `requires_grad=True` on them and pass `(a, x, b, y)` to `loss`.

## 2) Second‑order HVP (autograd double backward)

You can get HVP by differentiating a dot‑product against a vector:

```python
v = torch.randn_like(x)
val = loss(x, y)
grad_x = torch.autograd.grad(val, x, create_graph=True)[0]
hvp_x = torch.autograd.grad((grad_x * v).sum(), x)[0]
```

Internally this calls the **OTT‑Hessian‑style CG solve** on the linear system for the potentials (no plan or cost materialization).

### Preconditioners (CG)

The HVP CG solver supports:
- `preconditioner="none"` (default, often fastest wall‑clock)
- `preconditioner="jacobi"` (diagonal on unscaled system)
- `preconditioner="neumann"` with `precond_terms=k` (Neumann series depth)

To control these through `SamplesLoss`:

```python
loss = SamplesLoss(
    ...,
    hvp_preconditioner="neumann",
    hvp_precond_terms=3,
    hvp_use_preconditioner=True,
)
```

## 3) Direct HVP from potentials (manual path)

If you already have potentials, call the HVP primitive directly:

```python
from ot_triton.hvp import (
    geomloss_to_ott_potentials,
    hvp_x_sqeuclid_from_potentials,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)

# 1) Solve for prelast GeomLoss‑style potentials
_, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
    x, y, a, b,
    use_epsilon_scaling=False,
    last_extrapolation=True,
    eps_list=[0.1] * 8,
)

# 2) Convert to OTT convention
f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=0.1)

# 3) HVP with vector A (same shape as x)
A = torch.randn_like(x)
hvp_x, info = hvp_x_sqeuclid_from_potentials(
    x, y, f_hat, g_hat, A, eps=0.1,
    preconditioner="none",
)
```

## 4) Multiscale HVP (D=1/2/3 only)

```python
from ot_triton.hvp import hvp_x_sqeuclid_multiscale

hvp_x, info = hvp_x_sqeuclid_multiscale(
    x, y, a, b, A,
    eps_list=[0.1] * 8,
    truncate=5.0,
    max_coarse_levels=1,
)
```

## Tips

- For strict FP32: set `allow_tf32=False`.
- For stability with small `eps`, prefer `use_exp2=True`.
- If CG stalls, try `hvp_cg_stabilise_every=10` (recompute true residual).
