from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from ot_triton.hvp import geomloss_to_ott_potentials, ott_to_geomloss_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    epsilon_schedule,
    log_weights,
    max_diameter,
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_ott_sqeuclid import (
    sinkhorn_potentials_sqeuclid as sinkhorn_ott_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)


class _SinkhornGradFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        f_grad: torch.Tensor,
        g_grad: torch.Tensor,
        eps: float,
        allow_tf32: bool,
        use_exp2: bool,
        autotune: bool,
        block_m: Optional[int],
        block_n: Optional[int],
        block_k: Optional[int],
        num_warps: Optional[int],
        num_stages: int,
        grad_scale: torch.Tensor,
        hvp_tau2: float,
        hvp_max_cg_iter: int,
        hvp_cg_rtol: float,
        hvp_cg_atol: float,
        hvp_cg_stabilise_every: int,
        hvp_preconditioner: str,
        hvp_precond_terms: int,
        hvp_use_preconditioner: bool,
        compute_grad_x: bool,
        compute_grad_y: bool,
        rho_x: Optional[float] = None,  # For semi-unbalanced OT HVP
        rho_y: Optional[float] = None,  # For semi-unbalanced OT HVP
        cost_scale: float = 1.0,  # Cost scaling: 1.0 for full, 0.5 for half
        # OTDD label-augmented cost parameters
        label_x: Optional[torch.Tensor] = None,
        label_y: Optional[torch.Tensor] = None,
        label_cost_matrix: Optional[torch.Tensor] = None,
        lambda_x: float = 1.0,
        lambda_y: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gx, gy = sinkhorn_geomloss_online_grad_sqeuclid(
            x,
            y,
            a,
            b,
            f_grad,
            g_grad,
            eps=float(eps),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=int(num_stages),
            autotune=bool(autotune),
            grad_scale=grad_scale,
            compute_grad_x=bool(compute_grad_x),
            compute_grad_y=bool(compute_grad_y),
            cost_scale=float(cost_scale),
            # OTDD label-augmented cost
            label_x=label_x,
            label_y=label_y,
            label_cost_matrix=label_cost_matrix,
            lambda_x=float(lambda_x),
            lambda_y=float(lambda_y),
        )

        ctx.save_for_backward(x, y, a, b, f_grad, g_grad, grad_scale)
        ctx.eps = float(eps)
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.use_exp2 = bool(use_exp2)
        ctx.autotune = bool(autotune)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.block_k = block_k
        ctx.num_warps = num_warps
        ctx.num_stages = int(num_stages)
        ctx.hvp_tau2 = float(hvp_tau2)
        ctx.hvp_max_cg_iter = int(hvp_max_cg_iter)
        ctx.hvp_cg_rtol = float(hvp_cg_rtol)
        ctx.hvp_cg_atol = float(hvp_cg_atol)
        ctx.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        ctx.hvp_preconditioner = str(hvp_preconditioner)
        ctx.hvp_precond_terms = int(hvp_precond_terms)
        ctx.hvp_use_preconditioner = bool(hvp_use_preconditioner)
        ctx.rho_x = rho_x  # For semi-unbalanced OT HVP
        ctx.rho_y = rho_y  # For semi-unbalanced OT HVP
        ctx.cost_scale = float(cost_scale)  # For HVP
        # OTDD label cost (HVP with labels not yet supported)
        ctx.use_label_cost = (
            label_x is not None and label_y is not None
            and label_cost_matrix is not None and lambda_y != 0.0
        )
        return gx, gy

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_x: Optional[torch.Tensor], grad_grad_y: Optional[torch.Tensor]
    ):
        x, y, a, b, f_grad, g_grad, grad_scale = ctx.saved_tensors

        if grad_grad_y is not None:
            raise NotImplementedError("Double backward w.r.t y is not implemented yet.")

        if ctx.use_label_cost:
            raise NotImplementedError(
                "Double backward (HVP) is not supported with OTDD label-augmented cost. "
                "Use gradient flow without HVP."
            )

        out_x = None
        if grad_grad_x is not None:
            if grad_grad_x.shape != x.shape:
                raise ValueError("grad_grad_x must have the same shape as x.")
            if not grad_grad_x.is_cuda:
                raise ValueError("grad_grad_x must be a CUDA tensor.")

            # Use no_grad to prevent autograd from tracking tensor operations
            # during backward. Triton autotune mutates tensor metadata (version
            # counter), which triggers unnecessary autograd bookkeeping without
            # this guard.
            # Follows FlashAttention's pattern (flash_attn_func.py line 1040).
            with torch.no_grad():
                f_hat, g_hat = geomloss_to_ott_potentials(
                    f_grad, g_grad, a, b, eps=ctx.eps
                )
                hvp_x, _ = hvp_x_sqeuclid_from_potentials(
                    x,
                    y,
                    f_hat,
                    g_hat,
                    grad_grad_x,
                    eps=ctx.eps,
                    rho_x=ctx.rho_x,  # For semi-unbalanced OT HVP
                    rho_y=ctx.rho_y,  # For semi-unbalanced OT HVP
                    cost_scale=ctx.cost_scale,  # Cost scaling for half cost
                    tau2=ctx.hvp_tau2,
                    max_cg_iter=ctx.hvp_max_cg_iter,
                    cg_rtol=ctx.hvp_cg_rtol,
                    cg_atol=ctx.hvp_cg_atol,
                    cg_stabilise_every=ctx.hvp_cg_stabilise_every,
                    preconditioner=ctx.hvp_preconditioner,
                    precond_terms=ctx.hvp_precond_terms,
                    use_preconditioner=ctx.hvp_use_preconditioner,
                    allow_tf32=False,  # HVP requires full fp32 precision, TF32 causes numerical instability
                    use_exp2=ctx.use_exp2,
                    block_m=ctx.block_m,
                    block_n=ctx.block_n,
                    block_k=ctx.block_k,
                    num_warps=int(ctx.num_warps or 4),
                    num_stages=ctx.num_stages,
                )
                out_x = hvp_x * grad_scale

        # Inputs: x,y,a,b,f_grad,g_grad,eps,allow_tf32,use_exp2,autotune,
        # block_m,block_n,block_k,num_warps,num_stages,grad_scale,
        # hvp_tau2,hvp_max_cg_iter,hvp_cg_rtol,hvp_cg_atol,hvp_cg_stabilise_every,hvp_preconditioner,hvp_precond_terms,hvp_use_preconditioner,
        # compute_grad_x, compute_grad_y, rho_x, rho_y, cost_scale,
        # label_x, label_y, label_cost_matrix, lambda_x, lambda_y
        return (
            out_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # rho_x
            None,  # rho_y
            None,  # cost_scale
            None,  # label_x
            None,  # label_y
            None,  # label_cost_matrix
            None,  # lambda_x
            None,  # lambda_y
        )


@dataclass(frozen=True)
class _ParsedInputs:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    batched: bool
    a_view_shape: Tuple[int, ...]
    b_view_shape: Tuple[int, ...]


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("Expected a torch.Tensor.")
    if not x.is_floating_point():
        raise TypeError("Expected a floating-point tensor.")
    return x


def _normalize_weights(w: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    """Normalize weights to sum to 1 along the last dimension.

    Args:
        w: Input weights tensor
        eps: Small value to add to weights before normalization (for stability)

    Returns:
        Normalized weights (sum to 1 along last dimension)

    Note:
        Uses a small clamp (1e-40) on the sum to avoid division by zero
        when all weights are zero. This is a defensive guard for edge cases.
    """
    w = w.float()
    if eps > 0:
        w = w + eps
    z = w.sum(dim=-1, keepdim=True)
    # Clamp to avoid division by zero when all weights are zero
    z = z.clamp(min=1e-40)
    return w / z


def _process_args(*args, normalize: bool) -> _ParsedInputs:
    if len(args) == 2:
        x, y = args
        a = None
        b = None
    elif len(args) == 4:
        a, x, b, y = args
    else:
        raise TypeError(
            "SamplesLoss expects either (x, y) or (a, x, b, y). "
            f"Got {len(args)} arguments."
        )

    x = _as_float_tensor(x)
    y = _as_float_tensor(y)

    if x.ndim == 2:
        batched = False
        n, dx = x.shape
        m, dy = y.shape
        if dx != dy:
            raise ValueError("x and y must have the same feature dimension.")

        if a is None:
            a = torch.full((n,), 1.0 / n, device=x.device, dtype=torch.float32)
            a_view_shape = (n,)
        else:
            a = _as_float_tensor(a)
            if a.ndim == 2 and a.shape == (n, 1):
                a = a[:, 0]
                a_view_shape = (n, 1)
            elif a.ndim == 1 and a.shape == (n,):
                a_view_shape = (n,)
            else:
                raise ValueError("a must have shape (n,) or (n,1) matching x.")

        if b is None:
            b = torch.full((m,), 1.0 / m, device=y.device, dtype=torch.float32)
            b_view_shape = (m,)
        else:
            b = _as_float_tensor(b)
            if b.ndim == 2 and b.shape == (m, 1):
                b = b[:, 0]
                b_view_shape = (m, 1)
            elif b.ndim == 1 and b.shape == (m,):
                b_view_shape = (m,)
            else:
                raise ValueError("b must have shape (m,) or (m,1) matching y.")

        if normalize:
            a = _normalize_weights(a)
            b = _normalize_weights(b)

        return _ParsedInputs(
            x=x,
            y=y,
            a=a,
            b=b,
            batched=batched,
            a_view_shape=a_view_shape,
            b_view_shape=b_view_shape,
        )

    if x.ndim == 3:
        batched = True
        if y.ndim != 3:
            raise ValueError("If x is batched (B,N,D), y must be batched too.")
        bsz, n, dx = x.shape
        bsz2, m, dy = y.shape
        if bsz != bsz2:
            raise ValueError("x and y must have the same batch size.")
        if dx != dy:
            raise ValueError("x and y must have the same feature dimension.")

        if a is None:
            a = torch.full((bsz, n), 1.0 / n, device=x.device, dtype=torch.float32)
            a_view_shape = (bsz, n)
        else:
            a = _as_float_tensor(a)
            if a.ndim == 3 and a.shape == (bsz, n, 1):
                a = a[:, :, 0]
                a_view_shape = (bsz, n, 1)
            elif a.ndim == 2 and a.shape == (bsz, n):
                a_view_shape = (bsz, n)
            else:
                raise ValueError(
                    "a must have shape (B,n) or (B,n,1) matching x."
                )

        if b is None:
            b = torch.full((bsz, m), 1.0 / m, device=y.device, dtype=torch.float32)
            b_view_shape = (bsz, m)
        else:
            b = _as_float_tensor(b)
            if b.ndim == 3 and b.shape == (bsz, m, 1):
                b = b[:, :, 0]
                b_view_shape = (bsz, m, 1)
            elif b.ndim == 2 and b.shape == (bsz, m):
                b_view_shape = (bsz, m)
            else:
                raise ValueError(
                    "b must have shape (B,m) or (B,m,1) matching y."
                )

        if normalize:
            a = _normalize_weights(a)
            b = _normalize_weights(b)

        return _ParsedInputs(
            x=x,
            y=y,
            a=a,
            b=b,
            batched=batched,
            a_view_shape=a_view_shape,
            b_view_shape=b_view_shape,
        )

    raise ValueError("x and y must be shaped (N,D) or (B,N,D).")


class SamplesLoss(torch.nn.Module):
    """GeomLoss-like API for (online) Sinkhorn OT using Triton.

    This is a minimal, CUDA-only subset of GeomLoss's `SamplesLoss`:
    - `loss` must be "sinkhorn".
    - Only squared Euclidean ground cost is supported (with optional label cost).
    - Supports balanced, unbalanced, and semi-unbalanced OT.

    Two backends are available:
    - `backend="symmetric"` (default): GeomLoss-style symmetric Sinkhorn loop (Jacobi).
      Supports all features: debiasing, unbalanced OT, epsilon scaling, label cost.
    - `backend="alternating"`: OTT-JAX-style alternating Sinkhorn loop (Gauss-Seidel).
      Matches OTT-JAX's `update_potential` exactly. Requires fixed eps and n_iters.
      Supports: debiasing, unbalanced/semi-unbalanced OT.
      Does NOT support: epsilon scaling or label cost.

    The implementation returns either:
    - a scalar OT cost (default), or
    - a pair of potentials (f, g) when `potentials=True`.

    Unbalanced and Semi-Unbalanced OT
    ---------------------------------
    Control marginal relaxation via `reach`, `reach_x`, and `reach_y` parameters.
    The marginal penalty strength is rho = reach^2 (for squared Euclidean cost, p=2).

    - reach=None (or reach_x=reach_y=None): Balanced OT (strict marginal constraints)
    - reach>0: Unbalanced OT with equal relaxation on both marginals
    - reach_x>0, reach_y=None: Semi-unbalanced OT (relax source, strict target)
    - reach_x=None, reach_y>0: Semi-unbalanced OT (strict source, relax target)
    - reach_x>0, reach_y>0: Fully asymmetric unbalanced OT

    Semi-unbalanced OT is useful when one distribution is trusted (e.g., a fixed
    reference) while the other may have mass differences or outliers.

    OTDD Label-Augmented Cost
    -------------------------
    For OTDD-style dataset distance computation, supports augmented cost:

        C[i,j] = lambda_x * ||x_i - y_j||² + lambda_y * W[label_i, label_j]

    Where W is a precomputed (V × V) label-to-label distance matrix.

    Example:
        loss = SamplesLoss(
            loss='sinkhorn', blur=0.316, half_cost=True,
            label_cost_matrix=W,  # (V, V) label distances
            lambda_x=1.0,         # Feature weight
            lambda_y=1.0,         # Label weight
        )
        dist = loss(x, y, label_x=labels_x, label_y=labels_y)

    Note: Gradients w.r.t. x and y are supported (for gradient flows).
    Double backward (HVP) is not yet supported with label cost.

    FlashSinkhorn (Shifted Potentials)
    ----------------------------------
    Set `use_flashstyle=True` to use the new FlashSinkhorn kernels, which offer:
    - 67% fewer memory loads per iteration (via shifted potential formulation)
    - 7-34% speedup at n >= 10,000, regardless of iteration count
    - Break-even at n=5,000 with ~100 iterations; overhead at fewer iterations
    - Recommended for large-scale OT (n >= 10,000)

    FlashSinkhorn currently does NOT support:
    - Debiased Sinkhorn divergence (debias=True)
    - Unbalanced OT (reach parameter)
    - OTDD label-augmented cost
    - last_extrapolation=True (always uses final potentials)

    Use `use_flashstyle=False` (default) for these features.

    Notes
    -----
    - Gradients are computed analytically (no backprop through Sinkhorn iterations),
      matching GeomLoss's `last_extrapolation` convention.
    - `potentials=True` returns (f, g) without autograd support.
    """

    def __init__(
        self,
        loss: str = "sinkhorn",
        *,
        p: int = 2,
        blur: float = 0.05,
        reach: Optional[float] = None,
        reach_x: Optional[float] = None,
        reach_y: Optional[float] = None,
        scaling: float = 0.5,
        debias: bool = False,
        potentials: bool = False,
        backend: str = "symmetric",
        normalize: bool = True,
        use_epsilon_scaling: bool = True,
        last_extrapolation: bool = True,
        allow_tf32: bool = True,
        use_exp2: bool = True,
        autotune: bool = True,
        half_cost: bool = False,
        eps: Optional[float] = None,
        n_iters: Optional[int] = None,
        diameter: Optional[float] = None,
        eps_list: Optional[Sequence[float]] = None,
        block_m: Optional[int] = None,
        block_n: Optional[int] = None,
        block_k: Optional[int] = None,
        num_warps: Optional[int] = None,
        num_stages: int = 2,
        hvp_tau2: float = 1e-5,
        hvp_max_cg_iter: int = 300,
        hvp_cg_rtol: float = 1e-6,
        hvp_cg_atol: float = 1e-6,
        hvp_cg_stabilise_every: int = 0,
        hvp_preconditioner: str = "none",
        hvp_precond_terms: int = 3,
        hvp_use_preconditioner: bool = True,
        # OTDD label-augmented cost parameters
        label_cost_matrix: Optional[torch.Tensor] = None,
        lambda_x: float = 1.0,
        lambda_y: float = 0.0,
        # Early stopping parameters (like OTT-JAX)
        threshold: Optional[float] = None,  # Convergence threshold (None = no early stopping)
        inner_iterations: int = 10,  # Check convergence every N iterations (like OTT-JAX)
        # FlashSinkhorn: use new shifted-potential kernels (67% fewer loads)
        use_flashstyle: bool = True,  # Default True: uses FlashSinkhorn kernels (faster)
    ):
        super().__init__()

        if loss != "sinkhorn":
            raise ValueError('Only loss="sinkhorn" is supported.')
        if p != 2:
            raise ValueError("Only p=2 (squared Euclidean cost) is supported.")
        # Debiasing is now supported
        if backend not in ("symmetric", "alternating", "triton", "auto"):
            raise ValueError(
                'Only backend in {"symmetric","alternating","triton","auto"} is supported.'
            )
        # OTT backend requires specific settings
        if backend == "alternating":
            if use_epsilon_scaling:
                raise ValueError(
                    'backend="alternating" requires use_epsilon_scaling=False. '
                    'Provide fixed eps and n_iters instead.'
                )
            if eps is None or n_iters is None:
                raise ValueError(
                    'backend="alternating" requires eps and n_iters to be specified.'
                )
            # NOTE: debias and unbalanced OT are now supported for alternating backend
            if label_cost_matrix is not None:
                raise ValueError(
                    'backend="alternating" does not support OTDD label cost. '
                    'Use backend="symmetric" for label-augmented cost.'
                )
        # Validate reach parameters
        if reach is not None and reach <= 0:
            raise ValueError("reach must be positive (or None for balanced OT).")
        if reach_x is not None and reach_x <= 0:
            raise ValueError("reach_x must be positive (or None for balanced source).")
        if reach_y is not None and reach_y <= 0:
            raise ValueError("reach_y must be positive (or None for balanced target).")

        # Handle reach → reach_x/reach_y conversion (legacy API compatibility)
        if reach is not None:
            if reach_x is None:
                reach_x = reach
            if reach_y is None:
                reach_y = reach

        # Check unbalanced compatibility
        is_unbalanced = reach_x is not None or reach_y is not None

        self.loss = loss
        self.p = p
        self.blur = float(blur)
        # Support semi-unbalanced OT with separate reach_x/reach_y
        self.reach_x = None if reach_x is None else float(reach_x)
        self.reach_y = None if reach_y is None else float(reach_y)
        self.rho_x = None if reach_x is None else float(reach_x) ** 2  # rho_x = reach_x^p
        self.rho_y = None if reach_y is None else float(reach_y) ** 2  # rho_y = reach_y^p
        # Legacy: self.reach/self.rho for backward compatibility (only if symmetric)
        if reach_x == reach_y:
            self.reach = self.reach_x
            self.rho = self.rho_x
        else:
            self.reach = None  # Semi-unbalanced: no single reach
            self.rho = None
        self.scaling = float(scaling)
        self.debias = bool(debias)
        self.potentials = bool(potentials)
        self.backend = backend
        self.normalize = bool(normalize)

        self.use_epsilon_scaling = bool(use_epsilon_scaling)
        self.last_extrapolation = bool(last_extrapolation)
        self.allow_tf32 = bool(allow_tf32)
        self.use_exp2 = bool(use_exp2)
        self.autotune = bool(autotune)
        self.half_cost = bool(half_cost)
        self.cost_scale = 0.5 if half_cost else 1.0

        self.eps = None if eps is None else float(eps)
        self.n_iters = None if n_iters is None else int(n_iters)
        self.diameter = None if diameter is None else float(diameter)
        self.eps_list = None if eps_list is None else list(map(float, eps_list))

        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = int(num_stages)

        self.hvp_tau2 = float(hvp_tau2)
        self.hvp_max_cg_iter = int(hvp_max_cg_iter)
        self.hvp_cg_rtol = float(hvp_cg_rtol)
        self.hvp_cg_atol = float(hvp_cg_atol)
        self.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        self.hvp_preconditioner = str(hvp_preconditioner)
        self.hvp_precond_terms = int(hvp_precond_terms)
        self.hvp_use_preconditioner = bool(hvp_use_preconditioner)

        # OTDD label-augmented cost: C[i,j] = lambda_x * ||x_i - y_j||² + lambda_y * W[label_i, label_j]
        self.label_cost_matrix = label_cost_matrix  # W: (V, V) label distance matrix
        self.lambda_x = float(lambda_x)  # Weight for Euclidean cost
        self.lambda_y = float(lambda_y)  # Weight for label cost

        # Early stopping (like OTT-JAX)
        self.threshold = None if threshold is None else float(threshold)
        self.inner_iterations = int(inner_iterations)

        # FlashSinkhorn: use new shifted-potential kernels
        self.use_flashstyle = bool(use_flashstyle)

    def _eps_list_for_inputs(self, x: torch.Tensor, y: torch.Tensor) -> Sequence[float]:
        if self.eps_list is not None:
            eps_list = list(self.eps_list)
        elif self.use_epsilon_scaling:
            diameter = self.diameter
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = list(epsilon_schedule(diameter, self.blur, self.scaling, p=2.0))
        else:
            if self.eps is None or self.n_iters is None:
                raise ValueError(
                    "When use_epsilon_scaling=False, provide eps and n_iters."
                )
            eps_list = [float(self.eps)] * int(self.n_iters)

        if self.n_iters is not None:
            eps_list = eps_list[: int(self.n_iters)]
        if len(eps_list) == 0:
            raise ValueError("eps_list is empty after applying n_iters.")
        return eps_list

    def forward(
        self,
        *args: Union[torch.Tensor, float],
        label_x: Optional[torch.Tensor] = None,
        label_y: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        parsed = _process_args(*args, normalize=self.normalize)

        # Store label tensors for use in nested function
        _label_x = label_x
        _label_y = label_y

        if not parsed.x.is_cuda or not parsed.y.is_cuda:
            raise ValueError("ot_triton.SamplesLoss requires CUDA tensors.")

        class _SinkhornCostFn(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                x: torch.Tensor,
                y: torch.Tensor,
                a: torch.Tensor,
                b: torch.Tensor,
                eps_list: Tuple[float, ...],
                last_extrapolation: bool,
                allow_tf32: bool,
                use_exp2: bool,
                autotune: bool,
                block_m: Optional[int],
                block_n: Optional[int],
                block_k: Optional[int],
                num_warps: Optional[int],
                num_stages: int,
            ) -> torch.Tensor:
                # OTT backend: use alternating-update Sinkhorn (matches OTT-JAX)
                if self.backend == "alternating":
                    eps = float(eps_list[-1])  # Fixed eps for OTT backend
                    n_iters = len(eps_list)

                    if self.use_flashstyle:
                        # NEW: FlashSinkhorn alternating (shifted potentials, 67% fewer loads)
                        # Note: FlashSinkhorn uses internal autotuning, doesn't expose block sizes
                        # Supports semi-unbalanced OT via reach_x/reach_y
                        f_cost, g_cost = sinkhorn_flashstyle_alternating(
                            x,
                            y,
                            a,
                            b,
                            eps=eps,
                            n_iters=n_iters,
                            cost_scale=self.cost_scale,
                            reach_x=self.reach_x,
                            reach_y=self.reach_y,
                            autotune=autotune,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                        )
                        # FlashSinkhorn returns GeomLoss-convention potentials directly
                        f_grad, g_grad = f_cost, g_cost
                    else:
                        # OLD: OTT-style alternating Sinkhorn
                        loga = log_weights(a).contiguous()
                        logb = log_weights(b).contiguous()
                        f_cost, g_cost = sinkhorn_ott_potentials_sqeuclid(
                            x,
                            y,
                            loga,
                            logb,
                            eps,
                            n_iters,
                            autotune=autotune,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                        # CRITICAL: Convert OTT potentials to GeomLoss convention for gradient kernel.
                        # OTT potentials: f_ott = eps*log(a) - LSE, g_ott = eps*log(b) - LSE
                        # GeomLoss gradient kernel expects: f_geo, g_geo (without marginal weights baked in)
                        # The kernel computes: exp((g - C)/eps) * b, so adding logb again would double-count.
                        f_grad, g_grad = ott_to_geomloss_potentials(f_cost, g_cost, a, b, eps=eps)

                    ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
                    ctx.eps = eps
                    ctx.rho_x = self.rho_x  # Semi-unbalanced OT now supported
                    ctx.rho_y = self.rho_y
                    ctx.allow_tf32 = bool(allow_tf32)
                    ctx.use_exp2 = bool(use_exp2)
                    ctx.autotune = bool(autotune)
                    ctx.block_m = block_m
                    ctx.block_n = block_n
                    ctx.block_k = block_k
                    ctx.num_warps = num_warps
                    ctx.num_stages = int(num_stages)
                    ctx.label_x = None
                    ctx.label_y = None
                    ctx.label_cost_matrix_stored = None
                    ctx.lambda_x = 1.0
                    ctx.lambda_y = 0.0

                    # Balanced OT cost: <a, f> + <b, g>
                    return (a * f_cost).sum() + (b * g_cost).sum()

                # GeomLoss backend (default): symmetric-update Sinkhorn
                if self.use_flashstyle:
                    # FlashSinkhorn symmetric (shifted potentials, 67% fewer loads)
                    # Now supports: last_extrapolation, OTDD label cost, unbalanced OT
                    if last_extrapolation:
                        f_cost, g_cost, f_grad, g_grad = sinkhorn_flashstyle_symmetric(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=self.use_epsilon_scaling,
                            eps=self.eps,
                            n_iters=self.n_iters,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            autotune=autotune,
                            cost_scale=self.cost_scale,
                            rho_x=self.rho_x,
                            rho_y=self.rho_y,
                            last_extrapolation=True,
                            return_prelast=True,
                            # OTDD label-augmented cost
                            label_x=_label_x,
                            label_y=_label_y,
                            label_cost_matrix=self.label_cost_matrix,
                            lambda_x=self.lambda_x,
                            lambda_y=self.lambda_y,
                            # Early stopping
                            threshold=self.threshold,
                            check_every=self.inner_iterations,
                        )
                    else:
                        f_cost, g_cost = sinkhorn_flashstyle_symmetric(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=self.use_epsilon_scaling,
                            eps=self.eps,
                            n_iters=self.n_iters,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            autotune=autotune,
                            cost_scale=self.cost_scale,
                            rho_x=self.rho_x,
                            rho_y=self.rho_y,
                            last_extrapolation=False,
                            # OTDD label-augmented cost
                            label_x=_label_x,
                            label_y=_label_y,
                            label_cost_matrix=self.label_cost_matrix,
                            lambda_x=self.lambda_x,
                            lambda_y=self.lambda_y,
                            # Early stopping
                            threshold=self.threshold,
                            check_every=self.inner_iterations,
                        )
                        f_grad, g_grad = f_cost, g_cost
                elif last_extrapolation:
                    f_cost, g_cost, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
                        x,
                        y,
                        a,
                        b,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=False,
                        last_extrapolation=True,
                        allow_tf32=allow_tf32,
                        use_exp2=use_exp2,
                        eps=None,
                        n_iters=None,
                        diameter=None,
                        eps_list=eps_list,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        autotune=autotune,
                        return_prelast=True,
                        cost_scale=self.cost_scale,
                        # OTDD label-augmented cost
                        label_x=_label_x,
                        label_y=_label_y,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        # Early stopping
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                    )
                else:
                    f_cost, g_cost = sinkhorn_geomloss_online_potentials_sqeuclid(
                        x,
                        y,
                        a,
                        b,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=False,
                        last_extrapolation=False,
                        allow_tf32=allow_tf32,
                        use_exp2=use_exp2,
                        eps=None,
                        n_iters=None,
                        diameter=None,
                        eps_list=eps_list,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        autotune=autotune,
                        cost_scale=self.cost_scale,
                        # OTDD label-augmented cost
                        label_x=_label_x,
                        label_y=_label_y,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        # Early stopping
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                    )
                    f_grad, g_grad = f_cost, g_cost

                ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
                ctx.eps = float(eps_list[-1])
                ctx.rho_x = self.rho_x  # For semi-unbalanced OT gradient
                ctx.rho_y = self.rho_y
                ctx.allow_tf32 = bool(allow_tf32)
                ctx.use_exp2 = bool(use_exp2)
                ctx.autotune = bool(autotune)
                ctx.block_m = block_m
                ctx.block_n = block_n
                ctx.block_k = block_k
                ctx.num_warps = num_warps
                ctx.num_stages = int(num_stages)
                # OTDD label-augmented cost (stored for backward pass)
                ctx.label_x = _label_x
                ctx.label_y = _label_y
                ctx.label_cost_matrix_stored = self.label_cost_matrix
                ctx.lambda_x = self.lambda_x
                ctx.lambda_y = self.lambda_y

                # Cost computation: differs for balanced vs unbalanced/semi-unbalanced OT
                # Note: With epsilon_schedule matching GeomLoss exactly, potentials match
                # and the standard cost formulas apply without scaling factors.
                is_balanced_x = self.rho_x is None
                is_balanced_y = self.rho_y is None

                if is_balanced_x and is_balanced_y:
                    # Balanced OT: cost = <a, f> + <b, g>
                    return (a * f_cost).sum() + (b * g_cost).sum()
                else:
                    # Unbalanced / Semi-unbalanced OT cost computation
                    #
                    # For fully unbalanced (both rho_x and rho_y not None):
                    #   Use standard formula: (rho+eps/2)·a·(1-exp(-f/rho))
                    #   This works because symmetric damping keeps potentials positive.
                    #
                    # For semi-unbalanced (one rho is None, one is not):
                    #   Use balanced formula: <a, f> + <b, g>
                    #   The unbalanced formula fails because asymmetric damping causes
                    #   the relaxed potential to shift negative, making (1-exp(-f/rho)) negative.
                    #   The balanced formula gives correct positive costs.
                    #
                    # Note: Semi-unbalanced is a FlashSinkhorn extension not in GeomLoss.
                    is_semi_unbalanced = is_balanced_x != is_balanced_y

                    if is_semi_unbalanced:
                        # For semi-unbalanced, use balanced formula to avoid negative costs
                        return (a * f_cost).sum() + (b * g_cost).sum()
                    else:
                        # Fully unbalanced: use standard unbalanced formula
                        cost = torch.tensor(0.0, device=x.device, dtype=torch.float32)
                        unbal_weight_x = self.rho_x + ctx.eps / 2
                        cost_a = (a * (1 - (-f_cost / self.rho_x).exp())).sum()
                        cost = cost + unbal_weight_x * cost_a
                        unbal_weight_y = self.rho_y + ctx.eps / 2
                        cost_b = (b * (1 - (-g_cost / self.rho_y).exp())).sum()
                        cost = cost + unbal_weight_y * cost_b
                        return cost

            @staticmethod
            def backward(ctx, grad_out):
                x, y, a, b, f_cost, g_cost, f_grad, g_grad = ctx.saved_tensors
                grad_x = grad_y = grad_a = grad_b = None

                if x.requires_grad or y.requires_grad:
                    gx, gy = _SinkhornGradFn.apply(
                        x,
                        y,
                        a,
                        b,
                        f_grad,
                        g_grad,
                        ctx.eps,
                        ctx.allow_tf32,
                        ctx.use_exp2,
                        ctx.autotune,
                        ctx.block_m,
                        ctx.block_n,
                        ctx.block_k,
                        ctx.num_warps,
                        ctx.num_stages,
                        grad_out,
                        self.hvp_tau2,
                        self.hvp_max_cg_iter,
                        self.hvp_cg_rtol,
                        self.hvp_cg_atol,
                        self.hvp_cg_stabilise_every,
                        self.hvp_preconditioner,
                        self.hvp_precond_terms,
                        self.hvp_use_preconditioner,
                        x.requires_grad,
                        y.requires_grad,
                        ctx.rho_x,  # For semi-unbalanced OT HVP
                        ctx.rho_y,  # For semi-unbalanced OT HVP
                        self.cost_scale,  # Cost scale for half cost
                        # OTDD label-augmented cost
                        ctx.label_x,
                        ctx.label_y,
                        ctx.label_cost_matrix_stored,
                        ctx.lambda_x,
                        ctx.lambda_y,
                    )
                    grad_x = gx if x.requires_grad else None
                    grad_y = gy if y.requires_grad else None

                if a.requires_grad:
                    grad_a = grad_out * f_cost
                if b.requires_grad:
                    grad_b = grad_out * g_cost

                return (
                    grad_x,
                    grad_y,
                    grad_a,
                    grad_b,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        def _raw_cost_with_eps(
            xb: torch.Tensor, yb: torch.Tensor, ab: torch.Tensor, bb: torch.Tensor,
            eps_list: tuple,
            label_x_arg: Optional[torch.Tensor] = None,
            label_y_arg: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute raw OT cost (with 0.5 scaling for GeomLoss compatibility).

            Args:
                xb, yb: Point clouds
                ab, bb: Weights
                eps_list: Epsilon schedule
                label_x_arg, label_y_arg: Labels for OTDD (must be explicitly passed
                    for correct debiased computation where self-transport uses same labels)
            """
            # Store original labels before temporarily overriding
            nonlocal _label_x, _label_y
            original_label_x, original_label_y = _label_x, _label_y

            # Override labels for this call if provided
            if label_x_arg is not None or label_y_arg is not None:
                _label_x = label_x_arg
                _label_y = label_y_arg

            try:
                result = _SinkhornCostFn.apply(
                    xb,
                    yb,
                    ab,
                    bb,
                    eps_list,
                    self.last_extrapolation,
                    self.allow_tf32,
                    self.use_exp2,
                    self.autotune,
                    self.block_m,
                    self.block_n,
                    self.block_k,
                    self.num_warps,
                    self.num_stages,
                )
            finally:
                # Restore original labels
                _label_x, _label_y = original_label_x, original_label_y

            return result

        def _cost(xb: torch.Tensor, yb: torch.Tensor, ab: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
            """Compute OT cost, with debiasing if enabled.

            Debiased Sinkhorn divergence (when debias=True):
              S_ε(α, β) = OT_ε(α, β) - 0.5 * OT_ε(α, α) - 0.5 * OT_ε(β, β)

            This makes the divergence:
            - Zero when α = β (positive semi-definite)
            - Interpolates between OT and MMD as ε varies

            IMPORTANT: All three Sinkhorn problems use the same epsilon schedule
            (computed from the cross-transport diameter), following GeomLoss convention.

            For OTDD label-augmented cost, each OT problem must use correct labels:
            - OT(x, y): label_x for source, label_y for target
            - OT(x, x): label_x for BOTH source and target
            - OT(y, y): label_y for BOTH source and target
            """
            # Compute eps_list from the cross-transport diameter
            eps_list = tuple(self._eps_list_for_inputs(xb, yb))

            # For cross-transport, use the original labels
            cost_xy = _raw_cost_with_eps(xb, yb, ab, bb, eps_list, _label_x, _label_y)

            if not self.debias:
                return cost_xy

            # Compute self-transport terms for debiasing
            # Use the SAME eps_list as the cross-transport (GeomLoss convention)
            # CRITICAL for OTDD: Self-transport uses same labels for both source and target
            cost_xx = _raw_cost_with_eps(xb, xb, ab, ab, eps_list, _label_x, _label_x)  # OT(x, x) with labels_x, labels_x
            cost_yy = _raw_cost_with_eps(yb, yb, bb, bb, eps_list, _label_y, _label_y)  # OT(y, y) with labels_y, labels_y

            # Sinkhorn divergence: OT(x,y) - 0.5*OT(x,x) - 0.5*OT(y,y)
            return cost_xy - 0.5 * cost_xx - 0.5 * cost_yy

        # Batched inputs: match GeomLoss by returning a vector of size (B,).
        if parsed.batched:
            if self.potentials:
                f_list = []
                g_list = []
                for xb, yb, ab, bb in zip(parsed.x, parsed.y, parsed.a, parsed.b):
                    eps_list = tuple(self._eps_list_for_inputs(xb, yb))
                    fb, gb = sinkhorn_geomloss_online_potentials_sqeuclid(
                        xb,
                        yb,
                        ab,
                        bb,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=False,
                        last_extrapolation=self.last_extrapolation,
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                        eps_list=eps_list,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        block_m=self.block_m,
                        block_n=self.block_n,
                        block_k=self.block_k,
                        num_warps=self.num_warps,
                        num_stages=self.num_stages,
                        autotune=self.autotune,
                        cost_scale=self.cost_scale,
                        # OTDD label-augmented cost (batched not fully supported)
                        label_x=_label_x,
                        label_y=_label_y,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        # Early stopping
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                    )
                    f_list.append(fb)
                    g_list.append(gb)
                f_b = torch.stack(f_list, dim=0).view(parsed.a_view_shape)
                g_b = torch.stack(g_list, dim=0).view(parsed.b_view_shape)
                return f_b, g_b

            costs = [_cost(xb, yb, ab, bb) for xb, yb, ab, bb in zip(parsed.x, parsed.y, parsed.a, parsed.b)]
            return torch.stack(costs, dim=0)

        if self.potentials:
            eps_list = tuple(self._eps_list_for_inputs(parsed.x, parsed.y))
            if self.backend == "alternating":
                eps = float(eps_list[-1])
                n_iters = len(eps_list)
                if self.use_flashstyle:
                    # NEW: FlashSinkhorn alternating (supports semi-unbalanced OT)
                    f, g = sinkhorn_flashstyle_alternating(
                        parsed.x,
                        parsed.y,
                        parsed.a,
                        parsed.b,
                        eps=eps,
                        n_iters=n_iters,
                        cost_scale=self.cost_scale,
                        reach_x=self.reach_x,
                        reach_y=self.reach_y,
                        autotune=self.autotune,
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                    )
                else:
                    # OLD: OTT-style alternating Sinkhorn
                    loga = log_weights(parsed.a).contiguous()
                    logb = log_weights(parsed.b).contiguous()
                    f, g = sinkhorn_ott_potentials_sqeuclid(
                        parsed.x,
                        parsed.y,
                        loga,
                        logb,
                        eps,
                        n_iters,
                        autotune=self.autotune,
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                        block_m=self.block_m,
                        block_n=self.block_n,
                        block_k=self.block_k,
                        num_warps=self.num_warps,
                        num_stages=self.num_stages,
                    )
            else:
                # GeomLoss backend (default): symmetric-update Sinkhorn
                if self.use_flashstyle:
                    # FlashSinkhorn symmetric (supports all features)
                    f, g = sinkhorn_flashstyle_symmetric(
                        parsed.x,
                        parsed.y,
                        parsed.a,
                        parsed.b,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=self.use_epsilon_scaling,
                        eps=self.eps,
                        n_iters=self.n_iters,
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                        autotune=self.autotune,
                        cost_scale=self.cost_scale,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        last_extrapolation=self.last_extrapolation,
                        # OTDD label-augmented cost
                        label_x=_label_x,
                        label_y=_label_y,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        # Early stopping
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                    )
                else:
                    # OLD: GeomLoss-style symmetric Sinkhorn
                    f, g = sinkhorn_geomloss_online_potentials_sqeuclid(
                        parsed.x,
                        parsed.y,
                        parsed.a,
                        parsed.b,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=False,
                        last_extrapolation=self.last_extrapolation,
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                        eps_list=eps_list,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        block_m=self.block_m,
                        block_n=self.block_n,
                        block_k=self.block_k,
                        num_warps=self.num_warps,
                        num_stages=self.num_stages,
                        autotune=self.autotune,
                        cost_scale=self.cost_scale,
                        # OTDD label-augmented cost
                        label_x=_label_x,
                        label_y=_label_y,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        # Early stopping
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                    )
            return f.view(parsed.a_view_shape), g.view(parsed.b_view_shape)

        return _cost(parsed.x, parsed.y, parsed.a, parsed.b)
