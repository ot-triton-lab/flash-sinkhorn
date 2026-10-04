"""High-level SamplesLoss API for Sinkhorn OT (GeomLoss-compatible).

This module provides the ``SamplesLoss`` class which wraps FlashSinkhorn
solvers and gradient kernels behind a simple ``nn.Module`` interface.

The autograd machinery (``_SinkhornCostFn``, ``_SinkhornGradFn``) lives in
``_autograd.py`` — this module contains only the public API and input parsing.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

import warnings

from ._autograd import _SinkhornConfig, _SinkhornCostFn
from .kernels._common import epsilon_schedule, max_diameter
from .kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)

#: Default target marginal residual of backend="multiscale".
MULTISCALE_DEFAULT_TOL = 5e-3

# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedInputs:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    batched: bool
    a_view_shape: Tuple[int, ...]
    b_view_shape: Tuple[int, ...]


def _center_coords(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, batched: bool):
    """Translate both clouds by their joint mass-weighted centroid, in fp32.

    Squared-Euclidean OT is translation invariant, so the cost, the potentials and the gradients are unchanged in
    exact arithmetic. The symmetric and alternating solvers subtract alpha = cost_scale * |x|^2 from the potentials
    explicitly; far from the origin the fp32 rounding of alpha approaches eps and destroys the potentials without
    any error. The result is fp32 whatever the input dtype: subtracting in fp16 or bf16 can erase gaps the input
    holds exactly and overflow finite fp16 inputs, and the solvers compute in fp32 anyway. The centroid is detached,
    so it is a constant shift for autograd.
    """
    with torch.no_grad():
        dims = 1 if batched else 0
        total = (a.float().sum(dim=-1) + b.float().sum(dim=-1)).clamp_min(torch.finfo(torch.float32).tiny)
        mu = ((a.float().unsqueeze(-1) * x.float()).sum(dim=dims)
              + (b.float().unsqueeze(-1) * y.float()).sum(dim=dims)) / total.unsqueeze(-1)
        if batched:
            mu = mu.unsqueeze(1)
    return x.float() - mu, y.float() - mu


def _round_to_tf32(t: torch.Tensor) -> torch.Tensor:
    """Round to nearest-even TF32 values with an identity autograd derivative.

    Run after centring: subtracting the centroid produces new FP32 values.
    Rounding makes the FP32 norms and TF32 dot products use the same coordinates;
    FP32 arithmetic error remains. The detached displacement preserves first
    derivatives and double backward. fp16/bf16 tensors need no further rounding.
    """
    from .multiscale._preprocess import round_to_tf32

    if t.dtype not in (torch.float32, torch.float64):
        return t
    return t + (round_to_tf32(t) - t).detach()


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


# ---------------------------------------------------------------------------
# Adaptive padding helpers
# ---------------------------------------------------------------------------


def _pad_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    multiple: int,
    label_x: Optional[torch.Tensor] = None,
    label_y: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
    """Pad point clouds and weights to next multiple of ``multiple``.

    Zero-weight padding is mathematically exact for Sinkhorn: phantom points
    carry no mass, so they do not affect the transport plan or OT cost.
    """
    n, d = x.shape
    m = y.shape[0]
    pad_n = (multiple - n % multiple) % multiple
    pad_m = (multiple - m % multiple) % multiple
    if pad_n == 0 and pad_m == 0:
        return x, y, a, b, label_x, label_y, n, m
    if pad_n > 0:
        x = torch.cat([x, x.new_zeros(pad_n, d)])
        a = torch.cat([a, a.new_zeros(pad_n)])
        if label_x is not None:
            label_x = torch.cat([label_x, label_x.new_zeros(pad_n, dtype=label_x.dtype)])
    if pad_m > 0:
        y = torch.cat([y, y.new_zeros(pad_m, d)])
        b = torch.cat([b, b.new_zeros(pad_m)])
        if label_y is not None:
            label_y = torch.cat([label_y, label_y.new_zeros(pad_m, dtype=label_y.dtype)])
    return x, y, a, b, label_x, label_y, n, m


def _trim_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    n_orig: int,
    m_orig: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim padded potentials back to original size."""
    return f[:n_orig], g[:m_orig]


# ---------------------------------------------------------------------------
# SamplesLoss: public API
# ---------------------------------------------------------------------------


class SamplesLoss(torch.nn.Module):
    """GeomLoss-like API for (online) Sinkhorn OT using Triton.

    This is a minimal, CUDA-only subset of GeomLoss's `SamplesLoss`:
    - `loss` must be "sinkhorn".
    - Only squared Euclidean ground cost is supported (with optional label cost).
    - Supports balanced, unbalanced, and semi-unbalanced OT.

    Three solvers are available (all use FlashSinkhorn kernels):
    - `backend="symmetric"` (default): Symmetric Sinkhorn (Jacobi-style updates).
      Supports all features: debiasing, unbalanced OT, epsilon scaling, label cost.
    - `backend="alternating"`: Alternating Sinkhorn (Gauss-Seidel-style updates).
      Requires fixed eps and n_iters.
      Supports: debiasing, unbalanced/semi-unbalanced OT.
      Does NOT support: epsilon scaling or label cost.
    - `backend="multiscale"`: for large three-dimensional point clouds. Solves on
      Morton-cell centroids, lifts to the points and finishes with block-sparse
      updates, returning once a sampled check finds the marginal residual
      max(||P1 - a||_1, ||P^T 1 - b||_1) below `tol` (default 5e-3). It solves the
      centered, TF32-rounded clouds at eps = blur**2 with its own annealing schedule.
      Supports: balanced OT, debiasing, half_cost, potentials. Forward only (no
      gradients); no unbalanced OT, label cost or batched inputs.

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

    Where W is a precomputed (V × V) label-to-label distance matrix, the whole multiplied by cost_scale. The label
    term is active only when label_cost_matrix, both label vectors and lambda_y != 0 are given; otherwise the cost is
    the plain feature cost and lambda_x is ignored.

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

    Notes
    -----
    - Dense costs and weight gradients assume probability weights on each side.
      With ``normalize=False``, supply weights that already sum to one; arbitrary
      input mass totals are not supported for these outputs.
    - ``allow_tf32=True`` enables TF32 cost dot products. ``truncate_tf32=None``
      follows ``allow_tf32``: the dense backends round centred FP32/FP64 input
      coordinates to nearest-even TF32 values before padding, so the norms and
      dot products describe the same rounded cloud. FP32 arithmetic error remains.
      fp16/bf16 inputs receive no additional rounding. Rounding uses an identity
      derivative for gradients and supported HVPs. Set ``truncate_tf32=False`` to
      retain unrounded centred coordinates, or ``True`` to round even with FP32
      dot products. Multiscale always rounds and rejects ``truncate_tf32=False``.
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
        truncate_tf32: Optional[bool] = None,
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
        threshold: Optional[float] = None,
        inner_iterations: int = 10,
        # Deprecated: FlashSinkhorn is now the only backend
        use_flashstyle: Optional[bool] = None,
        # Adaptive padding for variable-size OT
        pad_to_multiple: Optional[int] = None,
        # Target marginal residual of backend="multiscale"
        tol: Optional[float] = None,
    ):
        super().__init__()

        if use_flashstyle is not None and not use_flashstyle:
            warnings.warn(
                "use_flashstyle=False is deprecated. FlashSinkhorn is now the only backend.",
                DeprecationWarning,
                stacklevel=2,
            )

        if loss != "sinkhorn":
            raise ValueError('Only loss="sinkhorn" is supported.')
        if p != 2:
            raise ValueError("Only p=2 (squared Euclidean cost) is supported.")
        if backend not in ("symmetric", "alternating", "multiscale", "triton", "auto"):
            raise ValueError(
                'Only backend in {"symmetric","alternating","multiscale","triton","auto"} is supported.'
            )
        # Normalize legacy aliases to canonical names
        if backend in ("triton", "auto"):
            warnings.warn(
                f'backend="{backend}" is a legacy alias for "symmetric". '
                'Use backend="symmetric" or backend="alternating" explicitly.',
                DeprecationWarning,
                stacklevel=2,
            )
            backend = "symmetric"
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
            if label_cost_matrix is not None:
                raise ValueError(
                    'backend="alternating" does not support OTDD label cost. '
                    'Use backend="symmetric" for label-augmented cost.'
                )
        truncate_tf32 = bool(allow_tf32) if truncate_tf32 is None else bool(truncate_tf32)
        if backend == "multiscale":
            unsupported = dict(reach=reach, reach_x=reach_x, reach_y=reach_y,
                               label_cost_matrix=label_cost_matrix, threshold=threshold, eps=eps,
                               eps_list=eps_list, n_iters=n_iters, diameter=diameter,
                               pad_to_multiple=pad_to_multiple)
            given = [name for name, value in unsupported.items() if value is not None]
            given += [name for name, value in (("allow_tf32=False", allow_tf32),
                                               ("truncate_tf32=False", truncate_tf32),
                                               ("use_epsilon_scaling=False", use_epsilon_scaling))
                      if not value]
            if given:
                raise ValueError(f'backend="multiscale" does not support {", ".join(given)}. '
                                 'It always rounds centered coordinates to TF32.')
            tol = MULTISCALE_DEFAULT_TOL if tol is None else float(tol)
            if not 0.0 < tol <= 1.0:
                raise ValueError("tol must be in (0, 1].")
        elif tol is not None:
            raise ValueError('tol applies only to backend="multiscale".')
        if reach is not None and reach <= 0:
            raise ValueError("reach must be positive (or None for balanced OT).")
        if reach_x is not None and reach_x <= 0:
            raise ValueError("reach_x must be positive (or None for balanced source).")
        if reach_y is not None and reach_y <= 0:
            raise ValueError("reach_y must be positive (or None for balanced target).")

        # Handle reach -> reach_x/reach_y conversion (legacy API compatibility)
        if reach is not None:
            if reach_x is None:
                reach_x = reach
            if reach_y is None:
                reach_y = reach

        self.loss = loss
        self.p = p
        self.blur = float(blur)
        self.reach_x = None if reach_x is None else float(reach_x)
        self.reach_y = None if reach_y is None else float(reach_y)
        self.rho_x = None if reach_x is None else float(reach_x) ** 2
        self.rho_y = None if reach_y is None else float(reach_y) ** 2
        if reach_x == reach_y:
            self.reach = self.reach_x
            self.rho = self.rho_x
        else:
            self.reach = None
            self.rho = None
        self.scaling = float(scaling)
        self.debias = bool(debias)
        self.potentials = bool(potentials)
        self.backend = backend
        self.tol = tol
        #: MultiscaleInfo of the last backend="multiscale" solve of OT(a, b), or None.
        self.last_multiscale_info = None
        self.normalize = bool(normalize)

        self.use_epsilon_scaling = bool(use_epsilon_scaling)
        self.last_extrapolation = bool(last_extrapolation)
        self.allow_tf32 = bool(allow_tf32)
        self.truncate_tf32 = truncate_tf32
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

        self.label_cost_matrix = label_cost_matrix
        self.lambda_x = float(lambda_x)
        self.lambda_y = float(lambda_y)

        self.threshold = None if threshold is None else float(threshold)
        self.inner_iterations = int(inner_iterations)

        # Adaptive padding
        self.pad_to_multiple = None if pad_to_multiple is None else int(pad_to_multiple)
        if self.pad_to_multiple is not None:
            if self.pad_to_multiple < 32 or self.pad_to_multiple % 32 != 0:
                raise ValueError(
                    "pad_to_multiple must be a positive multiple of 32 "
                    f"(e.g. 32, 64, 128), got {pad_to_multiple}"
                )

    def _make_config(self) -> _SinkhornConfig:
        """Create a frozen config snapshot for autograd."""
        return _SinkhornConfig(
            backend=self.backend,
            blur=self.blur,
            scaling=self.scaling,
            use_epsilon_scaling=self.use_epsilon_scaling,
            eps=self.eps,
            n_iters=self.n_iters,
            cost_scale=self.cost_scale,
            rho_x=self.rho_x,
            rho_y=self.rho_y,
            last_extrapolation=self.last_extrapolation,
            allow_tf32=self.allow_tf32,
            use_exp2=self.use_exp2,
            autotune=self.autotune,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            hvp_tau2=self.hvp_tau2,
            hvp_max_cg_iter=self.hvp_max_cg_iter,
            hvp_cg_rtol=self.hvp_cg_rtol,
            hvp_cg_atol=self.hvp_cg_atol,
            hvp_cg_stabilise_every=self.hvp_cg_stabilise_every,
            hvp_preconditioner=self.hvp_preconditioner,
            hvp_precond_terms=self.hvp_precond_terms,
            hvp_use_preconditioner=self.hvp_use_preconditioner,
            lambda_x=self.lambda_x,
            lambda_y=self.lambda_y,
            threshold=self.threshold,
            inner_iterations=self.inner_iterations,
        )

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

    def _forward_multiscale(self, parsed, label_x=None, label_y=None):
        """Cost or potentials from the multiscale solver (forward only).

        The solver runs on the centered, TF32-rounded coordinates, for the target
        ``eps = blur**2``, until its sampled check finds the marginal residual below
        ``tol`` or it stops unconfirmed and warns; it uses its own annealing schedule,
        so ``scaling`` is not used. With ``debias=True`` each of the three problems is
        checked against ``tol`` separately.
        """
        from .multiscale import solve_multiscale
        from .multiscale._preprocess import preprocess_coordinates

        if parsed.batched:
            raise ValueError('backend="multiscale" does not support batched inputs.')
        if label_x is not None or label_y is not None:
            raise ValueError('backend="multiscale" does not support label costs.')
        if parsed.x.shape[-1] != 3 or parsed.y.shape[-1] != 3:
            raise ValueError('backend="multiscale" supports three-dimensional point clouds.')
        if not self.potentials and any(t.requires_grad for t in (parsed.x, parsed.y, parsed.a, parsed.b)):
            raise NotImplementedError('backend="multiscale" does not support gradients.')
        # The solver uses the full cost ||x - y||^2; a scaled cost s * ||x - y||^2 at eps is the
        # full-cost problem at eps / s, with potentials scaled by s.
        s = self.cost_scale
        eps = self.blur ** 2 / s
        not_accepted = []

        def solve(x, y, a, b, name):
            f, g, info = solve_multiscale(x, y, a, b, eps, self.tol)
            if not info.accepted:
                not_accepted.append((name, info))
            return s * f, s * g, info

        a, b = parsed.a.detach().float().contiguous(), parsed.b.detach().float().contiguous()
        x, y = preprocess_coordinates(parsed.x, parsed.y, a, b)
        f, g, self.last_multiscale_info = solve(x, y, a, b, "OT(a, b)")
        if self.potentials:
            result = (f.view(parsed.a_view_shape), g.view(parsed.b_view_shape))
        else:
            result = (a * f).sum() + (b * g).sum()
            if self.debias:
                fx, gx, _ = solve(x, x, a, a, "OT(a, a)")
                fy, gy, _ = solve(y, y, b, b, "OT(b, b)")
                result = result - 0.5 * ((a * fx).sum() + (a * gx).sum()) - 0.5 * ((b * fy).sum() + (b * gy).sum())
        if not_accepted:
            stops = "; ".join(f"{name} stopped as {info.stop} and returned {info.returned}@{info.returned_step} "
                              f"({info.detail})" for name, info in not_accepted)
            warnings.warn(f'backend="multiscale": the sampled check did not confirm tol={self.tol:g} in '
                          f"{len(not_accepted)} solve(s): {stops}. The best candidate was returned unconfirmed; "
                          "the API documentation's Limits say how to read these numbers.",
                          RuntimeWarning, stacklevel=3)
        return result

    def forward(
        self,
        *args: Union[torch.Tensor, float],
        label_x: Optional[torch.Tensor] = None,
        label_y: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        parsed = _process_args(*args, normalize=self.normalize)

        if not parsed.x.is_cuda or not parsed.y.is_cuda:
            raise ValueError("flash_sinkhorn.SamplesLoss requires CUDA tensors.")

        if self.backend == "multiscale":
            return self._forward_multiscale(parsed, label_x, label_y)

        cx, cy = _center_coords(parsed.x, parsed.y, parsed.a, parsed.b, batched=parsed.batched)
        if self.truncate_tf32:
            # Centring promotes every dtype to FP32; preserve the input's precision policy per side.
            if parsed.x.dtype in (torch.float32, torch.float64):
                cx = _round_to_tf32(cx)
            if parsed.y.dtype in (torch.float32, torch.float64):
                cy = _round_to_tf32(cy)
        parsed = dataclasses.replace(parsed, x=cx, y=cy)
        config = self._make_config()

        # --- Adaptive padding ---
        should_pad = self.pad_to_multiple is not None

        x, y, a, b = parsed.x, parsed.y, parsed.a, parsed.b
        lx_used, ly_used = label_x, label_y
        n_orig = x.shape[-2] if parsed.batched else x.shape[0]
        m_orig = y.shape[-2] if parsed.batched else y.shape[0]

        # Pre-compute eps_list from UNPADDED geometry
        if not parsed.batched:
            _precomputed_eps_list = tuple(self._eps_list_for_inputs(parsed.x, parsed.y))
        else:
            _precomputed_eps_list = None

        if should_pad:
            if parsed.batched:
                B, N, D = x.shape
                M = y.shape[1]
                pad_n = (self.pad_to_multiple - N % self.pad_to_multiple) % self.pad_to_multiple
                pad_m = (self.pad_to_multiple - M % self.pad_to_multiple) % self.pad_to_multiple
                if pad_n > 0:
                    x = torch.cat([x, x.new_zeros(B, pad_n, D)], dim=1)
                    a = torch.cat([a, a.new_zeros(B, pad_n)], dim=1)
                    if lx_used is not None:
                        lx_used = torch.cat([lx_used, lx_used.new_zeros(pad_n, dtype=lx_used.dtype)])
                if pad_m > 0:
                    y = torch.cat([y, y.new_zeros(B, pad_m, D)], dim=1)
                    b = torch.cat([b, b.new_zeros(B, pad_m)], dim=1)
                    if ly_used is not None:
                        ly_used = torch.cat([ly_used, ly_used.new_zeros(pad_m, dtype=ly_used.dtype)])
            else:
                x, y, a, b, lx_used, ly_used, _, _ = _pad_inputs(
                    x, y, a, b, self.pad_to_multiple,
                    label_x=label_x, label_y=label_y,
                )
            config = dataclasses.replace(config, n_orig=n_orig, m_orig=m_orig)

        def _raw_cost(xb, yb, ab, bb, lx, ly, eps_list,
                      override_n_orig=None, override_m_orig=None):
            """Compute raw OT cost via autograd-wrapped Sinkhorn solver, on the given epsilon schedule."""
            cfg = config
            if override_n_orig is not None or override_m_orig is not None:
                cfg = dataclasses.replace(
                    config,
                    n_orig=override_n_orig if override_n_orig is not None else config.n_orig,
                    m_orig=override_m_orig if override_m_orig is not None else config.m_orig,
                )
            return _SinkhornCostFn.apply(
                xb, yb, ab, bb,
                eps_list, cfg,
                lx, ly, self.label_cost_matrix,
            )

        def _cost(xb, yb, ab, bb, xb_unpadded=None, yb_unpadded=None):
            """Compute OT cost, with debiasing if enabled.

            Debiased Sinkhorn divergence (when debias=True):
              S_eps(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)

            For OTDD label-augmented cost, each OT problem uses correct labels:
            - OT(x, y): label_x for source, label_y for target
            - OT(x, x): label_x for BOTH source and target
            - OT(y, y): label_y for BOTH source and target
            """
            # One schedule for all three problems, from the unpadded OT(a, b) geometry, as in GeomLoss.
            if _precomputed_eps_list is not None:
                eps_list = _precomputed_eps_list
            elif xb_unpadded is not None:
                eps_list = tuple(self._eps_list_for_inputs(xb_unpadded, yb_unpadded))
            else:
                eps_list = tuple(self._eps_list_for_inputs(xb, yb))
            cost_xy = _raw_cost(xb, yb, ab, bb, lx_used, ly_used, eps_list)

            if not self.debias:
                return cost_xy

            cost_xx = _raw_cost(
                xb, xb, ab, ab, lx_used, lx_used, eps_list,
                override_n_orig=n_orig if should_pad else None,
                override_m_orig=n_orig if should_pad else None,
            )
            cost_yy = _raw_cost(
                yb, yb, bb, bb, ly_used, ly_used, eps_list,
                override_n_orig=m_orig if should_pad else None,
                override_m_orig=m_orig if should_pad else None,
            )
            return cost_xy - 0.5 * cost_xx - 0.5 * cost_yy

        # --- Batched path ---
        if parsed.batched:
            if self.potentials:
                f_list = []
                g_list = []
                for xb, yb, ab, bb, xb_orig, yb_orig in zip(x, y, a, b, parsed.x, parsed.y):
                    eps_list_b = self._eps_list_for_inputs(xb_orig, yb_orig)
                    if self.backend == "alternating":
                        fb, gb = sinkhorn_flashstyle_alternating(
                            xb, yb, ab, bb,
                            eps=float(eps_list_b[-1]),
                            n_iters=len(eps_list_b),
                            cost_scale=self.cost_scale,
                            reach_x=self.reach_x,
                            reach_y=self.reach_y,
                            autotune=self.autotune,
                            allow_tf32=self.allow_tf32,
                            use_exp2=self.use_exp2,
                            threshold=self.threshold,
                            check_every=self.inner_iterations,
                            n_orig=n_orig if should_pad else None,
                            m_orig=m_orig if should_pad else None,
                        )
                        if should_pad:
                            fb, gb = _trim_potentials(fb, gb, n_orig, m_orig)
                        f_list.append(fb)
                        g_list.append(gb)
                        continue
                    fb, gb = sinkhorn_flashstyle_symmetric(
                        xb, yb, ab, bb,
                        blur=self.blur,
                        scaling=self.scaling,
                        use_epsilon_scaling=self.use_epsilon_scaling,
                        eps=self.eps,
                        n_iters=self.n_iters,
                        eps_list=list(eps_list_b),
                        allow_tf32=self.allow_tf32,
                        use_exp2=self.use_exp2,
                        autotune=self.autotune,
                        cost_scale=self.cost_scale,
                        rho_x=self.rho_x,
                        rho_y=self.rho_y,
                        last_extrapolation=self.last_extrapolation,
                        label_x=lx_used,
                        label_y=ly_used,
                        label_cost_matrix=self.label_cost_matrix,
                        lambda_x=self.lambda_x,
                        lambda_y=self.lambda_y,
                        threshold=self.threshold,
                        check_every=self.inner_iterations,
                        n_orig=n_orig if should_pad else None,
                        m_orig=m_orig if should_pad else None,
                    )
                    if should_pad:
                        fb, gb = _trim_potentials(fb, gb, n_orig, m_orig)
                    f_list.append(fb)
                    g_list.append(gb)
                f_b = torch.stack(f_list, dim=0).view(parsed.a_view_shape)
                g_b = torch.stack(g_list, dim=0).view(parsed.b_view_shape)
                return f_b, g_b

            costs = [
                _cost(xb, yb, ab, bb,
                      xb_unpadded=xb_orig, yb_unpadded=yb_orig)
                for xb, yb, ab, bb, xb_orig, yb_orig in zip(
                    x, y, a, b, parsed.x, parsed.y,
                )
            ]
            return torch.stack(costs, dim=0)

        # --- Potentials path ---
        if self.potentials:
            eps_list = _precomputed_eps_list
            if self.backend == "alternating":
                eps = float(eps_list[-1])
                n_iters = len(eps_list)
                f, g = sinkhorn_flashstyle_alternating(
                    x, y, a, b,
                    eps=eps,
                    n_iters=n_iters,
                    cost_scale=self.cost_scale,
                    reach_x=self.reach_x,
                    reach_y=self.reach_y,
                    autotune=self.autotune,
                    allow_tf32=self.allow_tf32,
                    use_exp2=self.use_exp2,
                    threshold=self.threshold,
                    check_every=self.inner_iterations,
                    n_orig=n_orig if should_pad else None,
                    m_orig=m_orig if should_pad else None,
                )
            else:
                f, g = sinkhorn_flashstyle_symmetric(
                    x, y, a, b,
                    blur=self.blur,
                    scaling=self.scaling,
                    use_epsilon_scaling=self.use_epsilon_scaling,
                    eps=self.eps,
                    n_iters=self.n_iters,
                    eps_list=list(eps_list),
                    allow_tf32=self.allow_tf32,
                    use_exp2=self.use_exp2,
                    autotune=self.autotune,
                    cost_scale=self.cost_scale,
                    rho_x=self.rho_x,
                    rho_y=self.rho_y,
                    last_extrapolation=self.last_extrapolation,
                    label_x=lx_used,
                    label_y=ly_used,
                    label_cost_matrix=self.label_cost_matrix,
                    lambda_x=self.lambda_x,
                    lambda_y=self.lambda_y,
                    threshold=self.threshold,
                    check_every=self.inner_iterations,
                    n_orig=n_orig if should_pad else None,
                    m_orig=m_orig if should_pad else None,
                )
            if should_pad:
                f, g = _trim_potentials(f, g, n_orig, m_orig)
            return f.view(parsed.a_view_shape), g.view(parsed.b_view_shape)

        # --- Standard cost path ---
        return _cost(x, y, a, b)
