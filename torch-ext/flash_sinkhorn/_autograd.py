"""Autograd Functions for Sinkhorn OT cost and gradient computation.

Extracted from samples_loss.py following FlashAttention's pattern of
separating autograd machinery from the public API module.

Contains:
- _SinkhornConfig: Frozen dataclass capturing all SamplesLoss configuration
- _SinkhornGradFn: Custom backward for analytical gradient (supports HVP)
- _SinkhornCostFn: Custom backward for OT cost (calls solver + gradient)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .hvp import geomloss_to_ott_potentials, hvp_x_sqeuclid_from_potentials
from .kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from .kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)


# ---------------------------------------------------------------------------
# Config dataclass: immutable snapshot of SamplesLoss parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SinkhornConfig:
    """Frozen snapshot of SamplesLoss configuration.

    Replaces the closure-captured ``self`` pattern. Stored on ``ctx`` in
    forward so that backward can access all parameters without a closure
    reference to the enclosing ``SamplesLoss`` instance.
    """

    # Solver config
    backend: str
    blur: float
    scaling: float
    use_epsilon_scaling: bool
    eps: Optional[float]
    n_iters: Optional[int]
    cost_scale: float
    rho_x: Optional[float]
    rho_y: Optional[float]
    last_extrapolation: bool
    # Kernel config
    allow_tf32: bool
    use_exp2: bool
    autotune: bool
    block_m: Optional[int]
    block_n: Optional[int]
    block_k: Optional[int]
    num_warps: Optional[int]
    num_stages: int
    # HVP config (used in double backward)
    hvp_tau2: float
    hvp_max_cg_iter: int
    hvp_cg_rtol: float
    hvp_cg_atol: float
    hvp_cg_stabilise_every: int
    hvp_preconditioner: str
    hvp_precond_terms: int
    hvp_use_preconditioner: bool
    # OTDD label cost weights (tensors passed separately to .apply())
    lambda_x: float
    lambda_y: float
    # Early stopping
    threshold: Optional[float]
    inner_iterations: int
    # Adaptive padding: original unpadded sizes for masked early stopping
    n_orig: Optional[int] = None
    m_orig: Optional[int] = None


# ---------------------------------------------------------------------------
# Gradient Function (double backward → HVP)
# ---------------------------------------------------------------------------


class _SinkhornGradFn(torch.autograd.Function):
    """Analytical gradient of the Sinkhorn cost w.r.t. point locations.

    Forward: computes grad_x, grad_y via the gradient kernel.
    Backward: computes Hessian-vector product (HVP) for Newton methods.
    """

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
        rho_x: Optional[float] = None,
        rho_y: Optional[float] = None,
        cost_scale: float = 1.0,
        label_x: Optional[torch.Tensor] = None,
        label_y: Optional[torch.Tensor] = None,
        label_cost_matrix: Optional[torch.Tensor] = None,
        lambda_x: float = 1.0,
        lambda_y: float = 0.0,
        f_mass: Optional[torch.Tensor] = None,
        g_mass: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # The gradient is computed at grad_scale = 1 and scaled afterwards, so that the double backward can return
        # its derivative with respect to grad_scale, which a loss composed with a nonlinear function needs.
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
            grad_scale=torch.ones_like(grad_scale),
            compute_grad_x=bool(compute_grad_x),
            compute_grad_y=bool(compute_grad_y),
            cost_scale=float(cost_scale),
            label_x=label_x,
            label_y=label_y,
            label_cost_matrix=label_cost_matrix,
            lambda_x=float(lambda_x),
            lambda_y=float(lambda_y),
        )
        # The kernel returns the balanced gradient a_i * 2cs (x_i - T_i), where T_i is the barycentre under the
        # weights of the last update (built from f_grad, g_grad). Under a KL penalty the row masses of the plan
        # are a_i exp(-f_i / rho_x) at the returned potentials (columns: b_j exp(-g_j / rho_y)), so the rows are
        # rescaled by them. With the final extrapolation this is GeomLoss's gradient through the stopped last
        # update (balanced or one common reach), without the factor (rho + eps/2) / (rho + eps) its relaxed
        # gradients carry. At convergence it is the OT gradient; before convergence it is not the total derivative
        # of the finite run, and the implicit HVP is not the derivative of this construction.
        f_mass = f_grad if f_mass is None else f_mass
        g_mass = g_grad if g_mass is None else g_mass
        if rho_x is not None and gx is not None:
            gx = gx * torch.exp(-f_mass.float().masked_fill(a == 0, 0) / rho_x).to(gx.dtype)[:, None]
        if rho_y is not None and gy is not None:
            gy = gy * torch.exp(-g_mass.float().masked_fill(b == 0, 0) / rho_y).to(gy.dtype)[:, None]

        ctx.save_for_backward(x, y, a, b, f_grad, g_grad, grad_scale, gx)
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
        ctx.rho_x = rho_x
        ctx.rho_y = rho_y
        ctx.cost_scale = float(cost_scale)
        ctx.use_label_cost = (
            label_x is not None and label_y is not None
            and label_cost_matrix is not None and lambda_y != 0.0
        )
        return (None if gx is None else gx * grad_scale), (None if gy is None else gy * grad_scale)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_x: Optional[torch.Tensor], grad_grad_y: Optional[torch.Tensor]
    ):
        x, y, a, b, f_grad, g_grad, grad_scale, gx_unit = ctx.saved_tensors

        if grad_grad_y is not None:
            raise NotImplementedError(
                "Hessian-vector products are implemented with respect to x only: y must not require a gradient, "
                "and debias must be False (the debiased cost's self term OT(x, x) has x on both sides)."
            )

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
                hvp_x, info = hvp_x_sqeuclid_from_potentials(
                    x,
                    y,
                    f_hat,
                    g_hat,
                    grad_grad_x,
                    eps=ctx.eps,
                    rho_x=ctx.rho_x,
                    rho_y=ctx.rho_y,
                    cost_scale=ctx.cost_scale,
                    tau2=ctx.hvp_tau2,
                    max_cg_iter=ctx.hvp_max_cg_iter,
                    cg_rtol=ctx.hvp_cg_rtol,
                    cg_atol=ctx.hvp_cg_atol,
                    cg_stabilise_every=ctx.hvp_cg_stabilise_every,
                    preconditioner=ctx.hvp_preconditioner,
                    precond_terms=ctx.hvp_precond_terms,
                    use_preconditioner=ctx.hvp_use_preconditioner,
                    allow_tf32=False,  # HVP requires full fp32 precision
                    use_exp2=ctx.use_exp2,
                    block_m=ctx.block_m,
                    block_n=ctx.block_n,
                    block_k=ctx.block_k,
                    num_warps=int(ctx.num_warps or 4),
                    num_stages=ctx.num_stages,
                )
                if not info.cg_converged:
                    required = max(ctx.hvp_cg_atol, ctx.hvp_cg_rtol * info.cg_initial_residual)
                    warnings.warn(
                        f"Hessian-vector product: conjugate gradients did not confirm convergence after "
                        f"{info.cg_iters} iterations (residual {info.cg_residual:.3g}, required <= {required:.3g}), "
                        "so the product may be inaccurate. Raising hvp_max_cg_iter may help. For balanced OT a larger "
                        "hvp_tau2 can improve the conditioning but changes the product; relaxed marginals ignore it.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                out_x = hvp_x * grad_scale

        # One entry per forward input; grad_scale is input 15. d(grad_x . v) / d grad_scale = <unit gradient, v>.
        grads = [None] * len(ctx.needs_input_grad)
        grads[0] = out_x
        if grad_grad_x is not None and ctx.needs_input_grad[15] and gx_unit is not None:
            grads[15] = (gx_unit * grad_grad_x).sum().reshape(grad_scale.shape)
        return tuple(grads)


# ---------------------------------------------------------------------------
# Cost Function (forward: run Sinkhorn solver, backward: analytical gradient)
# ---------------------------------------------------------------------------


def _dual_cost(a, b, f, g, rho_x, rho_y, eps):
    """Cost of balanced, unbalanced or semi-unbalanced entropic OT from its potentials (GeomLoss convention).

    A strict marginal contributes ``<w, pot>``; a marginal relaxed by a KL penalty of strength ``rho``
    contributes ``(rho + eps / 2) * <w, 1 - exp(-pot / rho)>``. For probability weights and optimal
    potentials this equals the value of the primal problem; it is not the dual objective at other potentials.
    """
    def side(w, pot, rho):
        pot = torch.where(w > 0, pot, torch.zeros_like(pot))  # 0 * inf would give nan
        if rho is None:
            return (w * pot).sum()
        return (rho + eps / 2) * (w * (1 - (-pot / rho).exp())).sum()

    return side(a, f, rho_x) + side(b, g, rho_y)


class _SinkhornCostFn(torch.autograd.Function):
    """Compute Sinkhorn OT cost with analytical backward pass.

    This is a module-level autograd.Function (following FlashAttention's pattern).
    All configuration is passed via a frozen ``_SinkhornConfig`` dataclass rather
    than captured via closure.

    Args to .apply():
        x, y, a, b: Point clouds and weights (tensors, tracked by autograd)
        eps_list: Epsilon schedule (tuple of floats, non-tensor)
        config: _SinkhornConfig (frozen dataclass, non-tensor)
        label_x, label_y: Optional OTDD label tensors (may be None)
        label_cost_matrix: Optional (V, V) label distance matrix (may be None)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        eps_list: Tuple[float, ...],
        config: _SinkhornConfig,
        label_x: Optional[torch.Tensor],
        label_y: Optional[torch.Tensor],
        label_cost_matrix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # OTT backend: use alternating-update Sinkhorn (matches OTT-JAX)
        if config.backend == "alternating":
            eps = float(eps_list[-1])  # Fixed eps for OTT backend
            n_iters = len(eps_list)

            f_cost, g_cost = sinkhorn_flashstyle_alternating(
                x,
                y,
                a,
                b,
                eps=eps,
                n_iters=n_iters,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                autotune=config.autotune,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
            f_grad, g_grad = f_cost, g_cost

            ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
            ctx.eps = eps
            ctx.rho_x = config.rho_x
            ctx.rho_y = config.rho_y
            ctx.allow_tf32 = config.allow_tf32
            ctx.use_exp2 = config.use_exp2
            ctx.autotune = config.autotune
            ctx.block_m = config.block_m
            ctx.block_n = config.block_n
            ctx.block_k = config.block_k
            ctx.num_warps = config.num_warps
            ctx.num_stages = config.num_stages
            ctx.label_x = None
            ctx.label_y = None
            ctx.label_cost_matrix_stored = None
            ctx.lambda_x = 1.0
            ctx.lambda_y = 0.0
            ctx.config = config

            return _dual_cost(a, b, f_cost, g_cost, config.rho_x, config.rho_y, eps)

        # GeomLoss backend (default): symmetric-update Sinkhorn
        if config.last_extrapolation:
            f_cost, g_cost, f_grad, g_grad = sinkhorn_flashstyle_symmetric(
                x,
                y,
                a,
                b,
                blur=config.blur,
                scaling=config.scaling,
                use_epsilon_scaling=config.use_epsilon_scaling,
                eps=config.eps,
                n_iters=config.n_iters,
                eps_list=list(eps_list),
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                last_extrapolation=True,
                return_prelast=True,
                label_x=label_x,
                label_y=label_y,
                label_cost_matrix=label_cost_matrix,
                lambda_x=config.lambda_x,
                lambda_y=config.lambda_y,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
        else:
            f_cost, g_cost = sinkhorn_flashstyle_symmetric(
                x,
                y,
                a,
                b,
                blur=config.blur,
                scaling=config.scaling,
                use_epsilon_scaling=config.use_epsilon_scaling,
                eps=config.eps,
                n_iters=config.n_iters,
                eps_list=list(eps_list),
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                last_extrapolation=False,
                label_x=label_x,
                label_y=label_y,
                label_cost_matrix=label_cost_matrix,
                lambda_x=config.lambda_x,
                lambda_y=config.lambda_y,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
            f_grad, g_grad = f_cost, g_cost

        ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
        ctx.eps = float(eps_list[-1])
        ctx.rho_x = config.rho_x
        ctx.rho_y = config.rho_y
        ctx.allow_tf32 = config.allow_tf32
        ctx.use_exp2 = config.use_exp2
        ctx.autotune = config.autotune
        ctx.block_m = config.block_m
        ctx.block_n = config.block_n
        ctx.block_k = config.block_k
        ctx.num_warps = config.num_warps
        ctx.num_stages = config.num_stages
        ctx.label_x = label_x
        ctx.label_y = label_y
        ctx.label_cost_matrix_stored = label_cost_matrix
        ctx.lambda_x = config.lambda_x
        ctx.lambda_y = config.lambda_y
        ctx.config = config

        return _dual_cost(a, b, f_cost, g_cost, config.rho_x, config.rho_y, ctx.eps)

    @staticmethod
    def backward(ctx, grad_out):
        x, y, a, b, f_cost, g_cost, f_grad, g_grad = ctx.saved_tensors
        config = ctx.config
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
                config.hvp_tau2,
                config.hvp_max_cg_iter,
                config.hvp_cg_rtol,
                config.hvp_cg_atol,
                config.hvp_cg_stabilise_every,
                config.hvp_preconditioner,
                config.hvp_precond_terms,
                config.hvp_use_preconditioner,
                x.requires_grad,
                y.requires_grad,
                ctx.rho_x,
                ctx.rho_y,
                config.cost_scale,
                ctx.label_x,
                ctx.label_y,
                ctx.label_cost_matrix_stored,
                ctx.lambda_x,
                ctx.lambda_y,
                f_cost,
                g_cost,
            )
            grad_x = gx if x.requires_grad else None
            grad_y = gy if y.requires_grad else None

        def weight_grad(w, pot, rho):
            # d OT / d w: the potential on a strict side, (rho + eps)(1 - exp(-pot / rho)) on a relaxed one
            # (probability weights; the normalization in SamplesLoss adds its own chain rule).
            if rho is None:
                return pot
            pot = torch.where(w > 0, pot, torch.zeros_like(pot))
            return (rho + ctx.eps) * (1 - (-pot / rho).exp())

        if a.requires_grad:
            grad_a = grad_out * weight_grad(a, f_cost, ctx.rho_x)
        if b.requires_grad:
            grad_b = grad_out * weight_grad(b, g_cost, ctx.rho_y)

        # Returns: x, y, a, b, eps_list, config, label_x, label_y, label_cost_matrix
        return (
            grad_x,
            grad_y,
            grad_a,
            grad_b,
            None,  # eps_list
            None,  # config
            None,  # label_x
            None,  # label_y
            None,  # label_cost_matrix
        )
