"""Hessian-vector product (HVP) utilities for optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "HvpInfo",
    "geomloss_to_ott_potentials",
    "hvp_x_sqeuclid",
    "hvp_x_sqeuclid_from_potentials",
    "hvp_x_sqeuclid_multiscale",
]

from ot_triton.cg import conjugate_gradient
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import log_weights
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
)
from ot_triton.kernels.sinkhorn_triton_apply_blocksparse_sqeuclid import (
    apply_plan_mat_sqeuclid_taskcsr,
    apply_plan_vec_sqeuclid_taskcsr,
    mat5_sqeuclid_taskcsr,
)
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
)


@dataclass(frozen=True)
class HvpInfo:
    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float


def geomloss_to_ott_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert GeomLoss-style potentials to OTT-style potentials.

    GeomLoss convention corresponds to a plan:
      P = diag(a) * exp((f+g-C)/eps) * diag(b)
    OTT convention uses:
      P = exp((f_hat+g_hat-C)/eps)
    with:
      f_hat = f + eps*log(a), g_hat = g + eps*log(b).
    """

    loga = log_weights(a)
    logb = log_weights(b)
    eps_f = float(eps)
    return f.float() + eps_f * loga, g.float() + eps_f * logb


def sinkhorn_prelast_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps_list: Sequence[float],
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return prelast potentials (f_grad, g_grad) at final epsilon."""

    f_cost, g_cost, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        eps_list=eps_list,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        autotune=autotune,
        return_prelast=True,
    )
    del f_cost, g_cost
    return f_grad, g_grad


def hvp_x_sqeuclid_from_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, HvpInfo]:
    """OTT-Hessian-style HVP w.r.t x using streaming transport primitives.

    Notes
    -----
    - This implementation is correctness-first and uses vector transport kernels
      for all plan applications. It is *not* optimized yet (matrix applications
      are performed by looping over feature dimensions).
    - No cost/plan materialization occurs (except implicit work inside kernels).
    """

    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    if not x.is_cuda or not y.is_cuda or not f_hat.is_cuda or not g_hat.is_cuda:
        raise ValueError("hvp_x_sqeuclid_from_potentials requires CUDA tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)

    x_norm2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y_norm2 = (y.float() * y.float()).sum(dim=1).contiguous()

    def apply_axis1(vec_m: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_sqeuclid(
            x,
            y,
            f_hat,
            g_hat,
            vec_m,
            eps=eps_f,
            axis=1,
            x2=x_norm2,
            y2=y_norm2,
            allow_tf32=allow_tf32,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            use_exp2=use_exp2,
        )

    def apply_axis0(vec_n: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_sqeuclid(
            x,
            y,
            f_hat,
            g_hat,
            vec_n,
            eps=eps_f,
            axis=0,
            x2=x_norm2,
            y2=y_norm2,
            allow_tf32=allow_tf32,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            use_exp2=use_exp2,
        )

    ones_m = torch.ones((m,), device=x.device, dtype=torch.float32)
    ones_n = torch.ones((n,), device=x.device, dtype=torch.float32)
    a_hat = apply_axis1(ones_m)
    b_hat = apply_axis0(ones_n)

    vec1 = torch.sum(x * A, dim=1)  # (n,)
    Py = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps_f,
        axis=1,
        x2=x_norm2,
        y2=y_norm2,
        allow_tf32=allow_tf32,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=use_exp2,
    )

    x1 = 2.0 * (a_hat * vec1 - torch.sum(A * Py, dim=1))

    PT_A = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps_f,
        axis=0,
        x2=x_norm2,
        y2=y_norm2,
        allow_tf32=allow_tf32,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=use_exp2,
    )
    x2 = 2.0 * (apply_axis0(vec1) - torch.sum(y * PT_A, dim=1))

    y1 = x1 / a_hat
    y2_raw = -apply_axis0(y1) + x2
    denom = b_hat + eps_f * float(tau2)

    def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
        piz = apply_axis1(z_vec)
        piT_over_a_piz = apply_axis0(piz / a_hat)
        return piT_over_a_piz / denom

    precond_mode = str(preconditioner).lower()
    if not use_preconditioner:
        precond_mode = "none"
    if precond_mode not in ("none", "neumann", "jacobi"):
        raise ValueError("preconditioner must be one of {'none','neumann','jacobi'}.")
    precond_terms_i = int(precond_terms)
    if precond_terms_i < 0:
        raise ValueError("precond_terms must be >= 0.")

    if precond_mode == "none":
        rhs = y2_raw

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            piz = apply_axis1(z_vec)
            piT_over_a_piz = apply_axis0(piz / a_hat)
            return denom * z_vec - piT_over_a_piz

        z, cg_info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=None,
            stabilise_every=cg_stabilise_every,
        )
    else:
        rhs = y2_raw / denom

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            return z_vec - _apply_B(z_vec)

        precond_fn = None
        if precond_mode == "neumann":
            def precond_fn(v: torch.Tensor) -> torch.Tensor:
                out = v
                cur = v
                for _ in range(precond_terms_i):
                    cur = _apply_B(cur)
                    out = out + cur
                return out

        z, cg_info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=precond_fn,
            stabilise_every=cg_stabilise_every,
        )

    z1 = y1 - apply_axis1(z) / a_hat
    z2 = z

    vec2_z = apply_axis1(z2)  # (n,)
    Py_z2 = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps_f,
        axis=1,
        scale=z2,
        x2=x_norm2,
        y2=y_norm2,
        allow_tf32=allow_tf32,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=use_exp2,
    )
    RTz = 2.0 * (
        x * (a_hat * z1)[:, None] - Py * z1[:, None] + x * vec2_z[:, None] - Py_z2
    )

    Mat1 = 2.0 * a_hat[:, None] * A
    Mat2 = (-4.0 / eps_f) * x * (vec1 * a_hat)[:, None]
    Mat3 = (4.0 / eps_f) * Py * vec1[:, None]
    vec2 = torch.sum(Py * A, dim=1)
    Mat4 = (4.0 / eps_f) * x * vec2[:, None]

    Mat5 = mat5_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps_f,
        x2=x_norm2,
        y2=y_norm2,
        allow_tf32=allow_tf32,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=use_exp2,
    )

    EA = Mat1 + Mat2 + Mat3 + Mat4 + Mat5
    hvp = RTz / eps_f + EA
    return hvp, HvpInfo(
        cg_converged=bool(cg_info.cg_converged),
        cg_iters=int(cg_info.cg_iters),
        cg_residual=float(cg_info.cg_residual),
        cg_initial_residual=float(cg_info.cg_initial_residual),
    )


def hvp_x_sqeuclid_from_potentials_taskcsr(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    taskcsr_x,
    taskcsr_y,
    buckets=None,
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 1,
    num_stages: int = 2,
    use_exp2: bool = True,
) -> Tuple[torch.Tensor, HvpInfo]:
    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    if not x.is_cuda or not y.is_cuda or not f_hat.is_cuda or not g_hat.is_cuda:
        raise ValueError("hvp_x_sqeuclid_from_potentials_taskcsr requires CUDA tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")
    if d not in (1, 2, 3):
        raise ValueError("Blocksparse HVP only supports d in {1,2,3}.")

    eps_f = float(eps)

    def apply_axis1(vec_m: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_sqeuclid_taskcsr(
            x,
            y,
            f_hat,
            g_hat,
            vec_m,
            eps=eps_f,
            axis=1,
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            taskcsr_x=taskcsr_x,
            taskcsr_y=taskcsr_y,
            buckets=buckets,
            block_m=int(block_m),
            block_n=int(block_n),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_exp2=bool(use_exp2),
        )

    def apply_axis0(vec_n: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_sqeuclid_taskcsr(
            x,
            y,
            f_hat,
            g_hat,
            vec_n,
            eps=eps_f,
            axis=0,
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            taskcsr_x=taskcsr_x,
            taskcsr_y=taskcsr_y,
            buckets=buckets,
            block_m=int(block_m),
            block_n=int(block_n),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_exp2=bool(use_exp2),
        )

    ones_m = torch.ones((m,), device=x.device, dtype=torch.float32)
    ones_n = torch.ones((n,), device=x.device, dtype=torch.float32)
    a_hat = apply_axis1(ones_m).clamp_min(1e-12)
    b_hat = apply_axis0(ones_n).clamp_min(1e-12)

    vec1 = torch.sum(x * A, dim=1)
    Py = apply_plan_mat_sqeuclid_taskcsr(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps_f,
        axis=1,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        buckets=buckets,
        block_m=int(block_m),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        use_exp2=bool(use_exp2),
    )
    x1 = 2.0 * (a_hat * vec1 - torch.sum(A * Py, dim=1))

    PT_A = apply_plan_mat_sqeuclid_taskcsr(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps_f,
        axis=0,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        buckets=buckets,
        block_m=int(block_m),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        use_exp2=bool(use_exp2),
    )
    x2 = 2.0 * (apply_axis0(vec1) - torch.sum(y * PT_A, dim=1))

    y1 = x1 / a_hat
    y2_raw = -apply_axis0(y1) + x2
    denom = b_hat + eps_f * float(tau2)

    def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
        piz = apply_axis1(z_vec)
        piT_over_a_piz = apply_axis0(piz / a_hat)
        return piT_over_a_piz / denom

    precond_mode = str(preconditioner).lower()
    if not use_preconditioner:
        precond_mode = "none"
    if precond_mode not in ("none", "neumann", "jacobi"):
        raise ValueError("preconditioner must be one of {'none','neumann','jacobi'}.")
    precond_terms_i = int(precond_terms)
    if precond_terms_i < 0:
        raise ValueError("precond_terms must be >= 0.")

    if precond_mode == "none":
        rhs = y2_raw

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            piz = apply_axis1(z_vec)
            piT_over_a_piz = apply_axis0(piz / a_hat)
            return denom * z_vec - piT_over_a_piz

        z, cg_info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=None,
            stabilise_every=cg_stabilise_every,
        )
    else:
        rhs = y2_raw / denom

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            return z_vec - _apply_B(z_vec)

        precond_fn = None
        if precond_mode == "neumann":
            def precond_fn(v: torch.Tensor) -> torch.Tensor:
                out = v
                cur = v
                for _ in range(precond_terms_i):
                    cur = _apply_B(cur)
                    out = out + cur
                return out

        z, cg_info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=precond_fn,
            stabilise_every=cg_stabilise_every,
        )

    z1 = y1 - apply_axis1(z) / a_hat
    z2 = z

    vec2_z = apply_axis1(z2)
    Py_z2 = apply_plan_mat_sqeuclid_taskcsr(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps_f,
        axis=1,
        scale=z2,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        buckets=buckets,
        block_m=int(block_m),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        use_exp2=bool(use_exp2),
    )
    RTz = 2.0 * (
        x * (a_hat * z1)[:, None] - Py * z1[:, None] + x * vec2_z[:, None] - Py_z2
    )

    Mat1 = 2.0 * a_hat[:, None] * A
    Mat2 = (-4.0 / eps_f) * x * (vec1 * a_hat)[:, None]
    Mat3 = (4.0 / eps_f) * Py * vec1[:, None]
    vec2 = torch.sum(Py * A, dim=1)
    Mat4 = (4.0 / eps_f) * x * vec2[:, None]
    Mat5 = mat5_sqeuclid_taskcsr(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps_f,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        buckets=buckets,
        block_m=int(block_m),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        use_exp2=bool(use_exp2),
    )

    EA = Mat1 + Mat2 + Mat3 + Mat4 + Mat5
    hvp = RTz / eps_f + EA
    return hvp, HvpInfo(
        cg_converged=bool(cg_info.cg_converged),
        cg_iters=int(cg_info.cg_iters),
        cg_residual=float(cg_info.cg_residual),
        cg_initial_residual=float(cg_info.cg_initial_residual),
    )


def hvp_x_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A: torch.Tensor,
    *,
    eps_list: Sequence[float],
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, HvpInfo]:
    """End-to-end HVP: solve prelast potentials then compute H@A (w.r.t x)."""

    eps = float(eps_list[-1])
    f_grad, g_grad = sinkhorn_prelast_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        eps_list=eps_list,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)

    return hvp_x_sqeuclid_from_potentials(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        tau2=tau2,
        max_cg_iter=max_cg_iter,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_stabilise_every=cg_stabilise_every,
        preconditioner=preconditioner,
        precond_terms=precond_terms,
        use_preconditioner=use_preconditioner,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=int(num_warps or 4),
        num_stages=num_stages,
    )


def hvp_x_sqeuclid_multiscale(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A: torch.Tensor,
    *,
    eps_list: Sequence[float],
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    truncate: float = 5.0,
    cluster_scale: Optional[float] = None,
    max_coarse_levels: int = 1,
    multiscale_blocksparse_backend: str = "taskcsr_bucketed",
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, HvpInfo]:
    """Multiscale (D<=3) HVP using the multiscale truncated plan structure.

    This uses the multiscale solver to build a blocksparse neighborhood pattern,
    then runs the OTT-Hessian-style HVP with blocksparse plan applications.
    """

    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    if not x.is_cuda or not y.is_cuda or not A.is_cuda:
        raise ValueError("hvp_x_sqeuclid_multiscale requires CUDA tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")
    if d not in (1, 2, 3):
        raise ValueError("Multiscale HVP is only supported for d in {1,2,3}.")

    eps = float(eps_list[-1])

    _, _, f_grad, g_grad, state = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        eps_list=list(map(float, eps_list)),
        truncate=float(truncate),
        cluster_scale=cluster_scale,
        max_coarse_levels=int(max_coarse_levels),
        blocksparse_backend=str(multiscale_blocksparse_backend),
        allow_tf32=bool(allow_tf32),
        use_exp2=bool(use_exp2),
        autotune=bool(autotune),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=int(num_stages),
        return_prelast=True,
        return_state=True,
    )
    (
        perm_x,
        perm_y,
        offsets_x,
        offsets_y,
        row_ptr_x,
        col_idx_x,
        row_ptr_y,
        col_idx_y,
    ) = state

    x_s = x[perm_x].contiguous()
    y_s = y[perm_y].contiguous()
    a_s = a[perm_x].contiguous()
    b_s = b[perm_y].contiguous()
    A_s = A[perm_x].contiguous()
    f_grad_s = f_grad[perm_x].contiguous()
    g_grad_s = g_grad[perm_y].contiguous()

    f_hat, g_hat = geomloss_to_ott_potentials(f_grad_s, g_grad_s, a_s, b_s, eps=eps)

    if block_m is None:
        block_m = 32
    if block_n is None:
        block_n = 64
    if num_warps is None:
        num_warps = 1

    tasks = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        block_m=int(block_m),
        block_n=int(block_n),
    )
    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    buckets = None
    if str(multiscale_blocksparse_backend) == "taskcsr_bucketed":
        buckets = blocksparse_build_taskcsr_buckets(taskcsr_x, taskcsr_y)

    hvp_s, info = hvp_x_sqeuclid_from_potentials_taskcsr(
        x_s,
        y_s,
        f_hat,
        g_hat,
        A_s,
        eps=eps,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        buckets=buckets,
        tau2=tau2,
        max_cg_iter=max_cg_iter,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_stabilise_every=cg_stabilise_every,
        preconditioner=preconditioner,
        precond_terms=precond_terms,
        use_preconditioner=use_preconditioner,
        block_m=int(block_m),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        use_exp2=bool(use_exp2),
    )

    hvp = torch.empty_like(hvp_s)
    hvp[perm_x] = hvp_s
    return hvp, info
