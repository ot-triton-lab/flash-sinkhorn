"""Conjugate gradient solver for symmetric positive definite systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

__all__ = ["CGInfo", "conjugate_gradient"]


@dataclass(frozen=True)
class CGInfo:
    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    max_iter: int = 300,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stabilise_every: int = 0,
) -> Tuple[torch.Tensor, CGInfo]:
    """Minimal CG for symmetric positive definite systems.

    stabilise_every recomputes the true residual periodically to reduce drift.
    """

    stabilise_every_i = int(stabilise_every)
    if stabilise_every_i < 0:
        raise ValueError("stabilise_every must be >= 0.")

    x = torch.zeros_like(b)
    r = b - matvec(x)
    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = torch.dot(r, z)

    init_res = float(torch.linalg.norm(r).detach().cpu())
    tol = max(atol, rtol * init_res)

    cg_converged = False
    for it in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if denom.abs() == 0:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if stabilise_every_i and (it + 1) % stabilise_every_i == 0:
            r = b - matvec(x)

        res = float(torch.linalg.norm(r).detach().cpu())
        if res <= tol:
            cg_converged = True
            iters = it + 1
            return x, CGInfo(
                cg_converged=True,
                cg_iters=iters,
                cg_residual=res,
                cg_initial_residual=init_res,
            )

        z = preconditioner(r) if preconditioner is not None else r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    res = float(torch.linalg.norm(r).detach().cpu())
    return x, CGInfo(
        cg_converged=cg_converged,
        cg_iters=int(max_iter),
        cg_residual=res,
        cg_initial_residual=init_res,
    )
