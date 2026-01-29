"""Common utilities shared across kernel modules."""

from typing import Optional, Sequence

import numpy as np
import torch


def dampening(eps: float, rho: Optional[float]) -> float:
    """Compute dampening factor for unbalanced OT with KL marginal penalties.

    For balanced OT (rho=None), returns 1.0.
    For unbalanced OT, returns 1/(1 + eps/rho), which controls the strength
    of marginal relaxation.

    Args:
        eps: Entropy regularization parameter.
        rho: Marginal constraint penalty (None for balanced OT).

    Returns:
        Dampening factor in (0, 1] range.
    """
    if rho is None:
        return 1.0
    return 1.0 / (1.0 + eps / rho)


def log_weights(w: torch.Tensor) -> torch.Tensor:
    """Convert weights to log domain, handling zeros safely."""
    w = w.float()
    out = w.log()
    out = torch.where(w > 0, out, torch.full_like(out, -100000.0))
    return out


def max_diameter(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the maximum diameter (bounding box diagonal) of point clouds x and y."""
    x_f = x.float()
    y_f = y.float()
    mins = torch.stack((x_f.min(dim=0).values, y_f.min(dim=0).values)).min(dim=0).values
    maxs = torch.stack((x_f.max(dim=0).values, y_f.max(dim=0).values)).max(dim=0).values
    return (maxs - mins).norm().item()


def epsilon_schedule(
    diameter: float, blur: float, scaling: float, p: float = 2.0
) -> Sequence[float]:
    """Generate exponential epsilon decay schedule for Sinkhorn iterations.

    Matches GeomLoss's epsilon_schedule exactly: exponential cooling from
    diameter^p down to blur^p with decay factor scaling^p per iteration.

    Args:
        diameter: Maximum diameter of the point clouds.
        blur: Target blur (the regularization "blur" parameter from GeomLoss).
        scaling: Decay factor per iteration (0 < scaling < 1).
        p: Cost exponent (default 2 for squared Euclidean).

    Returns:
        Sequence of epsilon values from diameter^p down to blur^p.
    """
    if diameter <= 0:
        raise ValueError("diameter must be > 0.")
    if blur <= 0:
        raise ValueError("blur must be > 0.")
    if not (0.0 < scaling < 1.0):
        raise ValueError("scaling must be in (0, 1).")

    # Match GeomLoss: final eps = blur^p
    final_eps = blur ** p
    log_final = p * np.log(blur)

    eps_list = [diameter**p]
    eps_list += [
        float(np.exp(e))
        for e in np.arange(p * np.log(diameter), log_final, p * np.log(scaling))
    ]
    eps_list += [final_eps]
    return eps_list
