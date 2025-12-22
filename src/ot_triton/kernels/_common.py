"""Common utilities shared across kernel modules."""

from typing import Sequence

import numpy as np
import torch


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

    Args:
        diameter: Maximum diameter of the point clouds.
        blur: Target blur (final epsilon^(1/p)).
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

    eps_list = [diameter**p]
    eps_list += [
        float(np.exp(e))
        for e in np.arange(p * np.log(diameter), p * np.log(blur), p * np.log(scaling))
    ]
    eps_list += [blur**p]
    return eps_list
