"""Coordinate preprocessing the multiscale solver applies before any kernel.

Two steps, in this order:

1. :func:`center_coords` subtracts the mass-weighted joint centroid. Squared
   Euclidean OT is translation invariant, so the problem is unchanged while
   ``||x||^2`` shrinks. The shifted-potential kernels subtract
   ``alpha = ||x||^2`` explicitly, and once alpha's fp32 spacing approaches
   ``eps`` the potentials lose their precision.
2. :func:`truncate_tf32` rounds the centered coordinates to TF32. Tensor cores
   round the inputs of every dot product to TF32 while the norms are computed
   in fp32; rounding once makes both describe the same point cloud, and the
   solver works on the rounded cloud throughout. Centering first
   matters: subtracting a centroid produces fresh unrounded values.
"""

from __future__ import annotations

import torch

#: Low mantissa bits dropped from fp32 (23) to reach TF32 (10).
TF32_DROPPED_BITS = 13
_HALF_ULP = (1 << (TF32_DROPPED_BITS - 1)) - 1   # 0x0FFF
_KEEP_MASK = -(1 << TF32_DROPPED_BITS)           # -0x2000


def round_to_tf32(t: torch.Tensor) -> torch.Tensor:
    """Round fp32/fp64 values to TF32 precision, round-to-nearest-even.

    fp16 and bf16 already carry at most 10 mantissa bits and are returned
    unchanged; fp64 keeps its dtype, since TF32 values are exact in fp64.
    """
    if t.dtype not in (torch.float32, torch.float64):
        return t
    u = t.detach().float().contiguous().view(torch.int32)
    r = ((u + _HALF_ULP + ((u >> TF32_DROPPED_BITS) & 1)) & _KEEP_MASK).view(torch.float32)
    return r.to(t.dtype)


def truncate_tf32(x: torch.Tensor, y: torch.Tensor):
    """Round both clouds to TF32; run after :func:`center_coords`."""
    return round_to_tf32(x), round_to_tf32(y)


def center_coords(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Subtract the mass-weighted joint centroid ``(sum a_i x_i + sum b_j y_j) / 2``."""
    with torch.no_grad():
        mu = 0.5 * ((a.float().unsqueeze(-1) * x.float()).sum(dim=0)
                    + (b.float().unsqueeze(-1) * y.float()).sum(dim=0))
    return x - mu.to(x.dtype), y - mu.to(y.dtype)


def preprocess_coordinates(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Centered, TF32-rounded fp32 coordinates: the point clouds the multiscale solver solves on.

    Centering and rounding happen in the input dtype, then the result is cast to fp32
    (exact, since TF32 values are representable in fp32).
    """
    x, y = center_coords(x.detach(), y.detach(), a, b)
    x, y = truncate_tf32(x, y)
    return x.float().contiguous(), y.float().contiguous()
