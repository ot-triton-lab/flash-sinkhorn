"""Multiscale hierarchical Sinkhorn solver."""

# Re-export from original file for now
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)

__all__ = [
    "sinkhorn_geomloss_multiscale_potentials_sqeuclid",
]
