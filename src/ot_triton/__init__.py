"""OT Triton: Sinkhorn OT kernels in PyTorch + Triton."""

from ot_triton.samples_loss import SamplesLoss
from ot_triton.cg import CGInfo, conjugate_gradient
from ot_triton.hvp import (
    HvpInfo,
    geomloss_to_ott_potentials,
    hvp_x_sqeuclid,
    hvp_x_sqeuclid_from_potentials,
    hvp_x_sqeuclid_multiscale,
)

__all__ = [
    "SamplesLoss",
    "CGInfo",
    "conjugate_gradient",
    "HvpInfo",
    "geomloss_to_ott_potentials",
    "hvp_x_sqeuclid",
    "hvp_x_sqeuclid_from_potentials",
    "hvp_x_sqeuclid_multiscale",
    "__version__",
]
__version__ = "0.1.0"
