"""FlashSinkhorn: Streaming Entropic Optimal Transport in PyTorch + Triton.

FlashSinkhorn uses FlashAttention-style streaming to compute Sinkhorn OT
without materializing the n×m cost matrix, enabling O(nd) memory usage; the
multiscale backend also stores its block masks.

Package name: flash_sinkhorn
"""

from .samples_loss import SamplesLoss
from .cg import CGInfo, conjugate_gradient
from .hvp import (
    HvpInfo,
    geomloss_to_ott_potentials,
    hvp_x_sqeuclid,
    hvp_x_sqeuclid_from_potentials,
    inverse_hvp_x_sqeuclid_from_potentials,
)
from .implicit_grad import (
    ImplicitGradInfo,
    implicit_grad_x,
    implicit_grad_x_from_potentials,
)
from .c_transform import c_transform_fwd, c_transform_cost

__all__ = [
    "SamplesLoss",
    "CGInfo",
    "conjugate_gradient",
    "HvpInfo",
    "geomloss_to_ott_potentials",
    "hvp_x_sqeuclid",
    "hvp_x_sqeuclid_from_potentials",
    "inverse_hvp_x_sqeuclid_from_potentials",
    "ImplicitGradInfo",
    "implicit_grad_x",
    "implicit_grad_x_from_potentials",
    "c_transform_fwd",
    "c_transform_cost",
    "__version__",
]
__version__ = "0.4.1"
