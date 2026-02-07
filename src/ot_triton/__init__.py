"""FlashSinkhorn: Streaming Entropic Optimal Transport in PyTorch + Triton.

FlashSinkhorn uses FlashAttention-style streaming to compute Sinkhorn OT
without materializing the n×m cost matrix, enabling O(nd) memory usage.

Package name: ot_triton
"""

from ot_triton.samples_loss import SamplesLoss
from ot_triton.cg import CGInfo, conjugate_gradient
from ot_triton.hvp import (
    HvpInfo,
    geomloss_to_ott_potentials,
    hvp_x_sqeuclid,
    hvp_x_sqeuclid_from_potentials,
    inverse_hvp_x_sqeuclid_from_potentials,
)
from ot_triton.implicit_grad import (
    ImplicitGradInfo,
    implicit_grad_x,
    implicit_grad_x_from_potentials,
)

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
    "__version__",
]
__version__ = "0.3.0"
