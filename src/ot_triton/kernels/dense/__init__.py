"""Dense (non-sparse) Sinkhorn kernel implementations."""

from ot_triton.kernels.dense.ott import (
    apply_lse_kernel_sqeuclid,
    apply_transport_from_potentials_sqeuclid,
    sinkhorn_potentials_sqeuclid,
    update_potential,
)
from ot_triton.kernels.dense.geomloss import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
    _default_block_sizes,
    _geomloss_autotune_configs,
    _geomloss_symmetric_step_sqeuclid_impl,
    _geomloss_symmetric_step_sqeuclid_autotune,
    _geomloss_symmetric_step_sqeuclid,
)
from ot_triton.kernels.dense.grad import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from ot_triton.kernels.dense.apply import (
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
)

__all__ = [
    # OTT-style
    "apply_lse_kernel_sqeuclid",
    "apply_transport_from_potentials_sqeuclid",
    "sinkhorn_potentials_sqeuclid",
    "update_potential",
    # GeomLoss-style
    "sinkhorn_geomloss_online_potentials_sqeuclid",
    "_default_block_sizes",
    "_geomloss_autotune_configs",
    "_geomloss_symmetric_step_sqeuclid_impl",
    "_geomloss_symmetric_step_sqeuclid_autotune",
    "_geomloss_symmetric_step_sqeuclid",
    # Gradient
    "sinkhorn_geomloss_online_grad_sqeuclid",
    # Apply
    "apply_plan_vec_sqeuclid",
    "apply_plan_mat_sqeuclid",
    "mat5_sqeuclid",
]
