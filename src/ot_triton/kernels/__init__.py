"""Triton kernels and wrappers."""

# Common utilities
from ot_triton.kernels._common import (
    epsilon_schedule,
    log_weights,
    max_diameter,
)

# OTT-style primitives (LSE kernels kept, solver delegates to FlashSinkhorn)
from ot_triton.kernels.sinkhorn_triton_ott_sqeuclid import (
    apply_lse_kernel_sqeuclid,
    apply_transport_from_potentials_sqeuclid,
    sinkhorn_potentials_sqeuclid,
    update_potential,
)

# GeomLoss-style (solver delegates to FlashSinkhorn)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)

# Gradient kernels
from ot_triton.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)

# Apply kernels (transport plan application)
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
    # FlashSinkhorn variants (shifted potentials)
    apply_plan_vec_flashstyle,
    apply_plan_mat_flashstyle,
)

# FlashSinkhorn (shifted potential formulation)
from ot_triton.kernels.sinkhorn_flashstyle_sqeuclid import (
    precompute_flashsinkhorn_inputs,
    compute_bias_f,
    compute_bias_g,
    flashsinkhorn_lse,
    flashsinkhorn_symmetric_step,
    shifted_to_standard_potentials,
    standard_to_shifted_potentials,
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)

__all__ = [
    # Common
    "epsilon_schedule",
    "log_weights",
    "max_diameter",
    # OTT-style
    "apply_lse_kernel_sqeuclid",
    "apply_transport_from_potentials_sqeuclid",
    "sinkhorn_potentials_sqeuclid",
    "update_potential",
    # GeomLoss-style
    "sinkhorn_geomloss_online_potentials_sqeuclid",
    # Gradient
    "sinkhorn_geomloss_online_grad_sqeuclid",
    # Apply
    "apply_plan_vec_sqeuclid",
    "apply_plan_mat_sqeuclid",
    "mat5_sqeuclid",
    "apply_plan_vec_flashstyle",
    "apply_plan_mat_flashstyle",
    # FlashSinkhorn
    "precompute_flashsinkhorn_inputs",
    "compute_bias_f",
    "compute_bias_g",
    "flashsinkhorn_lse",
    "flashsinkhorn_symmetric_step",
    "shifted_to_standard_potentials",
    "standard_to_shifted_potentials",
    "sinkhorn_flashstyle_alternating",
    "sinkhorn_flashstyle_symmetric",
]
