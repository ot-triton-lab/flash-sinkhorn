"""Triton kernels and wrappers."""

# Common utilities
from ot_triton.kernels._common import (
    epsilon_schedule,
    log_weights,
    max_diameter,
)

# Dense kernels (new organized structure)
from ot_triton.kernels.dense import (
    # OTT-style
    apply_lse_kernel_sqeuclid,
    apply_transport_from_potentials_sqeuclid,
    sinkhorn_potentials_sqeuclid,
    update_potential,
    # GeomLoss-style
    sinkhorn_geomloss_online_potentials_sqeuclid,
    # Gradient
    sinkhorn_geomloss_online_grad_sqeuclid,
    # Apply
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
)

# BlockSparse kernels
from ot_triton.kernels.blocksparse import (
    # Metadata
    blocksparse_prepare_metadata,
    # Basic blocksparse
    geomloss_blocksparse_symmetric_step_sqeuclid,
    geomloss_blocksparse_grad_sqeuclid,
    # Ranges/Tasks
    BlockSparseTasks,
    blocksparse_build_tasks_from_csr,
    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic,
    # TaskCSR
    BlockSparseTaskCSR,
    BlockSparseTaskCSRBuckets,
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed,
    # Apply
    apply_plan_vec_sqeuclid_taskcsr,
    apply_plan_mat_sqeuclid_taskcsr,
    mat5_sqeuclid_taskcsr,
)

# Multiscale solver
from ot_triton.kernels.multiscale import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)

__all__ = [
    # Common
    "epsilon_schedule",
    "log_weights",
    "max_diameter",
    # Dense OTT-style
    "apply_lse_kernel_sqeuclid",
    "apply_transport_from_potentials_sqeuclid",
    "sinkhorn_potentials_sqeuclid",
    "update_potential",
    # Dense GeomLoss-style
    "sinkhorn_geomloss_online_potentials_sqeuclid",
    # Dense Gradient
    "sinkhorn_geomloss_online_grad_sqeuclid",
    # Dense Apply
    "apply_plan_vec_sqeuclid",
    "apply_plan_mat_sqeuclid",
    "mat5_sqeuclid",
    # BlockSparse
    "blocksparse_prepare_metadata",
    "geomloss_blocksparse_symmetric_step_sqeuclid",
    "geomloss_blocksparse_grad_sqeuclid",
    "BlockSparseTasks",
    "blocksparse_build_tasks_from_csr",
    "geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic",
    "BlockSparseTaskCSR",
    "BlockSparseTaskCSRBuckets",
    "blocksparse_build_taskcsr",
    "blocksparse_build_taskcsr_buckets",
    "geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr",
    "geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed",
    "apply_plan_vec_sqeuclid_taskcsr",
    "apply_plan_mat_sqeuclid_taskcsr",
    "mat5_sqeuclid_taskcsr",
    # Multiscale
    "sinkhorn_geomloss_multiscale_potentials_sqeuclid",
]
