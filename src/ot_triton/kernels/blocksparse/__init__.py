"""BlockSparse Sinkhorn kernel implementations."""

# Re-export from original files for now - will be refactored into submodules later
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_sqeuclid import (
    blocksparse_prepare_metadata,
    geomloss_blocksparse_symmetric_step_sqeuclid,
    geomloss_blocksparse_grad_sqeuclid,
    _pid_maps_from_offsets,
    _geomloss_symmetric_step_sqeuclid_blocksparse_impl,
    _geomloss_grad_sqeuclid_blocksparse_impl,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    BlockSparseTasks,
    blocksparse_build_tasks_from_csr,
    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic,
    _lse_max_sqeuclid_tasks_atomic_impl,
    _lse_sumexp_sqeuclid_tasks_atomic_impl,
    _lse_finalize_from_ml_impl,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    BlockSparseTaskCSR,
    BlockSparseTaskCSRBuckets,
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr,
    geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed,
    _empty_taskcsr,
    _taskcsr_select,
)
from ot_triton.kernels.sinkhorn_triton_apply_blocksparse_sqeuclid import (
    apply_plan_vec_sqeuclid_taskcsr,
    apply_plan_mat_sqeuclid_taskcsr,
    mat5_sqeuclid_taskcsr,
)

__all__ = [
    # Metadata
    "blocksparse_prepare_metadata",
    "_pid_maps_from_offsets",
    # Basic blocksparse
    "geomloss_blocksparse_symmetric_step_sqeuclid",
    "geomloss_blocksparse_grad_sqeuclid",
    "_geomloss_symmetric_step_sqeuclid_blocksparse_impl",
    "_geomloss_grad_sqeuclid_blocksparse_impl",
    # Ranges/Tasks
    "BlockSparseTasks",
    "blocksparse_build_tasks_from_csr",
    "geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic",
    "_lse_max_sqeuclid_tasks_atomic_impl",
    "_lse_sumexp_sqeuclid_tasks_atomic_impl",
    "_lse_finalize_from_ml_impl",
    # TaskCSR
    "BlockSparseTaskCSR",
    "BlockSparseTaskCSRBuckets",
    "blocksparse_build_taskcsr",
    "blocksparse_build_taskcsr_buckets",
    "geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr",
    "geomloss_blocksparse_symmetric_step_sqeuclid_taskcsr_bucketed",
    "_empty_taskcsr",
    "_taskcsr_select",
    # Apply
    "apply_plan_vec_sqeuclid_taskcsr",
    "apply_plan_mat_sqeuclid_taskcsr",
    "mat5_sqeuclid_taskcsr",
]
