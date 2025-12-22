import pytest
import torch

from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials_taskcsr
from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.kernels.sinkhorn_triton_apply_blocksparse_sqeuclid import (
    apply_plan_mat_sqeuclid_taskcsr,
    apply_plan_vec_sqeuclid_taskcsr,
    mat5_sqeuclid_taskcsr,
)
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_mat_sqeuclid,
    apply_plan_vec_sqeuclid,
    mat5_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
)


def _dense_one_cluster_state(n: int, m: int, device: torch.device):
    offsets_x = torch.tensor([0, n], device=device, dtype=torch.int32)
    offsets_y = torch.tensor([0, m], device=device, dtype=torch.int32)
    row_ptr_x = torch.tensor([0, 1], device=device, dtype=torch.int32)
    col_idx_x = torch.tensor([0], device=device, dtype=torch.int32)
    row_ptr_y = torch.tensor([0, 1], device=device, dtype=torch.int32)
    col_idx_y = torch.tensor([0], device=device, dtype=torch.int32)
    return offsets_x, offsets_y, row_ptr_x, col_idx_x, row_ptr_y, col_idx_y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_blocksparse_apply_matches_dense_for_full_pattern():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 64, 80, 3
    eps = 0.7
    block_m, block_n, block_k = 32, 64, 16

    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    f = torch.randn(n, device=device, dtype=torch.float32)
    g = torch.randn(m, device=device, dtype=torch.float32)

    vec_m = torch.randn(m, device=device, dtype=torch.float32)
    vec_n = torch.randn(n, device=device, dtype=torch.float32)

    offsets_x, offsets_y, row_ptr_x, col_idx_x, row_ptr_y, col_idx_y = _dense_one_cluster_state(
        n, m, device
    )
    tasks = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        block_m=int(block_m),
        block_n=int(block_n),
    )
    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    out_dense_1 = apply_plan_vec_sqeuclid(
        x,
        y,
        f,
        g,
        vec_m,
        eps=eps,
        axis=1,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    out_sparse_1 = apply_plan_vec_sqeuclid_taskcsr(
        x,
        y,
        f,
        g,
        vec_m,
        eps=eps,
        axis=1,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )
    torch.testing.assert_close(out_sparse_1, out_dense_1, rtol=2e-4, atol=2e-4)

    out_dense_0 = apply_plan_vec_sqeuclid(
        x,
        y,
        f,
        g,
        vec_n,
        eps=eps,
        axis=0,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    out_sparse_0 = apply_plan_vec_sqeuclid_taskcsr(
        x,
        y,
        f,
        g,
        vec_n,
        eps=eps,
        axis=0,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )
    torch.testing.assert_close(out_sparse_0, out_dense_0, rtol=2e-4, atol=2e-4)

    Py_dense = apply_plan_mat_sqeuclid(
        x,
        y,
        f,
        g,
        y,
        eps=eps,
        axis=1,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    Py_sparse = apply_plan_mat_sqeuclid_taskcsr(
        x,
        y,
        f,
        g,
        y,
        eps=eps,
        axis=1,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )
    torch.testing.assert_close(Py_sparse, Py_dense, rtol=2e-4, atol=2e-4)

    A = torch.randn(n, d, device=device, dtype=torch.float32)
    PT_A_dense = apply_plan_mat_sqeuclid(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        axis=0,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    PT_A_sparse = apply_plan_mat_sqeuclid_taskcsr(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        axis=0,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )
    torch.testing.assert_close(PT_A_sparse, PT_A_dense, rtol=2e-4, atol=2e-4)

    Mat5_dense = mat5_sqeuclid(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    Mat5_sparse = mat5_sqeuclid_taskcsr(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )
    torch.testing.assert_close(Mat5_sparse, Mat5_dense, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_blocksparse_hvp_matches_dense_for_full_pattern():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 64, 80, 3
    eps = 0.7
    block_m, block_n, block_k = 32, 64, 16

    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    eps_list = [eps] * 20
    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        eps_list=eps_list,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
        return_prelast=True,
    )
    f, g = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)
    A = torch.randn(n, d, device=device, dtype=torch.float32)

    offsets_x, offsets_y, row_ptr_x, col_idx_x, row_ptr_y, col_idx_y = _dense_one_cluster_state(
        n, m, device
    )
    tasks = blocksparse_build_tasks_from_csr(
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        block_m=int(block_m),
        block_n=int(block_n),
    )
    taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
    taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

    hvp_dense, info_dense = hvp_x_sqeuclid_from_potentials(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        tau2=1e-2,
        max_cg_iter=80,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=True,
        allow_tf32=False,
        use_exp2=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=1,
        num_stages=2,
    )
    hvp_sparse, info_sparse = hvp_x_sqeuclid_from_potentials_taskcsr(
        x,
        y,
        f,
        g,
        A,
        eps=eps,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        taskcsr_x=taskcsr_x,
        taskcsr_y=taskcsr_y,
        buckets=None,
        tau2=1e-2,
        max_cg_iter=80,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=True,
        block_m=block_m,
        block_n=block_n,
        num_warps=1,
        num_stages=2,
        use_exp2=False,
    )

    assert info_dense.cg_converged == info_sparse.cg_converged
    torch.testing.assert_close(hvp_sparse, hvp_dense, rtol=5e-4, atol=5e-4)
