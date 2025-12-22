import pytest
import torch

from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_sqeuclid import (
    geomloss_blocksparse_symmetric_step_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import log_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_blocksparse_ranges_atomic_matches_padded_kernel():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 128, 96, 2
    eps_list = [0.5, 0.5, 0.5, 0.5]
    truncate = 5.0
    cluster_scale = 2.0

    x = torch.randn(n, d, device=device, dtype=torch.float32) * 10.0
    y = torch.randn(m, d, device=device, dtype=torch.float32) * 10.0
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    # Build a fine-level blocksparse state via the multiscale solver.
    _, _, _, _, state = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        eps_list=eps_list,
        truncate=truncate,
        cluster_scale=cluster_scale,
        max_coarse_levels=1,
        allow_tf32=False,
        use_exp2=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        return_prelast=True,
        return_state=True,
    )
    perm_x, perm_y, offsets_x, offsets_y, row_ptr_x, col_idx_x, row_ptr_y, col_idx_y = state

    x_s = x[perm_x].contiguous()
    y_s = y[perm_y].contiguous()
    a_s = a[perm_x].contiguous()
    b_s = b[perm_y].contiguous()
    loga_s = log_weights(a_s).contiguous()
    logb_s = log_weights(b_s).contiguous()
    x2_s = (x_s * x_s).sum(dim=1).contiguous()
    y2_s = (y_s * y_s).sum(dim=1).contiguous()

    f0 = torch.randn((n,), device=device, dtype=torch.float32)
    g0 = torch.randn((m,), device=device, dtype=torch.float32)
    f1 = torch.empty_like(f0)
    g1 = torch.empty_like(g0)
    f1_a = torch.empty_like(f0)
    g1_a = torch.empty_like(g0)

    geomloss_blocksparse_symmetric_step_sqeuclid(
        x_s,
        y_s,
        f0,
        g0,
        loga_s,
        logb_s,
        x2_s,
        y2_s,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        row_ptr_y=row_ptr_y,
        col_idx_y=col_idx_y,
        f_out=f1,
        g_out=g1,
        eps=eps_list[-1],
        alpha=0.5,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=2,
        allow_tf32=False,
        use_exp2=False,
    )

    geomloss_blocksparse_symmetric_step_sqeuclid_ranges_atomic(
        x_s,
        y_s,
        f0,
        g0,
        loga_s,
        logb_s,
        x2_s,
        y2_s,
        offsets_x=offsets_x,
        offsets_y=offsets_y,
        row_ptr_x=row_ptr_x,
        col_idx_x=col_idx_x,
        row_ptr_y=row_ptr_y,
        col_idx_y=col_idx_y,
        f_out=f1_a,
        g_out=g1_a,
        eps=eps_list[-1],
        alpha=0.5,
        block_m=64,
        block_n=64,
        block_k=32,
        allow_tf32=False,
        use_exp2=False,
    )

    torch.testing.assert_close(f1_a, f1, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(g1_a, g1, rtol=5e-3, atol=5e-3)

