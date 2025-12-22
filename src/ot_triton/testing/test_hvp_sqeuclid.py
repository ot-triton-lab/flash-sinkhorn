import pytest
import torch

from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import apply_plan_vec_sqeuclid
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import apply_plan_mat_sqeuclid
from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import mat5_sqeuclid
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.testing.reference_hvp import plan_from_potentials
from ot_triton.testing.reference_hvp import hvp_x_dense_sqeuclid


def _rand_inputs(n: int, m: int, d: int, device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    return x, y, a, b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_hvp_x_matches_dense_reference_small():
    device = torch.device("cuda")

    n, m, d = 32, 24, 8
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.2
    eps_list = [eps] * 4

    # Get prelast potentials (GeomLoss convention), then convert to OTT convention.
    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=False,
        use_exp2=False,
        eps_list=eps_list,
        autotune=False,
        return_prelast=True,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)

    A = torch.randn(n, d, device=device, dtype=torch.float32)

    hvp_ref, info_ref = hvp_x_dense_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        tau2=1e-5,
        max_cg_iter=80,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=True,
    )
    hvp, info = hvp_x_sqeuclid_from_potentials(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        tau2=1e-5,
        max_cg_iter=80,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        preconditioner="neumann",
        precond_terms=3,
        use_preconditioner=True,
        allow_tf32=False,
        use_exp2=False,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        num_stages=2,
    )

    assert info_ref["cg_converged"] in (0.0, 1.0)
    assert isinstance(info.cg_converged, bool)

    # Same algorithm, different transport kernels -> allow small numerical drift.
    torch.testing.assert_close(hvp, hvp_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_apply_plan_vec_matches_dense_reference_small():
    device = torch.device("cuda")

    n, m, d = 32, 24, 8
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.2
    eps_list = [eps] * 4

    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=False,
        use_exp2=False,
        eps_list=eps_list,
        autotune=False,
        return_prelast=True,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)
    P = plan_from_potentials(x, y, f_hat, g_hat, eps=eps)

    vec_m = torch.randn(m, device=device, dtype=torch.float32)
    out_axis1 = apply_plan_vec_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        vec_m,
        eps=eps,
        axis=1,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(out_axis1, P @ vec_m, rtol=2e-4, atol=2e-4)

    vec_n = torch.randn(n, device=device, dtype=torch.float32)
    out_axis0 = apply_plan_vec_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        vec_n,
        eps=eps,
        axis=0,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(out_axis0, P.t() @ vec_n, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_mat5_matches_dense_reference_small():
    device = torch.device("cuda")

    n, m, d = 32, 24, 8
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.2
    eps_list = [eps] * 4

    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=False,
        use_exp2=False,
        eps_list=eps_list,
        autotune=False,
        return_prelast=True,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)
    P = plan_from_potentials(x, y, f_hat, g_hat, eps=eps)

    A = torch.randn(n, d, device=device, dtype=torch.float32)

    dot_Ay = A @ y.t()
    mat5_ref = (-4.0 / eps) * ((P * dot_Ay) @ y)
    mat5 = mat5_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(mat5, mat5_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_apply_plan_mat_matches_dense_reference_small():
    device = torch.device("cuda")

    n, m, d = 32, 24, 8
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.2
    eps_list = [eps] * 4

    _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=False,
        use_exp2=False,
        eps_list=eps_list,
        autotune=False,
        return_prelast=True,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)
    P = plan_from_potentials(x, y, f_hat, g_hat, eps=eps)

    Py = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps,
        axis=1,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(Py, P @ y, rtol=2e-4, atol=2e-4)

    A = torch.randn(n, d, device=device, dtype=torch.float32)
    PT_A = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        axis=0,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(PT_A, P.t() @ A, rtol=2e-4, atol=2e-4)

    z2 = torch.randn(m, device=device, dtype=torch.float32)
    Py_z2 = apply_plan_mat_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        y,
        eps=eps,
        axis=1,
        scale=z2,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=4,
        use_exp2=False,
        allow_tf32=False,
    )
    torch.testing.assert_close(Py_z2, P @ (y * z2[:, None]), rtol=2e-4, atol=2e-4)
