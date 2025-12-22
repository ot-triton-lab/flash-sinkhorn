import pytest
import torch

from ot_triton import SamplesLoss
from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.hvp import hvp_x_sqeuclid_multiscale
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.testing.reference_sinkhorn import sinkhorn_geomloss_barycentric_grads_ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_unweighted_matches_weighted_uniform():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 128, 96, 32
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)

    a = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)

    loss = SamplesLoss(blur=0.1, scaling=0.5, debias=False, potentials=False)

    out_unweighted = loss(x, y)
    out_weighted = loss(a, x, b, y)

    torch.testing.assert_close(out_unweighted, out_weighted, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_potentials_and_cost_consistency():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 128, 96, 32
    x = torch.randn(n, d, device=device, dtype=torch.float16)
    y = torch.randn(m, d, device=device, dtype=torch.float16)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    loss_cost = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=False,
        use_epsilon_scaling=False,
        eps=0.1,
        n_iters=2,
        allow_tf32=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )
    loss_pots = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=True,
        use_epsilon_scaling=False,
        eps=0.1,
        n_iters=2,
        allow_tf32=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    cost = loss_cost(a, x, b, y)
    f, g = loss_pots(a, x, b, y)
    cost_from_pots = (a * f).sum() + (b * g).sum()

    torch.testing.assert_close(cost, cost_from_pots, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_matches_kernel_wrapper_for_potentials():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 64, 48, 32
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    loss = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=True,
        use_epsilon_scaling=False,
        eps=0.1,
        n_iters=2,
        allow_tf32=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    f_api, g_api = loss(a, x, b, y)

    f_k, g_k = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        blur=0.1,
        scaling=0.5,
        use_epsilon_scaling=False,
        eps=0.1,
        n_iters=2,
        allow_tf32=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    torch.testing.assert_close(f_api, f_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_api, g_k, rtol=1e-5, atol=1e-5)


def test_samplesloss_debias_not_implemented():
    with pytest.raises(NotImplementedError):
        SamplesLoss(debias=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_accepts_weight_singleton_dim_and_preserves_shape():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 32, 24, 16
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = (torch.rand(n, device=device, dtype=torch.float32) + 0.1)[:, None]
    b = (torch.rand(m, device=device, dtype=torch.float32) + 0.1)[:, None]
    a = a / a.sum()
    b = b / b.sum()

    loss = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=True,
        use_epsilon_scaling=False,
        eps=0.1,
        n_iters=2,
        allow_tf32=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    f, g = loss(a, x, b, y)
    assert f.shape == (n, 1)
    assert g.shape == (m, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_backward_matches_explicit_plan_grads():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 48, 40, 16
    eps = 0.2
    n_iters = 3

    x = torch.randn(n, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(m, d, device=device, dtype=torch.float32, requires_grad=True)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    loss_cost = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=False,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    val = loss_cost(a, x, b, y)
    grad_x, grad_y = torch.autograd.grad(val, (x, y))

    with torch.no_grad():
        _, _, f, g = sinkhorn_geomloss_online_potentials_sqeuclid(
            x.detach(),
            y.detach(),
            a,
            b,
            blur=0.1,
            scaling=0.5,
            use_epsilon_scaling=False,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            eps=eps,
            n_iters=n_iters,
            autotune=False,
            block_m=64,
            block_n=64,
            block_k=32,
            num_warps=4,
            return_prelast=True,
        )
        grad_x_ref, grad_y_ref = sinkhorn_geomloss_barycentric_grads_ref(
            x.detach(), y.detach(), a, b, f, g, eps
        )

    torch.testing.assert_close(grad_x, grad_x_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grad_y, grad_y_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_double_backward_matches_hvp_x_reference():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 64, 48, 16
    eps = 0.2
    n_iters = 3

    x = torch.randn(n, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(m, d, device=device, dtype=torch.float32)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    loss = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=False,
        normalize=False,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        hvp_tau2=1e-5,
        hvp_max_cg_iter=64,
        hvp_cg_rtol=1e-6,
        hvp_cg_atol=1e-6,
        hvp_use_preconditioner=True,
    )

    val = loss(a, x, b, y)
    grad_x = torch.autograd.grad(val, x, create_graph=True)[0]

    v = torch.randn_like(x)
    hvp_autograd = torch.autograd.grad((grad_x * v).sum(), x)[0]

    with torch.no_grad():
        _, _, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
            x.detach(),
            y,
            a,
            b,
            blur=0.1,
            scaling=0.5,
            use_epsilon_scaling=False,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            eps=eps,
            n_iters=n_iters,
            autotune=False,
            block_m=64,
            block_n=64,
            block_k=32,
            num_warps=4,
            return_prelast=True,
        )
        f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)
        hvp_ref, _ = hvp_x_sqeuclid_from_potentials(
            x.detach(),
            y,
            f_hat,
            g_hat,
            v,
            eps=eps,
            tau2=loss.hvp_tau2,
            max_cg_iter=loss.hvp_max_cg_iter,
            cg_rtol=loss.hvp_cg_rtol,
            cg_atol=loss.hvp_cg_atol,
            use_preconditioner=loss.hvp_use_preconditioner,
            allow_tf32=False,
            use_exp2=False,
            block_m=loss.block_m,
            block_n=loss.block_n,
            block_k=loss.block_k,
            num_warps=int(loss.num_warps or 4),
            num_stages=loss.num_stages,
        )

    torch.testing.assert_close(hvp_autograd, hvp_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_double_backward_multiscale_matches_hvp_x_reference():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 64, 48, 3
    eps = 0.2
    n_iters = 3

    x = torch.randn(n, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(m, d, device=device, dtype=torch.float32)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    loss = SamplesLoss(
        blur=0.1,
        scaling=0.5,
        debias=False,
        potentials=False,
        normalize=False,
        backend="multiscale",
        multiscale_blocksparse_backend="taskcsr",
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        truncate=5.0,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=32,
        block_n=32,
        block_k=16,
        num_warps=1,
        hvp_tau2=1e-5,
        hvp_max_cg_iter=64,
        hvp_cg_rtol=1e-6,
        hvp_cg_atol=1e-6,
        hvp_use_preconditioner=True,
    )

    val = loss(a, x, b, y)
    grad_x = torch.autograd.grad(val, x, create_graph=True)[0]

    v = torch.randn_like(x)
    hvp_autograd = torch.autograd.grad((grad_x * v).sum(), x)[0]

    with torch.no_grad():
        hvp_ref, _ = hvp_x_sqeuclid_multiscale(
            x.detach(),
            y,
            a,
            b,
            v,
            eps_list=[eps] * n_iters,
            truncate=loss.truncate,
            cluster_scale=loss.cluster_scale,
            max_coarse_levels=loss.max_coarse_levels,
            multiscale_blocksparse_backend=loss.multiscale_blocksparse_backend,
            tau2=loss.hvp_tau2,
            max_cg_iter=loss.hvp_max_cg_iter,
            cg_rtol=loss.hvp_cg_rtol,
            cg_atol=loss.hvp_cg_atol,
            use_preconditioner=loss.hvp_use_preconditioner,
            allow_tf32=False,
            use_exp2=False,
            autotune=False,
            block_m=loss.block_m,
            block_n=loss.block_n,
            block_k=loss.block_k,
            num_warps=loss.num_warps,
            num_stages=loss.num_stages,
        )

    torch.testing.assert_close(hvp_autograd, hvp_ref, rtol=1e-5, atol=1e-5)
