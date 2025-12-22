import pytest
import torch

from ot_triton import SamplesLoss
from ot_triton.testing.reference_sinkhorn import sinkhorn_geomloss_potentials_ref
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    max_diameter,
    sinkhorn_geomloss_online_potentials_sqeuclid,
)


def _rand_inputs(n, m, d, device):
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float16)
    y = torch.randn(m, d, device=device, dtype=torch.float16)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    return x, y, a, b


def _sqdist_cost_full(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    y_f = y.float()
    x2 = (x_f * x_f).sum(dim=-1, keepdim=True)
    y2 = (y_f * y_f).sum(dim=-1, keepdim=True).transpose(-2, -1)
    return x2 + y2 - 2.0 * torch.matmul(x_f, y_f.transpose(-2, -1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_geomloss_fixed_eps_matches_ref():
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.5
    n_iters = 3

    f_t, g_t = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )
    f_ref, g_ref = sinkhorn_geomloss_potentials_ref(
        x, y, a, b, use_epsilon_scaling=False, eps=eps, n_iters=n_iters
    )

    torch.testing.assert_close(f_t, f_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(g_t, g_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_geomloss_exp2_matches_exp():
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)

    eps = 0.5
    n_iters = 2

    f_exp2, g_exp2 = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        use_exp2=True,
    )
    f_exp, g_exp = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
        use_exp2=False,
    )

    torch.testing.assert_close(f_exp2, f_exp, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(g_exp2, g_exp, rtol=1e-4, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_geomloss_eps_scaling_matches_ref_prefix():
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)

    blur = 0.5
    scaling = 0.7
    n_iters = 4

    f_t, g_t = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        blur=blur,
        scaling=scaling,
        use_epsilon_scaling=True,
        n_iters=n_iters,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )
    f_ref, g_ref = sinkhorn_geomloss_potentials_ref(
        x,
        y,
        a,
        b,
        blur=blur,
        scaling=scaling,
        use_epsilon_scaling=True,
        n_iters=n_iters,
    )

    torch.testing.assert_close(f_t, f_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(g_t, g_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_geomloss_eps_scaling_matches_geomloss_tensorized():
    geomloss = pytest.importorskip("geomloss")
    from geomloss.sinkhorn_samples import sinkhorn_tensorized

    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)

    blur = 0.5
    scaling = 0.7
    diameter = max_diameter(x, y)

    f_t, g_t = sinkhorn_geomloss_online_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        blur=blur,
        scaling=scaling,
        use_epsilon_scaling=True,
        diameter=diameter,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )

    f_gl, g_gl = sinkhorn_tensorized(
        a.unsqueeze(0),
        x.unsqueeze(0),
        b.unsqueeze(0),
        y.unsqueeze(0),
        p=2,
        blur=blur,
        scaling=scaling,
        diameter=diameter,
        cost=_sqdist_cost_full,
        debias=False,
        potentials=True,
    )
    f_gl = f_gl.squeeze(0)
    g_gl = g_gl.squeeze(0)

    torch.testing.assert_close(f_t, f_gl, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(g_t, g_gl, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_samplesloss_backward_matches_geomloss_tensorized():
    geomloss = pytest.importorskip("geomloss")
    from geomloss import SamplesLoss as GeomLossSamplesLoss

    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 128, 96, 32
    x0 = torch.randn(n, d, device=device, dtype=torch.float32)
    y0 = torch.randn(m, d, device=device, dtype=torch.float32)

    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    blur = 0.2
    scaling = 0.5
    diameter = max_diameter(x0, y0)

    loss_t = SamplesLoss(
        "sinkhorn",
        blur=blur,
        scaling=scaling,
        debias=False,
        potentials=False,
        normalize=False,
        use_epsilon_scaling=True,
        diameter=diameter,
        last_extrapolation=True,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=32,
        num_warps=4,
    )
    loss_g = GeomLossSamplesLoss(
        "sinkhorn",
        p=2,
        blur=blur,
        scaling=scaling,
        diameter=diameter,
        cost=_sqdist_cost_full,
        debias=False,
        potentials=False,
        backend="tensorized",
    )

    x_t = x0.clone().requires_grad_(True)
    y_t = y0.clone().requires_grad_(True)
    x_g = x0.clone().requires_grad_(True)
    y_g = y0.clone().requires_grad_(True)

    val_t = loss_t(a, x_t, b, y_t)
    val_g = loss_g(a, x_g, b, y_g)

    grad_x_t, grad_y_t = torch.autograd.grad(val_t, (x_t, y_t))
    grad_x_g, grad_y_g = torch.autograd.grad(val_g, (x_g, y_g))

    torch.testing.assert_close(val_t, val_g, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grad_x_t, grad_x_g, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grad_y_t, grad_y_g, rtol=5e-3, atol=5e-3)
