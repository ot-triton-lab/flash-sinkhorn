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
        cost=_sqdist_cost_full,  # Use full cost to match our kernel's convention
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("d", [1025, 2048, 4096])
def test_large_d_gradient_regression(d):
    """Regression test for large-D gradient path (d > 1024 uses tiled kernel).

    This test exercises the large-D kernel path which tiles over the feature
    dimension and accumulates into global memory instead of shared memory.

    Tests dimensions up to d=4096 to match the documented "tested up to d=4096"
    in GRADIENT_HVP.md and README.md.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Use smaller n,m for larger d to keep test fast and avoid OOM
    n, m = (32, 32) if d >= 4096 else (64, 64)
    x = torch.randn(n, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(m, d, device=device, dtype=torch.float32, requires_grad=True)

    a = torch.ones(n, device=device, dtype=torch.float32) / n
    b = torch.ones(m, device=device, dtype=torch.float32) / m

    loss = SamplesLoss(
        "sinkhorn",
        blur=0.5,  # Larger blur for stability at high d
        scaling=0.5,
        debias=False,
        potentials=False,
        normalize=False,
        use_epsilon_scaling=False,
        eps=0.25,
        n_iters=32,
        allow_tf32=False,
        use_exp2=False,
    )

    # Forward pass
    val = loss(a, x, b, y)

    # Backward pass - this exercises the large-D kernel
    grad_x, grad_y = torch.autograd.grad(val, (x, y))

    # Basic sanity checks
    assert grad_x.shape == x.shape, f"grad_x shape mismatch: {grad_x.shape} vs {x.shape}"
    assert grad_y.shape == y.shape, f"grad_y shape mismatch: {grad_y.shape} vs {y.shape}"
    assert torch.isfinite(grad_x).all(), "grad_x contains non-finite values"
    assert torch.isfinite(grad_y).all(), "grad_y contains non-finite values"
    assert grad_x.abs().max() > 0, "grad_x is all zeros"
    assert grad_y.abs().max() > 0, "grad_y is all zeros"

    # Gradient direction consistency: moving in -grad direction should not increase loss significantly
    # Use relative tolerance to handle varying loss magnitudes across dimensions
    with torch.no_grad():
        # Use adaptive step size based on gradient norm (smaller for large gradients)
        grad_norm = grad_x.norm()
        step = 0.01 / max(grad_norm.item(), 1.0)
        x_new = x - step * grad_x
        val_new = loss(a, x_new, b, y)

        # Check with relative tolerance: val_new <= val * (1 + rtol) + atol
        # This is more robust to varying loss magnitudes and curvature
        rtol = 0.05  # 5% relative tolerance
        atol = 1e-2  # Absolute tolerance for small losses
        threshold = val * (1 + rtol) + atol
        assert val_new <= threshold, (
            f"Gradient direction inconsistent: val={val:.6f}, val_new={val_new:.6f}, "
            f"threshold={threshold:.6f}, step={step:.6f}"
        )
