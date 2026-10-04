"""Coordinate rounding, its opt-out, and derivatives through dense SamplesLoss.

The 2-D fixture separates coordinate/norm consistency from the remaining TF32
gradient arithmetic. Tolerances are relative norms measured on an A100:
values 1e-6, gradients 6e-4, and HVPs 2e-5. Exact contracts use zero tolerance.
"""

import pytest
import torch

from flash_sinkhorn import SamplesLoss
import flash_sinkhorn.samples_loss as sl


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SCHEDULE = [0.5, 0.2, 0.08] + [0.04] * 80


def _clouds(batched=False, dtype=torch.float32):
    gen = torch.Generator().manual_seed(0)
    shape = (2,) if batched else ()
    x = 0.2 * torch.rand(*shape, 257, 2, generator=gen) + 0.25
    y = 0.25 * torch.rand(*shape, 193, 2, generator=gen) + torch.tensor([0.6, 0.4])
    return x.cuda().to(dtype), y.cuda().to(dtype)


def _centered_cloud(x, y):
    a = torch.full(x.shape[:-1], 1 / x.shape[-2], device=x.device)
    b = torch.full(y.shape[:-1], 1 / y.shape[-2], device=y.device)
    a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
    total = a.sum(-1) + b.sum(-1)
    mu = ((a[..., None] * x.float()).sum(-2) + (b[..., None] * y.float()).sum(-2)) / total[..., None]
    return x.float() - mu.unsqueeze(-2), y.float() - mu.unsqueeze(-2), a, b


def _rounded(t):
    # Independent arithmetic reference: 10 fraction bits, round to nearest even.
    mantissa, exponent = torch.frexp(t.detach().float())
    return torch.ldexp(torch.round(mantissa * 2048) / 2048, exponent)


def _rounded_cloud(x, y):
    cx, cy, _, _ = _centered_cloud(x, y)
    return _rounded(cx), _rounded(cy)


def _options(backend, **kwargs):
    common = dict(backend=backend, half_cost=True, autotune=False)
    if backend == "symmetric":
        common.update(eps_list=SCHEDULE)
    else:
        common.update(use_epsilon_scaling=False, eps=0.04, n_iters=len(SCHEDULE))
    return dict(common, **kwargs)


def _value_and_grads(x, y, **kwargs):
    x, y = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    value = SamplesLoss(**kwargs)(x, y)
    gx, gy = torch.autograd.grad(value.sum(), (x, y))
    return value.detach(), gx, gy


def _relative_close(actual, expected, tolerance):
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    error = (actual.double() - expected.double()).norm() / expected.double().norm()
    assert error <= tolerance, f"relative error {error.item():.9g} > {tolerance:g}"


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pad", [None, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_tf32_value_matches_fp32_on_rounded_cloud(backend, batched, pad, dtype):
    """Omitting rounding, or rounding before centring, leaves a separable cost error."""
    x, y = _clouds(batched, dtype)
    xr, yr = _rounded_cloud(x, y)
    options = _options(backend, debias=True, pad_to_multiple=pad)
    actual = SamplesLoss(**options)(x, y)
    reference = SamplesLoss(**options, allow_tf32=False)(xr, yr)
    _relative_close(actual, reference, 1e-6)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("batched", [False, True])
def test_default_tf32_gradients_reach_both_inputs(backend, batched):
    """A detached round would sever autograd; unrounded coordinates miss this bound."""
    x, y = _clouds(batched)
    xr, yr = _rounded_cloud(x, y)
    options = _options(backend, debias=True)
    _, gx, gy = _value_and_grads(x, y, **options)
    _, gx_ref, gy_ref = _value_and_grads(xr, yr, **options, allow_tf32=False)
    _relative_close(gx, gx_ref, 6e-4)
    _relative_close(gy, gy_ref, 6e-4)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_default_tf32_hvp_reaches_input(backend, batched, swap):
    """Exercise the supported x-only HVP in either cloud order, without debiasing."""
    x, y = _clouds(batched)
    if swap:
        x, y = y, x
    xr, yr = _rounded_cloud(x, y)
    v = torch.randn(x.shape, generator=torch.Generator().manual_seed(7)).to(x)

    def evaluate(xx, yy, **kwargs):
        xx = xx.clone().requires_grad_(True)
        value = SamplesLoss(**_options(backend, debias=False, **kwargs))(xx, yy)
        (grad,) = torch.autograd.grad(value.sum(), xx, create_graph=True)
        (hv,) = torch.autograd.grad((grad * v).sum(), xx)
        assert torch.isfinite(grad).all()
        return hv

    actual = evaluate(x, y)
    reference = evaluate(xr, yr, allow_tf32=False)
    _relative_close(actual, reference, 2e-5)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("batched", [False, True])
def test_opt_out_matches_unrounded_solver_bitwise(backend, batched):
    """The opt-out must preserve the unrounded low-level solve, including potentials."""
    x, y = _clouds(batched)
    cx, cy, a, b = _centered_cloud(x, y)
    options = _options(backend, truncate_tf32=False)
    value = SamplesLoss(**options)(x, y)
    f, g = SamplesLoss(**options, potentials=True)(x, y)

    def solve(xx, yy, aa, bb):
        if backend == "symmetric":
            return sl.sinkhorn_flashstyle_symmetric(xx, yy, aa, bb, eps_list=SCHEDULE,
                                                    cost_scale=0.5, autotune=False)
        return sl.sinkhorn_flashstyle_alternating(xx, yy, aa, bb, eps=0.04, n_iters=len(SCHEDULE),
                                                 cost_scale=0.5, autotune=False)

    if batched:
        results = [solve(*args) for args in zip(cx, cy, a, b)]
        f_ref = torch.stack([pair[0] for pair in results])
        g_ref = torch.stack([pair[1] for pair in results])
        # SamplesLoss reduces each batch member separately; a batched reduction can differ by one FP32 ulp.
        value_ref = torch.stack([(aa * ff).sum() + (bb * gg).sum()
                                 for aa, bb, (ff, gg) in zip(a, b, results)])
    else:
        f_ref, g_ref = solve(cx, cy, a, b)
        value_ref = (a * f_ref).sum() + (b * g_ref).sum()
    torch.testing.assert_close(f, f_ref, rtol=0, atol=0)
    torch.testing.assert_close(g, g_ref, rtol=0, atol=0)
    torch.testing.assert_close(value, value_ref, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_are_not_rounded_again(backend, batched, dtype):
    """Centring promotes to FP32; deciding by that promoted dtype changes these inputs."""
    x, y = _clouds(batched, dtype)
    options = _options(backend, debias=True)
    expected = _value_and_grads(x, y, **options, truncate_tf32=False)
    actual = _value_and_grads(x, y, **options, truncate_tf32=True)
    default = _value_and_grads(x, y, **options)
    for result in (actual, default):
        for got, want in zip(result, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("swap", [False, True])
def test_mixed_dtypes_round_each_side_independently(backend, swap):
    x, y = _clouds()
    x, y = (x.half(), y.double()) if swap else (x.double(), y.bfloat16())
    cx, cy, _, _ = _centered_cloud(x, y)
    xr = _rounded(cx) if x.dtype == torch.float64 else cx
    yr = _rounded(cy) if y.dtype == torch.float64 else cy
    options = _options(backend, debias=True, allow_tf32=False)
    actual = SamplesLoss(**options, truncate_tf32=True)(x, y)
    reference = SamplesLoss(**options)(xr, yr)
    _relative_close(actual, reference, 1e-6)


@cuda
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
def test_none_follows_allow_tf32_and_explicit_true_overrides_it(backend):
    x, y = _clouds()
    options = _options(backend, debias=True, allow_tf32=False)
    expected = _value_and_grads(x, y, **options, truncate_tf32=False)
    actual = _value_and_grads(x, y, **options, truncate_tf32=None)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    xr, yr = _rounded_cloud(x, y)
    rounded_value = SamplesLoss(**options, truncate_tf32=True)(x, y)
    reference = SamplesLoss(**options)(xr, yr)
    _relative_close(rounded_value, reference, 1e-6)
    assert not torch.equal(rounded_value, actual[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rounding_has_identity_first_and_second_derivative(dtype):
    """Noncontiguous inputs, ties-to-even, negatives, and both autograd paths."""
    x = torch.tensor([[1 + 1 / 2048, 1 + 3 / 2048],
                      [-1 - 1 / 2048, -1 - 3 / 2048]], dtype=dtype).T.requires_grad_(True)
    expected = torch.tensor([[1, 1 + 2 / 1024], [-1, -1 - 2 / 1024]], dtype=dtype).T
    assert hasattr(sl, "_round_to_tf32"), "dense rounding needs a straight-through helper"
    rounded = sl._round_to_tf32(x)
    torch.testing.assert_close(rounded, expected, rtol=0, atol=0)
    # Nonlinear consumers expose a detached/zero first derivative or broken double backward.
    (grad,) = torch.autograd.grad(rounded.square().sum(), x, create_graph=True)
    (hv,) = torch.autograd.grad(grad.sum(), x)
    torch.testing.assert_close(grad, 2 * expected, rtol=0, atol=0)
    torch.testing.assert_close(hv, torch.full_like(x, 2), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rounding_helper_preserves_low_precision(dtype):
    x = torch.tensor([1.125, -2.5], dtype=dtype, requires_grad=True)
    assert hasattr(sl, "_round_to_tf32"), "dense rounding needs a straight-through helper"
    result = sl._round_to_tf32(x)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    (grad,) = torch.autograd.grad(result.sum(), x)
    torch.testing.assert_close(grad, torch.ones_like(x), rtol=0, atol=0)


def test_multiscale_rejects_disabling_coordinate_rounding():
    with pytest.raises(ValueError, match="truncate_tf32=False.*always rounds"):
        SamplesLoss(backend="multiscale", truncate_tf32=False)


@pytest.mark.parametrize("flag", [None, True])
def test_multiscale_accepts_mandatory_rounding(flag):
    loss = SamplesLoss(backend="multiscale", truncate_tf32=flag)
    assert loss.truncate_tf32
