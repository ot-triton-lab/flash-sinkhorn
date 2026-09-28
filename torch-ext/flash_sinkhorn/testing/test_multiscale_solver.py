"""End-to-end tests of backend="multiscale" and tables of its automatic rules."""

import warnings

import pytest
import torch

import flash_sinkhorn.multiscale as multiscale
from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import flashsinkhorn_lse
from flash_sinkhorn.multiscale import block_mask, coarse, fine, geometry, solver
from flash_sinkhorn.multiscale import MultiscaleInfo, solve_multiscale
from flash_sinkhorn.multiscale._preprocess import preprocess_coordinates, round_to_tf32
from flash_sinkhorn.multiscale.check import SampledCheck, _stage_one_action, sample_indices

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TOL = 5e-3


def clustered(n, seed):
    """32 Gaussian clusters (standard deviation 0.02) centered uniformly in the unit cube."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(32, 3, generator=g)
    idx = torch.randint(32, (n,), generator=g)
    return (centers[idx] + 0.02 * torch.randn(n, 3, generator=g)).cuda()


def uniform_cube(n, seed):
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(seed)).cuda()


def uniform_weights(n):
    return torch.full((n,), 1.0 / n, device="cuda")


def transform(x, y, b, g, eps):
    """Physical c-transform ``-eps log sum_j b_j exp((g_j - |x_i - y_j|^2) / eps)`` in IEEE fp32."""
    pad = lambda p: torch.nn.functional.pad(p, (0, geometry.PAD_K - 3))
    t = flashsinkhorn_lse(pad(x), pad(y), (g - (y * y).sum(1)) / eps + b.log(), eps,
                          allow_tf32=False, autotune=False)
    return t + (x * x).sum(1)


def residuals(x, y, a, b, f, g, eps):
    """Exact (||P1 - a||_1, ||P^T 1 - b||_1) over all pairs."""
    side = lambda x, y, a, b, f, g: float(
        (a.double() * torch.expm1((f.double() - transform(x, y, b, g, eps).double()) / eps).abs()).sum())
    return side(x, y, a, b, f, g), side(y, x, b, a, g, f)


def converged_cost(x, y, a, b, f, g, eps):
    """<a, f> + <b, g> at the fixed point, by symmetric IEEE updates from ``(f, g)``."""
    for _ in range(40):
        for _ in range(50):
            f, g = 0.5 * (f + transform(x, y, b, g, eps)), 0.5 * (g + transform(y, x, a, f, eps))
        if max(residuals(x, y, a, b, f, g, eps)) < 1e-6:
            return float((a.double() * f.double()).sum() + (b.double() * g.double()).sum())
    raise AssertionError("the reference did not converge")


def solve(x, y, a, b, eps):
    xp, yp = preprocess_coordinates(x, y, a, b)
    f, g, info = solve_multiscale(xp, yp, a, b, eps, TOL)
    return xp, yp, f, g, info


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("eps, path", [
    (1e-2, "lift"),    # the lifted coarse potentials already pass the check
    (1e-3, "sparse"),  # block-sparse fine updates
])
def test_paths_meet_tol(eps, path):
    n = 2**14
    x, y, a = clustered(n, 0), clustered(n, 1), uniform_weights(n)
    xp, yp, f, g, info = solve(x, y, a, a, eps)
    assert info.accepted and info.stop == "accepted"
    if path == "lift":
        assert info.returned == "lift" and info.fine_updates == 0
    else:
        assert info.fine_updates > 0 and len(info.tiles) == info.fine_updates
    assert max(residuals(xp, yp, a, a, f, g, eps)) <= 1.1 * TOL


@cuda
def test_dense_path_meets_tol():
    """At eps = 3e-2 the geometry gate makes every fine update of a uniform cube dense (no tiles);
    a tight tol keeps the lift from passing, so the fine stage runs."""
    n, eps, tol = 2**14, 3e-2, 2e-4
    x, y, a = uniform_cube(n, 0), uniform_cube(n, 1), uniform_weights(n)
    xp, yp = preprocess_coordinates(x, y, a, a)
    f, g, info = solve_multiscale(xp, yp, a, a, eps, tol)
    assert info.accepted and info.fine_updates > 0 and not info.tiles
    assert max(residuals(xp, yp, a, a, f, g, eps)) <= 1.1 * tol


@cuda
def test_dense_fine_step_matches_reference():
    """When the blur-scaled ball covers half the cloud the fine update is dense (no mask)."""
    n, eps = 2**12, 3e-2
    x, y, a = uniform_cube(n, 0), uniform_cube(n, 1), uniform_weights(n)
    xp, yp = preprocess_coordinates(x, y, a, a)
    joint = geometry.joint_sort(xp, yp, a, a)
    sx, sy = geometry.block_summaries(joint.source), geometry.block_summaries(joint.target)
    geo = geometry.execution_geometry(joint, 64, sx, sy)
    assert fine.dense_by_geometry(geo, eps)
    f0 = 0.01 * torch.randn(n, generator=torch.Generator().manual_seed(2)).cuda()
    g0 = torch.zeros(n, device="cuda")
    state, tiles = fine.fine_step(geo, fine.warm_state(geo, f0, g0), eps, 1e-6, alpha=0.5)
    f, g = fine.physical_pair(geo, state)
    f_ref = 0.5 * f0.double() + 0.5 * transform(xp, yp, a, g0, eps).double()
    g_ref = 0.5 * g0.double() + 0.5 * transform(yp, xp, a, f0, eps).double()
    assert tiles is None
    torch.testing.assert_close(f.double(), f_ref, rtol=0, atol=1e-5)
    torch.testing.assert_close(g.double(), g_ref, rtol=0, atol=1e-5)


@cuda
def test_unequal_sizes_nonuniform_and_zero_weights():
    x, y = clustered(2**14, 2), clustered(3 * 2**12, 3)
    gen = torch.Generator().manual_seed(0)
    a = (torch.rand(len(x), generator=gen) + 0.1).cuda()
    b = (torch.rand(len(y), generator=gen) + 0.1).cuda()
    a[:100], b[-100:] = 0.0, 0.0
    a, b = a / a.sum(), b / b.sum()
    xp, yp, f, g, info = solve(x, y, a, b, 1e-2)
    assert info.accepted
    ka, kb = a > 0, b > 0
    assert torch.isfinite(f[ka]).all() and torch.isfinite(g[kb]).all()
    assert max(residuals(xp[ka], yp[kb], a[ka], b[kb], f[ka], g[kb], 1e-2)) <= 1.1 * TOL


@cuda
@pytest.mark.parametrize("half_cost", [False, True])
def test_cost_matches_converged_reference(half_cost):
    """The returned cost is within eps * tol of the converged cost of the same (preprocessed) clouds."""
    n, blur = 2**12, 0.1
    x, y, a = clustered(n, 4), clustered(n, 5), uniform_weights(n)
    kw = dict(blur=blur, half_cost=half_cost, backend="multiscale")
    cost = float(SamplesLoss("sinkhorn", **kw)(a, x, a, y))
    f, g = SamplesLoss("sinkhorn", potentials=True, **kw)(a, x, a, y)
    s = 0.5 if half_cost else 1.0
    xp, yp = preprocess_coordinates(x, y, a, a)
    ref = s * converged_cost(xp, yp, a, a, f / s, g / s, blur**2 / s)
    assert abs(cost - ref) <= blur**2 * TOL


@cuda
def test_debiased_cost_matches_converged_reference():
    n, blur = 2**12, 0.1
    eps = blur**2
    x, y, a = clustered(n, 6), clustered(n, 7), uniform_weights(n)
    cost = float(SamplesLoss("sinkhorn", blur=blur, backend="multiscale", debias=True)(a, x, a, y))
    xp, yp = preprocess_coordinates(x, y, a, a)
    ref = 0.0
    for weight, (p, q) in ((1.0, (xp, yp)), (-0.5, (xp, xp)), (-0.5, (yp, yp))):
        f, g, _ = solve_multiscale(p, q, a, a, eps, TOL)
        ref += weight * converged_cost(p, q, a, a, f, g, eps)
    assert abs(cost - ref) <= 2 * eps * TOL


@cuda
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_input_precisions(dtype):
    """Any input precision is centered, TF32-rounded and solved in fp32 on that cloud."""
    n, blur = 2**12, 0.1
    x, y, a = clustered(n, 8).to(dtype), clustered(n, 9).to(dtype), uniform_weights(n)
    kw = dict(blur=blur, backend="multiscale")
    cost = float(SamplesLoss("sinkhorn", **kw)(x, y))
    f, g = SamplesLoss("sinkhorn", potentials=True, **kw)(x, y)
    assert f.dtype == g.dtype == torch.float32
    xp, yp = preprocess_coordinates(x, y, a, a)
    assert abs(cost - converged_cost(xp, yp, a, a, f, g, blur**2)) <= blur**2 * TOL


def _cells(k, seed):
    gen = torch.Generator().manual_seed(seed)
    points = round_to_tf32(torch.rand(k, 3, generator=gen) - 0.5)
    mass = torch.rand(k, generator=gen) + 0.1
    return geometry.Cells(points.cuda(), (mass / mass.sum()).cuda())


def _dense_transform(x, y, b, g, eps):
    """fp64 c-transform on the explicit cost matrix (tiny clouds only)."""
    C = torch.cdist(x.double(), y.double()) ** 2
    return -eps * torch.logsumexp((g.double()[None, :] - C) / eps + b.double().log()[None, :], dim=1)


@cuda
def test_coarse_updates_match_dense_reference():
    """A replacing update from zero, then averaged symmetric updates along the schedule."""
    cx, cy = _cells(300, 1), _cells(280, 2)
    schedule = (0.5, 0.2, 0.05)
    state = coarse.coarse_updates(coarse.coarse_state(cx, cy), schedule[:1], alpha=1.0)
    state = coarse.coarse_updates(state, schedule)
    f = torch.zeros(300, dtype=torch.float64, device="cuda")
    g = torch.zeros(280, dtype=torch.float64, device="cuda")
    for eps, alpha in zip(schedule[:1] + schedule, (1.0, 0.5, 0.5, 0.5)):
        f_new = _dense_transform(cx.points, cy.points, cy.mass, g, eps)
        g_new = _dense_transform(cy.points, cx.points, cx.mass, f, eps)
        f, g = (1 - alpha) * f + alpha * f_new, (1 - alpha) * g + alpha * g_new
    torch.testing.assert_close((state.f_hat + state.alpha).double(), f, rtol=0, atol=1e-5)
    torch.testing.assert_close((state.g_hat + state.beta).double(), g, rtol=0, atol=1e-5)
    assert state.averaged_steps == len(schedule)


@cuda
def test_lift_matches_dense_reference():
    """One half-step from every point against the coarse pair, returned in input order."""
    gen = torch.Generator().manual_seed(3)
    x = round_to_tf32(torch.rand(3000, 3, generator=gen) - 0.5).cuda()
    y = round_to_tf32(torch.rand(2500, 3, generator=gen) - 0.5).cuda()
    a, b = uniform_weights(3000), uniform_weights(2500)
    joint = geometry.joint_sort(x, y, a, b)
    sx, sy = geometry.block_summaries(joint.source), geometry.block_summaries(joint.target)
    geo = geometry.execution_geometry(joint, 64, sx, sy)
    cx, cy = (geometry.build_cells(c, 54, round_to_tf32) for c in (joint.source, joint.target))
    state = coarse.coarse_updates(coarse.coarse_state(cx, cy), (0.3,), alpha=1.0)
    f, g = coarse.lift(geo, state, 0.1)
    f_ref = _dense_transform(x, cy.points, cy.mass, state.g_hat + state.beta, 0.1)
    g_ref = _dense_transform(y, cx.points, cx.mass, state.f_hat + state.alpha, 0.1)
    torch.testing.assert_close(f.double(), f_ref, rtol=0, atol=1e-5)
    torch.testing.assert_close(g.double(), g_ref, rtol=0, atol=1e-5)


@cuda
def test_adapter_eps_potentials_debias_and_warning(monkeypatch):
    """half_cost solves ||x - y||^2 at eps / s and scales the potentials by s; debias adds two solves."""
    calls = []

    def fake(x, y, a, b, eps, tol):
        calls.append((len(x), len(y), eps, tol))
        info = MultiscaleInfo(accepted=len(calls) != 3, stop="accepted", B=64, t=0, coarse_cells=(1, 1),
                              schedule_entries=1,
                              continuation_updates=0, fine_updates=0, terminal_previews=0,
                              returned="lift", returned_step=0)
        return torch.ones(len(x), device=x.device), torch.full((len(y),), 2.0, device=y.device), info

    monkeypatch.setattr(multiscale, "solve_multiscale", fake)
    x, y = clustered(256, 10), clustered(128, 11)
    loss = SamplesLoss("sinkhorn", blur=0.1, backend="multiscale", half_cost=True, tol=1e-3, potentials=True)
    f, g = loss(x, y)
    assert calls == [(256, 128, pytest.approx(0.1**2 / 0.5), 1e-3)]
    assert torch.equal(f, torch.full_like(f, 0.5)) and torch.equal(g, torch.full_like(g, 1.0))
    assert loss.last_multiscale_info.accepted

    calls.clear()
    loss = SamplesLoss("sinkhorn", blur=0.1, backend="multiscale", debias=True)
    with pytest.warns(RuntimeWarning, match="1 solve"):
        cost = loss(x, y)
    assert [c[:2] for c in calls] == [(256, 128), (256, 256), (128, 128)]
    assert float(cost) == pytest.approx(0.0, abs=1e-6)  # 3 - 3/2 - 3/2


@cuda
def test_not_confirmed_returns_best_candidate():
    """A target far below the fp32 arithmetic floor (about 1e-6 here) is never confirmed: the fine
    stage stalls and the solve returns its best check with accepted=False."""
    x, y, a = clustered(2**12, 12), clustered(2**12, 13), uniform_weights(2**12)
    xp, yp = preprocess_coordinates(x, y, a, a)
    f, g, info = solve_multiscale(xp, yp, a, a, 1e-2, 1e-9)
    assert not info.accepted and info.stop in ("stalled", "ceiling")
    assert info.fine_updates >= solver.FINE_WINDOW * (1 + solver.FINE_SLOW_WINDOWS)
    ranked = [c for c in info.checks if c["rank"] is not None]
    best = min(ranked, key=lambda c: c["rank"])
    assert (info.returned, info.returned_step) == (best["kind"], best["step"])
    assert torch.isfinite(f).all() and torch.isfinite(g).all()


@cuda
def test_argument_validation():
    with pytest.raises(ValueError, match="tol applies only"):
        SamplesLoss("sinkhorn", tol=1e-3)
    for kwargs in (dict(reach=1.0), dict(eps=1e-2), dict(eps_list=[1e-2]), dict(n_iters=10),
                   dict(allow_tf32=False), dict(use_epsilon_scaling=False)):
        with pytest.raises(ValueError, match="multiscale"):
            SamplesLoss("sinkhorn", backend="multiscale", **kwargs)
    for tol in (0.0, -1e-3, 2.0, float("nan")):
        with pytest.raises(ValueError, match="tol"):
            SamplesLoss("sinkhorn", backend="multiscale", tol=tol)

    x, y = clustered(1024, 14), clustered(1024, 15)
    loss = SamplesLoss("sinkhorn", blur=0.1, backend="multiscale")
    with pytest.raises(ValueError, match="three-dimensional"):
        loss(x[:, :2].contiguous(), y[:, :2].contiguous())
    with pytest.raises(ValueError, match="batched"):
        loss(x[None], y[None])
    with pytest.raises(ValueError, match="CUDA"):
        loss(x.cpu(), y.cpu())
    with pytest.raises(NotImplementedError, match="gradients"):
        loss(x.clone().requires_grad_(), y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # accuracy is not the point here
        f, g = SamplesLoss("sinkhorn", blur=0.1, backend="multiscale", potentials=True)(
            x.clone().requires_grad_(), y)
    assert f.shape == g.shape == (1024,) and not f.requires_grad

    a = uniform_weights(1024)
    with pytest.raises(ValueError, match="sum to one"):
        solve_multiscale(x, y, 2 * a, a, 1e-2, TOL)
    negative = a.clone()
    negative[0], negative[1] = -a[0], 3 * a[1]  # still sums to one
    with pytest.raises(ValueError, match="nonnegative"):
        solve_multiscale(x, y, negative, a, 1e-2, TOL)
    with pytest.raises(ValueError, match="CUDA"):
        solve_multiscale(x.cpu(), y.cpu(), a.cpu(), a.cpu(), 1e-2, TOL)
    with pytest.raises(ValueError, match="TF32-rounded"):
        solve_multiscale(x, y, a, a, 1e-2, TOL)  # raw coordinates
    xp, yp = preprocess_coordinates(x, y, a, a)
    heavy = a * (1 + 2e-4)
    solver._validate(xp, yp, heavy, a, TOL)          # within min(1e-3, tol / 4)
    with pytest.raises(ValueError, match="sum to one within"):
        solver._validate(xp, yp, heavy, a, 5e-4)     # 2e-4 > 5e-4 / 4


# ---------------------------------------------------------------------------
# Sampled check and memory admission
# ---------------------------------------------------------------------------


@cuda
def test_sampled_check_estimates_the_residual():
    n, eps = 2**14, 1e-2
    x, y, a = clustered(n, 16), clustered(n, 17), uniform_weights(n)
    xp, yp, f, g, _ = solve(x, y, a, a, eps)
    noise = torch.randn(n, generator=torch.Generator().manual_seed(0)).cuda()
    check = SampledCheck(xp, yp, a, a, eps, TOL)
    for scale in (0.0, 0.05):
        fp = f + scale * eps * noise
        exact = residuals(xp, yp, a, a, fp, g, eps)
        entry = check.evaluate(fp, g)
        assert entry["accepted"] == (scale == 0.0)
        for stage in (entry["first"], entry["final"]):
            for (mean, se, _), value in zip((stage["row"], stage["col"]), exact):
                assert abs(mean - value) <= 4 * se


@cuda
def test_sampled_check_enlarges_when_inconclusive():
    """Near tol the stage-one interval straddles it: stage two decides, stage one still ranks."""
    n, eps = 2**14, 1e-2
    x, y, a = clustered(n, 16), clustered(n, 17), uniform_weights(n)
    xp, yp, f, g, _ = solve(x, y, a, a, eps)
    noise = torch.randn(n, generator=torch.Generator().manual_seed(1)).cuda()
    check = SampledCheck(xp, yp, a, a, eps, TOL)
    entries = [check.evaluate(f + s * eps * noise, g) for s in torch.linspace(0.0, 8e-3, 33).tolist()]
    enlarged = [e for e in entries if e["enlarged"]]
    assert enlarged and len(check._stages[8192][0].idx) == 8192
    for e in enlarged:
        assert e["final"] is not e["first"] and e["rank"] == e["first"]["score"]
        assert e["accepted"] == (e["final"]["score"] <= TOL)


def test_stage_one_action():
    score = lambda row, col: dict(finite=True, row=row, col=col)
    assert _stage_one_action(score([1e-3, 1e-4, 0], [2e-3, 1e-4, 0]), TOL) == "accept"     # upper 2.3e-3
    assert _stage_one_action(score([1e-2, 1e-4, 0], [2e-3, 1e-4, 0]), TOL) == "reject"     # lower 9.7e-3
    assert _stage_one_action(score([4.9e-3, 1e-4, 0], [1e-3, 1e-4, 0]), TOL) == "enlarge"  # 4.6e-3 .. 5.2e-3
    assert _stage_one_action(dict(finite=False, row=None, col=None), TOL) == "enlarge"


def test_sampled_indices_skip_zero_weights():
    w = torch.rand(1000, generator=torch.Generator().manual_seed(0))
    w[::3] = 0.0
    idx = sample_indices(w / w.sum(), 8192, 0)
    assert bool((w[idx] > 0).all())
    assert torch.equal(idx, sample_indices(w / w.sum(), 8192, 0))


@cuda
def test_memory_admission():
    device = torch.device("cuda")
    block_mask.csr_memory_admission(1, 1, 1, device)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        block_mask.csr_memory_admission(10**13, 1, 1, device)


@cuda
def test_refused_mask_raises(monkeypatch):
    def refuse(*args):
        raise torch.cuda.OutOfMemoryError("refused")

    monkeypatch.setattr(block_mask, "csr_memory_admission", refuse)
    x, y, a = clustered(2**14, 18), clustered(2**14, 19), uniform_weights(2**14)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="refused"):
        solve(x, y, a, a, 1e-3)


# ---------------------------------------------------------------------------
# Rules (CPU)
# ---------------------------------------------------------------------------


def test_preprocess_centers_and_rounds_to_tf32():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 3, generator=gen, dtype=torch.float64) + 100.0
    y = torch.randn(900, 3, generator=gen, dtype=torch.float64) + 100.0
    xp, yp = preprocess_coordinates(x, y, torch.full((1000,), 1e-3), torch.full((900,), 1 / 900))
    assert xp.dtype == yp.dtype == torch.float32
    for p in (xp, yp):
        assert not bool((p.view(torch.int32) & 0x1FFF).any())
    assert float(0.5 * (xp.mean(0) + yp.mean(0)).abs().max()) < 1e-3


def test_morton_keys_and_prefix_counts():
    gen = torch.Generator().manual_seed(0)
    x, y = torch.rand(5000, 3, generator=gen), torch.rand(4000, 3, generator=gen)
    joint = geometry.joint_sort(x, y, torch.full((5000,), 1 / 5000), torch.full((4000,), 1 / 4000))
    src = joint.source
    lo, hi = torch.minimum(x.amin(0), y.amin(0)), torch.maximum(x.amax(0), y.amax(0))
    q = ((src.points - lo) / (hi - lo) * geometry.MAX_QUANTIZED).long().clamp(0, geometry.MAX_QUANTIZED)
    naive = [sum(((int(row[axis]) >> bit) & 1) << (3 * bit + axis) for axis in range(3) for bit in range(21))
             for row in q[:64]]
    assert naive == src.keys[:64].tolist()
    assert bool((src.keys[1:] >= src.keys[:-1]).all())
    counts = geometry.prefix_counts(src)
    for t in (0, 30, 45, 51, 57, 62):
        assert counts[t] == int(torch.unique(src.keys >> t).numel())
    cells = geometry.build_cells(src, 45, lambda c: c)
    assert len(cells.mass) == counts[45] and float(cells.mass.sum()) == pytest.approx(1.0, abs=1e-5)


def test_cells_are_mass_weighted_centroids():
    gen = torch.Generator().manual_seed(4)
    x = torch.rand(2000, 3, generator=gen)
    w = torch.rand(2000, generator=gen)
    w[::7] = 0.0
    w = w / w.sum()
    src = geometry.joint_sort(x, x.clone(), w, w).source
    cells = geometry.build_cells(src, 54, lambda c: c)
    groups = {}
    for key, point, mass in zip((src.keys >> 54).tolist(), src.points.double(), src.weights.double()):
        groups.setdefault(key, []).append((point, mass))
    kept = [members for _, members in sorted(groups.items()) if sum(m for _, m in members) > 0]
    mass = torch.stack([sum(m for _, m in members) for members in kept])
    centers = torch.stack([sum(m * p for p, m in members) / sum(m for _, m in members) for members in kept])
    torch.testing.assert_close(cells.mass.double(), mass, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(cells.points.double(), centers, rtol=0.0, atol=1e-6)


def _morton_joint(points):
    """The cloud of integer ``points`` against itself, sorted in exact Morton order."""
    q = points.long()
    keys = geometry._spread(q[:, 0]) | (geometry._spread(q[:, 1]) << 1) | (geometry._spread(q[:, 2]) << 2)
    perm = torch.argsort(keys)
    n = len(points)
    cloud = geometry.SortedCloud(points[perm].float(), torch.full((n,), 1.0 / n), keys[perm], perm,
                                 torch.argsort(perm), n)
    sides = (points.amax(0) - points.amin(0)).float()
    return geometry.JointSort(cloud, cloud, float(sides.norm()), tuple(sides.tolist()))


def test_block_size_rule():
    r = torch.arange(16)
    grid = _morton_joint(torch.cartesian_prod(r, r, r))  # 64-blocks are 4x4x4 cubes, merged pairs 8x4x4
    line = _morton_joint(torch.stack((torch.arange(4096), torch.zeros(4096, dtype=torch.long),
                                      torch.zeros(4096, dtype=torch.long)), dim=1))  # merging doubles each diagonal
    for joint, B in ((grid, 128), (line, 64)):
        sx, sy = geometry.block_summaries(joint.source), geometry.block_summaries(joint.target)
        assert geometry.choose_block_size(joint, sx, sy, 2**30) == B
    sx, sy = geometry.block_summaries(grid.source), geometry.block_summaries(grid.target)
    assert geometry.summary_bytes(4096, 4096, 128) == 2048 and geometry.summary_bytes(4096, 4096, 256) == 1024
    assert geometry.choose_block_size(grid, sx, sy, 2000) == 256   # 2048 > 0.8 * 2000 >= 1024
    lx, ly = geometry.block_summaries(line.source), geometry.block_summaries(line.target)
    assert geometry.choose_block_size(line, lx, ly, 5000) == 128   # geometry says 64; 4096 > 0.8 * 5000
    with pytest.raises(RuntimeError):
        geometry.choose_block_size(grid, sx, sy, 1000)


def test_cell_level_rule():
    wide = ((1.0, 1.0, 1.0), 1.0)  # a unit box at blur 1: every cell is narrow enough
    # 2^20 points per side, B = 128, a 40 MiB L2: at most 2 * 8192 cells per side, which t = 18 first allows.
    counts = {t: max(1, 2 ** (20 - t // 3)) for t in range(geometry.KEY_BITS)}
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 40 * 2**20, *wide) == 18
    # A 1 MiB L2 caps the cells at 0.8 * 2^20 / (112 * 2) = 3744 per side: t = 24 (2^12 = 4096) is too many.
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 2**20, *wide) == 27
    # 2^12 points: the floor of 8192 cells admits every point, so the coarse stage runs on the points.
    small = {t: max(1, 2 ** (12 - t // 3)) for t in range(geometry.KEY_BITS)}
    assert geometry.choose_cell_level(small, small, 2**12, 2**12, 64, 40 * 2**20, *wide) == 0
    # Cells wider than CELL_EDGE_BLURS blurs: the coarsest narrow enough t, beyond the 1 MiB L2 budget. In a box of
    # MAX_QUANTIZED per side the longest cell edge is 2^ceil(t/3): 512 at t = 27, 128 at t = 19..21, 64 at t = 18.
    box = (float(geometry.MAX_QUANTIZED),) * 3
    blurs = lambda w: (w / geometry.CELL_EDGE_BLURS) ** 2  # the eps whose blur-scaled width is w
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 2**20, box, blurs(600)) == 27
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 2**20, box, blurs(200)) == 21
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 2**20, box, blurs(100)) == 18
    # No level within the cell limit (t >= 18) is narrow enough: the finest one allowed.
    assert geometry.choose_cell_level(counts, counts, 2**20, 2**20, 128, 2**20, box, blurs(50)) == 18
    # The constant itself, on boxes that fill their cells: a 2^27-point cube 512 blurs wide needs t = 45
    # (cells 8 blurs wide; t = 46 gives 16), while a 2^24-point box 273-276 blurs wide keeps t = 46 (8.54).
    eps = (1000 / 512) ** 2
    full = {t: min(2**27, 2 ** (geometry.KEY_BITS - t)) for t in range(geometry.KEY_BITS)}
    assert geometry.choose_cell_level(full, full, 2**27, 2**27, 256, 40 * 2**20, (1000.0,) * 3, eps) == 45
    sub = {t: min(2**24, 2 ** (geometry.KEY_BITS - t)) for t in range(geometry.KEY_BITS)}
    assert geometry.choose_cell_level(sub, sub, 2**24, 2**24, 128, 40 * 2**20, (533.75, 539.25, 537.0), eps) == 46


def test_cell_edge_follows_the_morton_interleave():
    side = float(geometry.MAX_QUANTIZED)
    assert geometry.cell_edge((side, 0.0, 0.0), 1) == 2.0  # x holds the lowest bit of each triple
    assert geometry.cell_edge((0.0, side, 0.0), 1) == 1.0
    assert geometry.cell_edge((side, side, side), 3) == 2.0 and geometry.cell_edge((side, side, side), 4) == 4.0


def test_fine_stall_rule():
    assert not solver._window_slow(None, 0.5, True)   # the first window only records its score
    assert solver._window_slow(None, None, True)      # a nonfinite score is no progress ...
    assert solver._window_slow(0.5, None, False)
    assert solver._window_slow(None, 0.5, False)      # ... and no progress can be measured from one
    assert solver._window_slow(0.5, 0.499, False)     # 0.2% < 1%
    assert not solver._window_slow(0.5, 0.49, False)  # 2%
    assert solver._window_slow(0.0, 0.0, False)


def _mocked_fine_stage(monkeypatch, score, accept=lambda kind, h: False, terminal_score=None, unstable_from=None,
                       unstable_previews_only=False):
    """Run solve_multiscale with the fine update and the sampled check replaced: every check scores
    ``score(h)`` (None = nonfinite; ``terminal_score(h)`` for previews if given) and passes when
    ``accept(kind, h)``; from ``unstable_from`` ordinary updates on, the mask of every update (or only of
    the previews) is refused as unstable."""
    progress = dict(h=0, kind="lift")

    def fake_fine_step(geo, state, eps, delta_rel, *, alpha=0.5):
        if (unstable_from is not None and state.ordinary_steps >= unstable_from
                and (alpha == 1.0 or not unstable_previews_only)):
            raise fine.UnstableMask("mocked")
        ordinary = alpha == 0.5
        progress["h"] += ordinary
        progress["kind"] = "ordinary" if ordinary else "terminal"
        steps = state.ordinary_steps + ordinary
        return fine.FineState(state.f_hat, state.g_hat, steps, progress["kind"]), (1, 1)

    def fake_evaluate(self, f, g):
        kind, h = progress["kind"], progress["h"]
        rank = terminal_score(h) if kind == "terminal" and terminal_score is not None else score(h)
        stage = dict(finite=rank is not None, score=rank)
        return dict(accepted=accept(kind, h), first=stage, final=stage, enlarged=False, rank=rank,
                    pair_finite=True)

    monkeypatch.setattr(solver, "fine_step", fake_fine_step)
    monkeypatch.setattr(SampledCheck, "evaluate", fake_evaluate)
    x, y, a = clustered(2**12, 12), clustered(2**12, 13), uniform_weights(2**12)
    xp, yp = preprocess_coordinates(x, y, a, a)
    _, _, info = solve_multiscale(xp, yp, a, a, 1e-2, TOL)
    windows = [c["rank"] for c in info.checks if c["kind"] == "ordinary" and c["step"] % solver.FINE_WINDOW == 0]
    previews = [c["step"] for c in info.checks if c["kind"] == "terminal"]
    return info, windows, previews


@cuda
@pytest.mark.parametrize("name, score, stop, updates", [
    ("flat", lambda h: 0.5, "stalled", 256),                          # first window, then three slow ones
    ("nonfinite first window", lambda h: None if h == 64 else 0.5, "stalled", 192),  # its preview is finite
    ("slow, one fast window, slow", lambda h: 0.5 if h < 256 else 0.49, "stalled", 448),
    ("2% per window", lambda h: 0.5 * 0.98 ** (h / 64), "ceiling", 2048),
])
def test_fine_stage_stop_rules(monkeypatch, name, score, stop, updates):
    info, windows, previews = _mocked_fine_stage(monkeypatch, score, terminal_score=lambda h: 0.5)
    assert not info.accepted and info.stop == stop and info.fine_updates == updates
    assert len(windows) == updates // solver.FINE_WINDOW and previews[-1] == updates
    ranked = [c for c in info.checks if c["rank"] is not None]
    if ranked:
        best = min(ranked, key=lambda c: c["rank"])
        assert (info.returned, info.returned_step) == (best["kind"], best["step"])
    else:
        assert (info.returned, info.returned_step) == ("lift", 0)


@cuda
def test_fine_stage_stops_as_unstable(monkeypatch):
    # No finite score at all: the preview after the last lift and the first check stop the stage.
    info, _, _ = _mocked_fine_stage(monkeypatch, lambda h: None)
    assert (info.accepted, info.stop) == (False, "unstable") and info.fine_updates == 2 - info.terminal_previews
    assert (info.returned, info.returned_step) == ("lift", 0)
    # Finite scores up to h = 100 and none after: the first two checks past 100.
    info, _, _ = _mocked_fine_stage(monkeypatch, lambda h: 0.5 if h <= 100 else None)
    assert info.stop == "unstable" and 100 < info.fine_updates <= 112 and info.returned_step <= 100
    # One nonfinite check between finite ones is no reason to stop (updates 1 and 2 are always checked).
    info, _, _ = _mocked_fine_stage(monkeypatch, lambda h: None if h == 1 else 0.5, terminal_score=lambda h: 0.5)
    assert info.stop == "stalled" and [c["rank"] for c in info.checks if c["step"] == 1][0] is None
    # A mask refused as unstable: the updates completed before it are counted.
    info, _, _ = _mocked_fine_stage(monkeypatch, lambda h: 0.5, unstable_from=50)
    assert (info.stop, info.fine_updates) == ("unstable", 50)
    # ... and a preview's mask refused the same way (the preview at 64 updates, a milestone).
    info, _, _ = _mocked_fine_stage(monkeypatch, lambda h: 0.5 if h < 64 else 2 * TOL, unstable_from=64,
                                    unstable_previews_only=True)
    assert (info.stop, info.fine_updates) == ("unstable", 64)


@cuda
def test_fine_step_refuses_masks_with_empty_block_rows(monkeypatch):
    n, eps = 2**14, 1e-3
    x, y, a = clustered(n, 0), clustered(n, 1), uniform_weights(n)
    xp, yp = preprocess_coordinates(x, y, a, a)
    f, g, _ = solve_multiscale(xp, yp, a, a, eps, TOL)
    joint = geometry.joint_sort(xp, yp, a, a)
    sx, sy = geometry.block_summaries(joint.source), geometry.block_summaries(joint.target)
    geo = geometry.execution_geometry(joint, 64, sx, sy)  # 256 block rows per side
    assert not fine.dense_by_geometry(geo, eps)
    state, delta_rel, refine = fine.warm_state(geo, f, g), fine.mask_delta_rel(geo, TOL), fine.refine_mask_exact

    def emptying(k, side="source"):
        """The real refinement, then the first k source (or target) block rows lose every block."""
        def refine_k(*args, **kw):
            crow, col = refine(*args, **kw)
            counts = (crow[1:] - crow[:-1]).long()
            rows = torch.repeat_interleave(torch.arange(len(counts), device=crow.device), counts)
            keep = (rows if side == "source" else col.long()) >= k
            crow2 = torch.zeros_like(crow)
            crow2[1:] = torch.bincount(rows[keep], minlength=len(counts)).cumsum(0).to(crow.dtype)
            return crow2, col[keep]
        return refine_k

    fine.fine_step(geo, state, eps, delta_rel)
    monkeypatch.setattr(fine, "refine_mask_exact", emptying(2))  # 0.8% of the block rows
    fine.fine_step(geo, state, eps, delta_rel)
    monkeypatch.setattr(fine, "refine_mask_exact", emptying(3))  # 1.2%
    with pytest.raises(fine.UnstableMask):
        fine.fine_step(geo, state, eps, delta_rel)
    monkeypatch.setattr(fine, "refine_mask_exact", emptying(3, side="target"))
    with pytest.raises(fine.UnstableMask):
        fine.fine_step(geo, state, eps, delta_rel)
    # Zero-weight block rows admit nothing and are not counted.
    monkeypatch.setattr(fine, "refine_mask_exact", refine)
    geo["log_a"] = geo["log_a"].clone()
    geo["log_a"][:3 * 64] = block_mask.LOG_ZERO_SENTINEL
    fine.fine_step(geo, state, eps, delta_rel)


@cuda
def test_fine_stage_always_previews_at_128_updates(monkeypatch):
    """A terminal preview is taken at update 128 whatever the score."""
    info, _, previews = _mocked_fine_stage(monkeypatch, lambda h: 0.5,
                                           accept=lambda kind, h: kind == "terminal" and h == 128)
    assert info.accepted and (info.returned, info.returned_step, info.fine_updates) == ("terminal", 128, 128)
    assert solver.FIXED_PREVIEW in previews


def test_annealing_schedule():
    prefix, tail = geometry.annealing_schedule(2.0, 1e-2)
    assert prefix[-1] == 1e-2 and 1e-2 not in prefix[:-1] and tail == (1e-2,) * geometry.TARGET_TAIL
    assert all(u >= v for u, v in zip(prefix, prefix[1:]))
    assert geometry.annealing_schedule(0.05, 1e-2) == ((1e-2,), (1e-2,) * geometry.TARGET_TAIL)


def test_check_schedule_rules():
    entry = lambda rank, enlarged=False: dict(rank=rank, enlarged=enlarged)
    far = entry(0.5)
    assert solver._interval(1.0, 1.0, far, TOL, False) == 16
    assert solver._interval(0.2, 1.0, far, TOL, False) == 2
    assert solver._interval(0.05, 1.0, far, TOL, False) == 1
    for args in ((1.0, None, far, TOL, False), (1.0, 1.0, entry(None), TOL, False),
                 (1.0, 1.0, entry(0.5, True), TOL, False), (1.0, 1.0, entry(2 * TOL), TOL, False),
                 (1.0, 1.0, far, TOL, True)):
        assert solver._interval(*args) == 1
    assert [solver._next_check(h, 128, 16) for h in (0, 1, 2, 16, 120, 128)] == [1, 2, 16, 32, 128, 128]
    assert solver._next_check(20, 128, 4) == 24 and solver._next_check(30, 128, 4) == 32
    assert solver._preview_due(2, 128, far, TOL) and solver._preview_due(128, 128, far, TOL)
    assert solver._preview_due(solver.FIXED_PREVIEW, 2048, far, TOL) and solver._preview_due(2048, 2048, far, TOL)
    assert not solver._preview_due(192, 2048, far, TOL)
    assert solver._preview_due(32, 128, entry(TOL), TOL) and not solver._preview_due(32, 128, far, TOL)
    assert not solver._preview_due(3, 128, entry(TOL), TOL) and not solver._preview_due(1, 128, entry(TOL), TOL)


def _continue(monkeypatch, statistics):
    """continue_coarse on a counting state whose stall statistic follows ``statistics``."""
    class State:
        def __init__(self, updates):
            self.updates = updates
    values = iter(statistics)
    monkeypatch.setattr(coarse, "coarse_updates", lambda s, schedule, alpha=0.5: State(s.updates + len(schedule)))
    monkeypatch.setattr(coarse, "stall_statistic", lambda before, after, eps: next(values))
    return coarse.continue_coarse(State(0), 1e-2, TOL)


def test_continuation_stop_rules(monkeypatch):
    batch = coarse.CONTINUATION_BATCH
    state, committed, steps = _continue(monkeypatch, [1.0, 0.5, 0.2 * TOL, 0.4 * TOL])
    assert committed and steps == state.updates == 4 * batch      # two batches at or below tol / 2
    state, committed, steps = _continue(monkeypatch, [1.0, 0.999, 0.998, 0.997])
    assert committed and steps == 4 * batch                        # three batches under 1% progress
    state, committed, steps = _continue(monkeypatch, [1.0, None])
    assert not committed and state.updates == 0                    # invalid statistic: nothing is kept
    state, committed, steps = _continue(monkeypatch, [1.0 / (k + 1) for k in range(64)])
    assert committed and steps == coarse.CONTINUATION_MAX_UPDATES  # the budget
