"""Invariants of the multiscale fine-stage kernels, checked against torch references."""

import math
import struct

import pytest
import torch

from flash_sinkhorn.kernels._common import log_weights
from flash_sinkhorn.kernels import sinkhorn_blocksparse_sqeuclid
from flash_sinkhorn.kernels.sinkhorn_blocksparse_sqeuclid import blocksparse_lse_fused
from flash_sinkhorn.kernels.sinkhorn_splitn_sqeuclid import flashsinkhorn_lse_splitn
from flash_sinkhorn.multiscale._preprocess import center_coords, round_to_tf32, truncate_tf32
from flash_sinkhorn.multiscale import block_mask, geometry, mask_refine
from flash_sinkhorn.multiscale.block_mask import build_block_mask_csr, csr_transpose
from flash_sinkhorn.multiscale.mask_refine import refine_mask_exact
from flash_sinkhorn.multiscale.sparse_step import sparse_symmetric_step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAD_K = 16
#: Block sizes of the fine stage; problem sizes below scale with B / 64.
BLOCKS = (64, 128, 256)


def _morton_order(points):
    """Sort 3-D points by a 30-bit Morton key."""
    lo, hi = points.min(0).values, points.max(0).values
    q = ((points - lo) / (hi - lo).clamp_min(1e-12) * 1023).round().long().clamp(0, 1023)
    key = torch.zeros(len(points), dtype=torch.int64, device=points.device)
    for bit in range(10):
        for axis in range(3):
            key |= ((q[:, axis] >> bit) & 1) << (3 * bit + axis)
    return torch.argsort(key)


def _clustered(n, seed, centers=8, std=0.03):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.rand(centers, 3, generator=gen)
    pick = torch.randint(centers, (n,), generator=gen)
    return (c[pick] + std * torch.randn(n, 3, generator=gen)).cuda()


def _problem(n, m, B, eps, *, seed=0, iters=30):
    """Sorted, centered, TF32-rounded, padded clouds and potentials of a short dense solve."""
    x, y = _clustered(n, seed), _clustered(m, seed + 1)
    a = torch.full((n,), 1.0 / n, device="cuda")
    b = torch.full((m,), 1.0 / m, device="cuda")
    x, y = center_coords(x, y, a, b)
    x, y = truncate_tf32(x, y)
    x, y = x[_morton_order(x)], y[_morton_order(y)]
    # A few symmetric log-domain iterations in float64 give realistic potentials.
    C = torch.cdist(x.double(), y.double()) ** 2
    f = torch.zeros(n, dtype=torch.float64, device="cuda")
    g = torch.zeros(m, dtype=torch.float64, device="cuda")
    la, lb = a.double().log(), b.double().log()
    for _ in range(iters):
        f_new = -eps * torch.logsumexp((g[None, :] - C) / eps + lb[None, :], dim=1)
        g_new = -eps * torch.logsumexp((f[:, None] - C) / eps + la[:, None], dim=0)
        f, g = 0.5 * (f + f_new), 0.5 * (g + g_new)

    def pad(points, weights, pot, size):
        extra = (-size) % B
        pts = torch.nn.functional.pad(points, (0, 0, 0, extra))
        w = torch.nn.functional.pad(weights, (0, extra))
        p = torch.nn.functional.pad(pot.float(), (0, extra))
        return pts.contiguous(), w, p

    xp, ap, fp = pad(x, a, f, n)
    yp, bp, gp = pad(y, b, g, m)
    alpha = (xp * xp).sum(1)
    beta = (yp * yp).sum(1)
    return dict(
        x3=xp, y3=yp, x=torch.nn.functional.pad(xp, (0, PAD_K - 3)).contiguous(),
        y=torch.nn.functional.pad(yp, (0, PAD_K - 3)).contiguous(),
        f_hat=(fp - alpha).contiguous(), g_hat=(gp - beta).contiguous(),
        alpha=alpha, beta=beta, log_a=log_weights(ap), log_b=log_weights(bp),
        a=ap, b=bp, B=B, eps=eps, N_B=len(xp) // B, M_B=len(yp) // B)


def _widen(points, cols):
    """Zero columns up to ``cols``: the same cost, and from 32 columns the BLOCK_K 32 launch."""
    return torch.nn.functional.pad(points, (0, cols - points.shape[1])).contiguous()


def _boxes(points, B):
    blocks = points.view(-1, B, points.shape[1])
    return blocks.amin(dim=1), blocks.amax(dim=1)


def _mask(p, delta_rel):
    x_lo, x_hi = _boxes(p["x3"], p["B"])
    y_lo, y_hi = _boxes(p["y3"], p["B"])
    return build_block_mask_csr(
        x_lo, x_hi, y_lo, y_hi, p["f_hat"], p["g_hat"], p["alpha"], p["beta"],
        p["log_a"], p["log_b"], p["eps"], B=p["B"], delta_rel=delta_rel)


def _tile_stats(p):
    """Exact tile maxima m_IJ and thresholds without delta_rel, in float64."""
    B = p["B"]
    f = (p["f_hat"] + p["alpha"]).double().masked_fill(p["a"] == 0, -math.inf)
    g = (p["g_hat"] + p["beta"]).double().masked_fill(p["b"] == 0, -math.inf)
    C = torch.cdist(p["x3"].double(), p["y3"].double()) ** 2
    s = f[:, None] + g[None, :] - C
    m_ij = s.view(p["N_B"], B, p["M_B"], B).amax(dim=(1, 3))
    log_mass = torch.maximum(
        torch.logsumexp(p["log_a"].double().view(p["N_B"], B), 1)[:, None],
        torch.logsumexp(p["log_b"].double().view(p["M_B"], B), 1)[None, :])
    return m_ij, log_mass


def _dense_to_csr(keep):
    crow = torch.zeros(keep.shape[0] + 1, dtype=torch.int32, device=keep.device)
    crow[1:] = keep.sum(1).cumsum(0)
    col = keep.nonzero(as_tuple=True)[1].to(torch.int32)
    return crow, col


def _csr_to_dense(crow, col, n_rows, n_cols):
    keep = torch.zeros(n_rows, n_cols, dtype=torch.bool, device=crow.device)
    rows = torch.repeat_interleave(torch.arange(n_rows, device=crow.device),
                                   (crow[1:] - crow[:-1]).long())
    keep[rows, col.long()] = True
    return keep


def _reference_lse(x, y, g_hat, log_w, eps, keep_points=None):
    """-eps LSE_j [2 x.y / eps + g_hat_j / eps + log w_j] over allowed pairs, float64."""
    logits = (2.0 * x.double() @ y.double().T + g_hat.double()[None, :]) / eps + log_w.double()[None, :]
    if keep_points is not None:
        logits = logits.masked_fill(~keep_points, -math.inf)
    return (-eps * torch.logsumexp(logits, dim=1)).float()


@pytest.mark.parametrize("B", BLOCKS)
@pytest.mark.parametrize("n,m", [(1024, 1024), (1000, 1150)])
def test_csr_valid_and_transpose_roundtrip(n, m, B):
    k = B // 64
    p = _problem(k * n, k * m, B, 0.003)
    crow, col = _mask(p, 1e-4)
    assert crow[0] == 0 and crow[-1] == len(col)
    assert bool((crow[1:] >= crow[:-1]).all())
    assert bool((col >= 0).all()) and bool((col < p["M_B"]).all())
    for i in range(p["N_B"]):
        row = col[crow[i]:crow[i + 1]]
        assert bool((row[1:] > row[:-1]).all())
    crow_t, col_t = csr_transpose(crow, col, p["N_B"], p["M_B"])
    dense = _csr_to_dense(crow, col, p["N_B"], p["M_B"])
    assert torch.equal(_csr_to_dense(crow_t, col_t, p["M_B"], p["N_B"]), dense.T)
    crow_tt, col_tt = csr_transpose(crow_t, col_t, p["M_B"], p["N_B"])
    assert torch.equal(crow_tt, crow) and torch.equal(col_tt, col)


@pytest.mark.parametrize("B", BLOCKS)
def test_box_screen_keeps_every_exactly_admitted_pair(B):
    k = B // 64
    p = _problem(k * 1500, k * 1400, B, 0.003)
    delta_rel = 1e-4
    crow, col = _mask(p, delta_rel)
    kept = _csr_to_dense(crow, col, p["N_B"], p["M_B"])
    m_ij, log_mass = _tile_stats(p)
    threshold = p["eps"] * (math.log(delta_rel) - log_mass)
    # A small margin keeps fp32 knife-edge pairs out of the comparison.
    exact = m_ij > threshold + 1e-5
    assert bool(exact.any()) and bool((~kept).any())
    assert not bool((exact & ~kept).any())


@pytest.mark.parametrize("cols", [PAD_K, 32])
@pytest.mark.parametrize("B", BLOCKS)
def test_refine_scores_equal_exact_tile_maxima(B, cols):
    k = B // 64
    p = _problem(k * 1500, k * 1400, B, 0.003)
    delta_rel = 1e-4
    crow, col = _mask(p, delta_rel)
    crow2, col2, scores = refine_mask_exact(
        _widen(p["x"], cols), _widen(p["y"], cols), p["f_hat"], p["g_hat"], p["log_a"], p["log_b"], p["eps"],
        crow, col, B=p["B"], delta_rel=delta_rel, return_scores=True)
    m_ij, log_mass = _tile_stats(p)
    rows = torch.repeat_interleave(torch.arange(p["N_B"], device="cuda"),
                                   (crow[1:] - crow[:-1]).long())
    exact_scores = m_ij[rows, col.long()]
    finite = torch.isfinite(exact_scores)
    torch.testing.assert_close(scores[finite].double(), exact_scores[finite], rtol=0, atol=2e-5)
    refined = _csr_to_dense(crow2, col2, p["N_B"], p["M_B"])
    threshold = p["eps"] * (math.log(delta_rel) - log_mass)
    candidate = _csr_to_dense(crow, col, p["N_B"], p["M_B"])
    clear = (m_ij - threshold).abs() > 1e-4
    expected = candidate & (m_ij > threshold)
    assert torch.equal(refined[clear], expected[clear])
    assert not bool((refined & ~candidate).any())


@pytest.mark.parametrize("cols", [PAD_K, 32])
@pytest.mark.parametrize("B", BLOCKS)
def test_sparse_lse_full_mask_matches_dense_reference(B, cols):
    k = B // 64
    p = _problem(k * 1024, k * 1216, B, 0.01)
    keep = torch.ones(p["N_B"], p["M_B"], dtype=torch.bool, device="cuda")
    crow, col = _dense_to_csr(keep)
    out = blocksparse_lse_fused(_widen(p["x"], cols), _widen(p["y"], cols), p["g_hat"], p["log_b"], p["eps"], 1.0,
                                crow, col, block_m=p["B"], block_n=p["B"])
    ref = _reference_lse(p["x"], p["y"], p["g_hat"], p["log_b"], p["eps"])
    live = p["a"] > 0
    torch.testing.assert_close(out[live], ref[live], rtol=0, atol=2e-5)


@pytest.mark.parametrize("B", BLOCKS)
def test_sparse_lse_partial_mask_matches_restricted_reference(B):
    k = B // 64
    p = _problem(k * 1024, k * 1024, B, 0.01)
    gen = torch.Generator(device="cpu").manual_seed(3)
    keep = torch.rand(p["N_B"], p["M_B"], generator=gen).cuda() < 0.3
    keep[torch.arange(p["N_B"]), torch.arange(p["N_B"]) % p["M_B"]] = True
    crow, col = _dense_to_csr(keep)
    out = blocksparse_lse_fused(p["x"], p["y"], p["g_hat"], p["log_b"], p["eps"], 1.0,
                                crow, col, block_m=p["B"], block_n=p["B"])
    points = keep.repeat_interleave(p["B"], 0).repeat_interleave(p["B"], 1)
    ref = _reference_lse(p["x"], p["y"], p["g_hat"], p["log_b"], p["eps"], points)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)


@pytest.mark.parametrize("B", BLOCKS)
def test_symmetric_step_repairs_empty_block_rows(B):
    k = B // 64
    p = _problem(k * 1024, k * 1088, B, 0.01)
    gen = torch.Generator(device="cpu").manual_seed(5)
    keep = torch.rand(p["N_B"], p["M_B"], generator=gen).cuda() < 0.25
    keep[1] = False                 # an empty source block row
    keep[:, 2] = False              # an empty target block row after transposing
    crow, col = _dense_to_csr(keep)
    crow_t, col_t = csr_transpose(crow, col, p["N_B"], p["M_B"])
    f, g = sparse_symmetric_step(
        p["x"], p["y"], p["f_hat"], p["g_hat"], p["log_a"], p["log_b"], p["eps"],
        crow, col, crow_t, col_t, block=p["B"])

    def expected(xs, ys, pot, log_w, blocks):
        empty = ~blocks.any(1)
        blocks = blocks | empty[:, None]
        points = blocks.repeat_interleave(p["B"], 0).repeat_interleave(p["B"], 1)
        return _reference_lse(xs, ys, pot, log_w, p["eps"], points)

    f_ref = 0.5 * p["f_hat"] + 0.5 * expected(p["x"], p["y"], p["g_hat"], p["log_b"], keep)
    g_ref = 0.5 * p["g_hat"] + 0.5 * expected(p["y"], p["x"], p["f_hat"], p["log_a"], keep.T)
    torch.testing.assert_close(f[p["a"] > 0], f_ref[p["a"] > 0], rtol=0, atol=2e-5)
    torch.testing.assert_close(g[p["b"] > 0], g_ref[p["b"] > 0], rtol=0, atol=2e-5)


@pytest.mark.parametrize("B", BLOCKS)
def test_box_screen_group_skip_keeps_the_same_pairs(B, monkeypatch):
    """The group test skips only groups that hold no kept pair: the CSR equals the pair-by-pair screen's
    bitwise, with zero-weight blocks, a whole zero-weight group and a partial last group, and most groups
    are skipped."""
    gen = torch.Generator(device="cpu").manual_seed(21)
    n, m, eps = 512 * B, (4096 + 37) * B, 1e-4                         # 16 groups of 256 blocks, then 37
    assert (m // B) % block_mask.BLOCK_J == 37
    x, y = torch.rand(n, 3, generator=gen).cuda(), torch.rand(m, 3, generator=gen).cuda()
    x, y = x[_morton_order(x)], y[_morton_order(y)]
    a = torch.rand(n, generator=gen).cuda(); b = torch.rand(m, generator=gen).cuda()
    a[5 * B:7 * B] = 0; b[300 * B:301 * B] = 0                        # zero-weight blocks on both sides
    b[1024 * B:1280 * B] = 0                                          # a whole group of zero-weight blocks
    a, b = a / a.sum(), b / b.sum()
    x_lo, x_hi = _boxes(x, B)
    y_lo, y_hi = _boxes(y, B)
    alpha, beta = (x * x).sum(1), (y * y).sum(1)
    f_hat = -alpha + 1e-3 * torch.randn(n, generator=gen).cuda()
    g_hat = -beta + 1e-3 * torch.randn(m, generator=gen).cuda()
    args = (x_lo, x_hi, y_lo, y_hi, f_hat, g_hat, alpha, beta, log_weights(a), log_weights(b), eps)
    crow, col = build_block_mask_csr(*args, B=B, delta_rel=1e-4)

    def never(y_lo, y_hi, g_bound, lb_bound):                       # groups that cannot reject
        k = -(-y_lo.shape[0] // block_mask.BLOCK_J)
        inf = torch.full((k, y_lo.shape[1]), float("inf"), device=y_lo.device)
        return -inf, inf, torch.full((k,), float("inf"), device=y_lo.device), torch.zeros(k, device=y_lo.device)

    monkeypatch.setattr(block_mask, "_target_groups", never)
    crow_ref, col_ref = build_block_mask_csr(*args, B=B, delta_rel=1e-4)
    assert torch.equal(crow, crow_ref) and torch.equal(col, col_ref)
    kept = _csr_to_dense(crow, col, len(x_lo), len(y_lo))
    groups = kept.view(len(x_lo), -1)[:, :(len(y_lo) // block_mask.BLOCK_J) * block_mask.BLOCK_J]
    empty_groups = ~groups.view(len(x_lo), -1, block_mask.BLOCK_J).any(2)
    assert bool(kept.any()) and float(empty_groups.double().mean()) > 0.5   # most (row, group) pairs are skipped


@pytest.mark.parametrize("B", BLOCKS)
def test_box_screen_group_screen_is_exact_at_the_threshold(B, monkeypatch):
    """Potentials near +-1e3 that cancel to about 1e-3 put many pairs within an ulp of the threshold: the two-level
    screen still keeps exactly the pairs of the flat one (both evaluate the same predicate code)."""
    gen = torch.Generator(device="cpu").manual_seed(22)
    n, m, eps = 512 * B, 1024 * B, 1e-5
    x, y = torch.rand(n, 3, generator=gen).cuda(), torch.rand(m, 3, generator=gen).cuda()
    x, y = x[_morton_order(x)], y[_morton_order(y)]
    a, b = torch.full((n,), 1.0 / n, device="cuda"), torch.full((m,), 1.0 / m, device="cuda")
    x_lo, x_hi = _boxes(x, B)
    y_lo, y_hi = _boxes(y, B)
    alpha, beta = (x * x).sum(1), (y * y).sum(1)
    f_hat = 1e3 + 1e-3 * torch.randn(n, generator=gen).cuda() - alpha
    g_hat = -1e3 + 1e-3 * torch.randn(m, generator=gen).cuda() - beta
    args = (x_lo, x_hi, y_lo, y_hi, f_hat, g_hat, alpha, beta, log_weights(a), log_weights(b), eps)
    crow, col = build_block_mask_csr(*args, B=B, delta_rel=1e-4)

    def never(y_lo, y_hi, g_bound, lb_bound):
        k = -(-y_lo.shape[0] // block_mask.BLOCK_J)
        inf = torch.full((k, y_lo.shape[1]), float("inf"), device=y_lo.device)
        return -inf, inf, torch.full((k,), float("inf"), device=y_lo.device), torch.zeros(k, device=y_lo.device)

    monkeypatch.setattr(block_mask, "_target_groups", never)
    crow_ref, col_ref = build_block_mask_csr(*args, B=B, delta_rel=1e-4)
    assert torch.equal(crow, crow_ref) and torch.equal(col, col_ref)
    kept = _csr_to_dense(crow, col, len(x_lo), len(y_lo))
    assert 0.01 < float(kept.double().mean()) < 0.99                    # a real boundary, not all or nothing


def test_launch_tables_cover_the_block_sizes():
    """Every block size the solver can choose has a fixed launch, one warp per 32 rows of a block."""
    for table in (sinkhorn_blocksparse_sqeuclid.LAUNCH, mask_refine.LAUNCH):
        assert sorted(table) == list(geometry.BLOCK_SIZES)
        assert all(warps == B // 32 for B, (warps, _) in table.items())
    p = _problem(1024, 1024, 64, 0.01)
    crow, col = _dense_to_csr(torch.ones(p["N_B"], p["M_B"], dtype=torch.bool, device="cuda"))
    with pytest.raises(ValueError, match="block sizes"):
        blocksparse_lse_fused(p["x"], p["y"], p["g_hat"], p["log_b"], p["eps"], 1.0, crow, col,
                              block_m=32, block_n=32)
    with pytest.raises(ValueError, match="block sizes"):
        refine_mask_exact(p["x"], p["y"], p["f_hat"], p["g_hat"], p["log_a"], p["log_b"], p["eps"],
                          crow, col, B=32, delta_rel=1e-4)


def test_splitn_lse_matches_reference():
    p = _problem(1000, 5000, 64, 0.01)
    eps = p["eps"]
    bias = (p["g_hat"] / eps + p["log_b"]).contiguous()
    out = flashsinkhorn_lse_splitn(p["x"][:1000], p["y"], bias, eps, num_splits=64,
                                   allow_tf32=False)
    ref = _reference_lse(p["x"][:1000], p["y"], p["g_hat"], p["log_b"], eps)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)


@pytest.mark.parametrize("dead", [(0, 2000), (0, 4900), (1000, 2600)])
def test_splitn_lse_drops_minus_inf_bias_slices(dead):
    """Targets with -inf bias drop out even when they fill whole splits, the first one included."""
    p = _problem(1000, 5000, 64, 0.01)
    eps = p["eps"]
    log_w = p["log_b"].clone()
    log_w[dead[0]:dead[1]] = -math.inf
    bias = (p["g_hat"] / eps + log_w).contiguous()
    out = flashsinkhorn_lse_splitn(p["x"][:1000], p["y"], bias, eps, num_splits=64,
                                   allow_tf32=False)
    ref = _reference_lse(p["x"][:1000], p["y"], p["g_hat"], log_w, eps)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)


def _rne_tf32_reference(value):
    """Round one fp32 value to TF32, nearest-even, on its bit pattern."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign, mag = bits & 0x80000000, bits & 0x7FFFFFFF
    q, rem = mag >> 13, mag & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and q & 1):
        q += 1
    return struct.unpack("<f", struct.pack("<I", sign | (q << 13)))[0]


def test_round_to_tf32_is_nearest_even():
    gen = torch.Generator(device="cpu").manual_seed(0)
    values = torch.cat([torch.randn(2000, generator=gen) * 10.0 ** torch.randint(-6, 6, (2000,), generator=gen),
                        torch.tensor([1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, -(1.0 + 2.0 ** -11)])])
    rounded = round_to_tf32(values)
    expected = torch.tensor([_rne_tf32_reference(float(v)) for v in values])
    assert torch.equal(rounded, expected)
    assert torch.equal(round_to_tf32(rounded), rounded)
    half = values.half()
    assert round_to_tf32(half) is half


def test_center_coords_zeroes_the_joint_centroid():
    x, y = _clustered(500, 0), _clustered(700, 1)
    a = torch.rand(500, device="cuda"); a /= a.sum()
    b = torch.rand(700, device="cuda"); b /= b.sum()
    xc, yc = center_coords(x, y, a, b)
    mu = 0.5 * ((a[:, None] * xc).sum(0) + (b[:, None] * yc).sum(0))
    torch.testing.assert_close(mu, torch.zeros(3, device="cuda"), rtol=0, atol=1e-6)
