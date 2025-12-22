import pytest
import torch

from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)
from ot_triton.samples_loss import SamplesLoss


def _log_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.float()
    out = w.log()
    out = torch.where(w > 0, out, torch.full_like(out, -100000.0))
    return out


def _jump_index(eps_list: list[float], *, scale2: float) -> int:
    jump = len(eps_list) - 1
    for k in range(2, len(eps_list)):
        if float(scale2) > float(eps_list[k]):
            jump = k - 1
            break
    return int(jump)


def _voxel_clusterize_ref(
    w: torch.Tensor, x: torch.Tensor, *, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f = x.float()
    w_f = w.float()
    mins = x_f.min(dim=0).values
    coords = torch.floor((x_f - mins) / float(scale)).to(torch.int32)
    _, labels = torch.unique(coords, dim=0, return_inverse=True)
    labels = labels.to(torch.int64)
    n_clusters = int(labels.max().item()) + 1 if labels.numel() else 0

    w_coarse = torch.zeros((n_clusters,), device=x.device, dtype=torch.float32)
    w_coarse.index_add_(0, labels, w_f)
    x_sum = torch.zeros((n_clusters, x.shape[1]), device=x.device, dtype=torch.float32)
    x_sum.index_add_(0, labels, w_f[:, None] * x_f)
    x_coarse = x_sum / w_coarse[:, None]

    perm = torch.argsort(labels)
    x_sorted = x[perm]
    w_sorted = w_f[perm]
    labels_sorted = labels[perm]
    counts = torch.bincount(labels_sorted, minlength=n_clusters).to(torch.int32)
    offsets = torch.empty((n_clusters + 1,), device=x.device, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return x_sorted, w_sorted, perm, offsets, x_coarse, w_coarse


def _dense_symmetric_step(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    *,
    eps: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cost = ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1)  # (n,m)

    vals_f = (g[None, :] - cost) / float(eps) + logb[None, :]
    lse_f = torch.logsumexp(vals_f, dim=1)
    cand_f = -float(eps) * lse_f
    f_new = (1.0 - float(alpha)) * f + float(alpha) * cand_f

    vals_g = (f[:, None] - cost) / float(eps) + loga[:, None]
    lse_g = torch.logsumexp(vals_g, dim=0)
    cand_g = -float(eps) * lse_g
    g_new = (1.0 - float(alpha)) * g + float(alpha) * cand_g
    return f_new, g_new


def _blocksparse_fine_step(
    x_sorted: torch.Tensor,
    y_sorted: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    offsets_x: torch.Tensor,
    offsets_y: torch.Tensor,
    keep: torch.Tensor,
    *,
    eps: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_new = f.clone()
    g_new = g.clone()

    for cx in range(keep.shape[0]):
        xs = int(offsets_x[cx].item())
        xe = int(offsets_x[cx + 1].item())
        if xs == xe:
            continue
        nbr_y = torch.nonzero(keep[cx], as_tuple=False).flatten()
        if nbr_y.numel() == 0:
            continue
        y_idx = []
        for cy in nbr_y.tolist():
            ys = int(offsets_y[cy].item())
            ye = int(offsets_y[cy + 1].item())
            if ys != ye:
                y_idx.append(torch.arange(ys, ye, device=x_sorted.device))
        if not y_idx:
            continue
        y_idx = torch.cat(y_idx, dim=0)

        x_block = x_sorted[xs:xe].float()
        y_block = y_sorted[y_idx].float()
        cost = ((x_block[:, None, :] - y_block[None, :, :]) ** 2).sum(dim=-1)
        vals = (g[y_idx][None, :] - cost) / float(eps) + logb[y_idx][None, :]
        lse = torch.logsumexp(vals, dim=1)
        cand = -float(eps) * lse
        f_new[xs:xe] = (1.0 - float(alpha)) * f[xs:xe] + float(alpha) * cand

    keep_t = keep.t().contiguous()
    for cy in range(keep_t.shape[0]):
        ys = int(offsets_y[cy].item())
        ye = int(offsets_y[cy + 1].item())
        if ys == ye:
            continue
        nbr_x = torch.nonzero(keep_t[cy], as_tuple=False).flatten()
        if nbr_x.numel() == 0:
            continue
        x_idx = []
        for cx in nbr_x.tolist():
            xs = int(offsets_x[cx].item())
            xe = int(offsets_x[cx + 1].item())
            if xs != xe:
                x_idx.append(torch.arange(xs, xe, device=x_sorted.device))
        if not x_idx:
            continue
        x_idx = torch.cat(x_idx, dim=0)

        y_block = y_sorted[ys:ye].float()
        x_block = x_sorted[x_idx].float()
        cost = ((x_block[:, None, :] - y_block[None, :, :]) ** 2).sum(dim=-1)
        vals = (f[x_idx][:, None] - cost) / float(eps) + loga[x_idx][:, None]
        lse = torch.logsumexp(vals, dim=0)
        cand = -float(eps) * lse
        g_new[ys:ye] = (1.0 - float(alpha)) * g[ys:ye] + float(alpha) * cand

    return f_new, g_new


def _multiscale_ref(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps_list: list[float],
    truncate: float,
    cluster_scale: float,
    last_extrapolation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_s, a_s, perm_x, offsets_x, x_c, a_c = _voxel_clusterize_ref(a, x, scale=cluster_scale)
    y_s, b_s, perm_y, offsets_y, y_c, b_c = _voxel_clusterize_ref(b, y, scale=cluster_scale)

    loga_c = _log_weights(a_c)
    logb_c = _log_weights(b_c)
    loga_s = _log_weights(a_s)
    logb_s = _log_weights(b_s)

    f = torch.zeros((x_c.shape[0],), device=x.device, dtype=torch.float32)
    g = torch.zeros((y_c.shape[0],), device=x.device, dtype=torch.float32)

    jump = _jump_index(eps_list, scale2=float(cluster_scale) * float(cluster_scale))

    # Init at eps_list[0].
    f, g = _dense_symmetric_step(x_c, y_c, f, g, loga_c, logb_c, eps=eps_list[0], alpha=1.0)

    keep = None
    f_fine = g_fine = None
    for it, step_eps in enumerate(eps_list):
        f, g = _dense_symmetric_step(x_c, y_c, f, g, loga_c, logb_c, eps=step_eps, alpha=0.5)
        if it == jump:
            cost_c = ((x_c[:, None, :] - y_c[None, :, :]) ** 2).sum(dim=-1)
            keep = (f[:, None] + g[None, :]) > (cost_c - float(truncate) * float(step_eps))

            # Extrapolate to fine (sorted) points.
            vals_f = (g[None, :] - ((x_s.float()[:, None, :] - y_c.float()[None, :, :]) ** 2).sum(-1)) / float(step_eps) + logb_c[None, :]
            f_fine = -float(step_eps) * torch.logsumexp(vals_f, dim=1)

            vals_g = (f[:, None] - ((x_c.float()[:, None, :] - y_s.float()[None, :, :]) ** 2).sum(-1)) / float(step_eps) + loga_c[:, None]
            g_fine = -float(step_eps) * torch.logsumexp(vals_g, dim=0)
            break

    assert keep is not None and f_fine is not None and g_fine is not None

    f_s = f_fine
    g_s = g_fine
    # Remaining iterations on fine.
    for step_eps in eps_list[jump + 1 :]:
        f_s, g_s = _blocksparse_fine_step(
            x_s,
            y_s,
            f_s,
            g_s,
            loga_s,
            logb_s,
            offsets_x,
            offsets_y,
            keep,
            eps=step_eps,
            alpha=0.5,
        )

    f_prelast = f_s
    g_prelast = g_s
    if last_extrapolation:
        f_s, g_s = _blocksparse_fine_step(
            x_s,
            y_s,
            f_s,
            g_s,
            loga_s,
            logb_s,
            offsets_x,
            offsets_y,
            keep,
            eps=eps_list[-1],
            alpha=1.0,
        )

    f_u = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    g_u = torch.empty((y.shape[0],), device=y.device, dtype=torch.float32)
    f_u[perm_x] = f_s
    g_u[perm_y] = g_s

    f_p_u = torch.empty_like(f_u)
    g_p_u = torch.empty_like(g_u)
    f_p_u[perm_x] = f_prelast
    g_p_u[perm_y] = g_prelast
    return f_u, g_u, f_p_u, g_p_u


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("d", [1, 2, 3])
def test_multiscale_matches_reference_potentials(d: int):
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m = 96, 80
    eps_list = [0.5, 0.5, 0.5, 0.5]
    truncate = 5.0
    cluster_scale = 2.0

    x = torch.randn(n, d, device=device, dtype=torch.float32) * 10.0
    y = torch.randn(m, d, device=device, dtype=torch.float32) * 10.0
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    f_ref, g_ref, f_p_ref, g_p_ref = _multiscale_ref(
        x,
        y,
        a,
        b,
        eps_list=eps_list,
        truncate=truncate,
        cluster_scale=cluster_scale,
        last_extrapolation=True,
    )

    f, g, f_p, g_p = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
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
    )

    torch.testing.assert_close(f, f_ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(g, g_ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(f_p, f_p_ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(g_p, g_p_ref, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_multiscale_samplesloss_grad_matches_reference():
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 128, 96, 2
    eps_list = [0.5, 0.5, 0.5, 0.5]
    truncate = 5.0
    cluster_scale = 2.0

    x = (torch.randn(n, d, device=device, dtype=torch.float32) * 10.0).requires_grad_(True)
    y = (torch.randn(m, d, device=device, dtype=torch.float32) * 10.0).requires_grad_(True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    f_ref, g_ref, f_p_ref, g_p_ref = _multiscale_ref(
        x.detach(),
        y.detach(),
        a,
        b,
        eps_list=eps_list,
        truncate=truncate,
        cluster_scale=cluster_scale,
        last_extrapolation=True,
    )

    # Reference gradients from prelast potentials on the fine problem.
    x_s, a_s, perm_x, offsets_x, x_c, a_c = _voxel_clusterize_ref(
        a, x.detach(), scale=cluster_scale
    )
    y_s, b_s, perm_y, offsets_y, y_c, b_c = _voxel_clusterize_ref(
        b, y.detach(), scale=cluster_scale
    )
    logb_s = _log_weights(b_s)
    loga_s = _log_weights(a_s)

    # Rebuild keep at the jump using reference (coarse) potentials.
    # (We reuse the fact that _multiscale_ref uses max_coarse_levels=1.)
    # For this test we take keep from the same construction by recomputing it.
    loga_c = _log_weights(a_c)
    logb_c = _log_weights(b_c)
    f_c = torch.zeros((x_c.shape[0],), device=device)
    g_c = torch.zeros((y_c.shape[0],), device=device)
    f_c, g_c = _dense_symmetric_step(x_c, y_c, f_c, g_c, loga_c, logb_c, eps=eps_list[0], alpha=1.0)
    jump = _jump_index(eps_list, scale2=cluster_scale * cluster_scale)
    for it, step_eps in enumerate(eps_list):
        f_c, g_c = _dense_symmetric_step(x_c, y_c, f_c, g_c, loga_c, logb_c, eps=step_eps, alpha=0.5)
        if it == jump:
            cost_c = ((x_c[:, None, :] - y_c[None, :, :]) ** 2).sum(dim=-1)
            keep = (f_c[:, None] + g_c[None, :]) > (cost_c - float(truncate) * float(step_eps))
            break

    f_p_s = f_p_ref[perm_x]
    g_p_s = g_p_ref[perm_y]

    grad_x_ref = torch.zeros((n, d), device=device, dtype=torch.float32)
    for cx in range(int(offsets_x.shape[0] - 1)):
        xs = int(offsets_x[cx].item())
        xe = int(offsets_x[cx + 1].item())
        if xs == xe:
            continue
        nbr_y = torch.nonzero(keep[cx], as_tuple=False).flatten()
        y_idx = []
        for cy in nbr_y.tolist():
            ys = int(offsets_y[cy].item())
            ye = int(offsets_y[cy + 1].item())
            if ys != ye:
                y_idx.append(torch.arange(ys, ye, device=device))
        if not y_idx:
            continue
        y_idx = torch.cat(y_idx, dim=0)

        x_block = x_s[xs:xe].float()
        y_block = y_s[y_idx].float()
        cost = ((x_block[:, None, :] - y_block[None, :, :]) ** 2).sum(dim=-1)
        log_w = (g_p_s[y_idx][None, :] - cost) / eps_list[-1] + logb_s[y_idx][None, :]
        max_log_w = log_w.max(dim=1, keepdim=True).values
        w = torch.exp(log_w - max_log_w)
        z = w.sum(dim=1, keepdim=True).clamp_min(1e-30)
        y_bar = (w @ y_block) / z
        grad_block = 2.0 * a_s[xs:xe].float()[:, None] * (x_block - y_bar)
        grad_x_ref[perm_x[xs:xe]] = grad_block

    grad_y_ref = torch.zeros((m, d), device=device, dtype=torch.float32)
    keep_t = keep.t().contiguous()
    for cy in range(int(offsets_y.shape[0] - 1)):
        ys = int(offsets_y[cy].item())
        ye = int(offsets_y[cy + 1].item())
        if ys == ye:
            continue
        nbr_x = torch.nonzero(keep_t[cy], as_tuple=False).flatten()
        x_idx = []
        for cx in nbr_x.tolist():
            xs = int(offsets_x[cx].item())
            xe = int(offsets_x[cx + 1].item())
            if xs != xe:
                x_idx.append(torch.arange(xs, xe, device=device))
        if not x_idx:
            continue
        x_idx = torch.cat(x_idx, dim=0)

        y_block = y_s[ys:ye].float()
        x_block = x_s[x_idx].float()
        cost = ((x_block[:, None, :] - y_block[None, :, :]) ** 2).sum(dim=-1)
        log_w = (f_p_s[x_idx][:, None] - cost) / eps_list[-1] + loga_s[x_idx][:, None]
        max_log_w = log_w.max(dim=0, keepdim=True).values
        w = torch.exp(log_w - max_log_w)
        z = w.sum(dim=0, keepdim=True).clamp_min(1e-30)
        x_bar = (w.t() @ x_block) / z.t()
        grad_block = 2.0 * b_s[ys:ye].float()[:, None] * (y_block - x_bar)
        grad_y_ref[perm_y[ys:ye]] = grad_block

    loss = SamplesLoss(
        "sinkhorn",
        backend="multiscale",
        normalize=True,
        use_epsilon_scaling=False,
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
    )
    val = loss(a, x, b, y)
    val.backward()

    torch.testing.assert_close(x.grad, grad_x_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(y.grad, grad_y_ref, rtol=5e-3, atol=5e-3)
