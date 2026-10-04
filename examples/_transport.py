"""Apply the transport plan of a SamplesLoss solve without forming its n x m matrix.

``SamplesLoss(potentials=True, debias=False)`` returns potentials f, g in the GeomLoss convention

    P = diag(a) exp((f + g - C) / eps) diag(b),    C_ij = cost_scale * |x_i - y_j|^2.

The streaming kernels ``apply_plan_vec_flashstyle`` and ``apply_plan_mat_flashstyle`` take "shifted"
potentials f - cost_scale |x|^2 and g - cost_scale |y|^2 with the log-weights, and compute P v or P^T u in
memory linear in n + m. These helpers do the conversion. The weights a and b must be positive and must be
the weights the solve used: SamplesLoss normalizes each side to sum to one by default (normalize=True), so
pass the normalized weights.

These helpers evaluate costs in FP32 on the supplied coordinates. Default TF32 solves round centred coordinates,
so reconstructing a plan on the original points can differ from the plan on the solved coordinates.
"""

from __future__ import annotations

import torch

from flash_sinkhorn.kernels import apply_plan_mat_flashstyle, apply_plan_vec_flashstyle


def _shifted(x, y, f, g, a, b, cost_scale):
    # Translate both clouds by their joint mass-weighted centroid, as SamplesLoss does: the plan is unchanged and
    # |x|^2 stays small, which the shifted potentials need in FP32.
    x, y, a, b = x.float(), y.float(), a.float(), b.float()
    mu = ((a[:, None] * x).sum(0) + (b[:, None] * y).sum(0)) / (a.sum() + b.sum())
    x, y = (x - mu).contiguous(), (y - mu).contiguous()
    f_hat = (f.float() - cost_scale * (x * x).sum(1)).contiguous()
    g_hat = (g.float() - cost_scale * (y * y).sum(1)).contiguous()
    return x, y, f_hat, g_hat, a.log().contiguous(), b.log().contiguous()


def _apply(x, y, f, g, a, b, v, eps, cost_scale, axis):
    xs, ys, f_hat, g_hat, log_a, log_b = _shifted(x, y, f, g, a, b, cost_scale)
    v = v.float().contiguous()
    kwargs = dict(eps=eps, axis=axis, cost_scale=cost_scale)
    if v.ndim == 1:
        return apply_plan_vec_flashstyle(xs, ys, f_hat, g_hat, log_a, log_b, v, **kwargs)
    if v.shape[1] == xs.shape[1]:
        # A plan is applied a few times here, so tuning the kernel's configuration would cost more than it saves.
        return apply_plan_mat_flashstyle(xs, ys, f_hat, g_hat, log_a, log_b, v, autotune=False, **kwargs)
    # The matrix kernel takes exactly d columns; other widths go one column at a time.
    columns = [apply_plan_vec_flashstyle(xs, ys, f_hat, g_hat, log_a, log_b, v[:, k].contiguous(), **kwargs)
               for k in range(v.shape[1])]
    return torch.stack(columns, dim=1)


def apply_plan(x, y, f, g, a, b, v, *, eps, cost_scale):
    """P @ v, for v of shape (m,) or (m, k)."""
    return _apply(x, y, f, g, a, b, v, eps, cost_scale, axis=1)


def apply_plan_transpose(x, y, f, g, a, b, u, *, eps, cost_scale):
    """P^T @ u, for u of shape (n,) or (n, k)."""
    return _apply(x, y, f, g, a, b, u, eps, cost_scale, axis=0)


def marginals(x, y, f, g, a, b, *, eps, cost_scale):
    """(P 1, P^T 1). For a converged balanced solve using the same costs, they approach a and b."""
    row = apply_plan(x, y, f, g, a, b, torch.ones(len(y), device=y.device), eps=eps, cost_scale=cost_scale)
    col = apply_plan_transpose(x, y, f, g, a, b, torch.ones(len(x), device=x.device), eps=eps,
                               cost_scale=cost_scale)
    return row, col


def barycentric_targets(x, y, f, g, a, b, *, eps, cost_scale):
    """(P y) / (P 1): the average position the plan sends each source point to."""
    row = apply_plan(x, y, f, g, a, b, torch.ones(len(y), device=y.device), eps=eps, cost_scale=cost_scale)
    return apply_plan(x, y, f, g, a, b, y, eps=eps, cost_scale=cost_scale) / row[:, None]
