from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

# Import from _common and re-export for backward compatibility
from ot_triton.kernels._common import epsilon_schedule, log_weights, max_diameter


def _geomloss_autotune_configs() -> Sequence[triton.Config]:
    configs = []
    for block_m, block_n in ((128, 64), (64, 128), (64, 64)):
        for block_k in (16, 32, 64):
            for num_warps in (4, 8):
                for num_stages in (2, 3):
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                            },
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.jit
def _geomloss_symmetric_step_sqeuclid_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    x2_ptr,
    y2_ptr,
    f_out_ptr,
    g_out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_loga,
    stride_logb,
    stride_x2,
    stride_y2,
    stride_f_out,
    stride_g_out,
    eps,
    alpha,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps
    blocks_f = tl.cdiv(n, BLOCK_M)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps_log2 = inv_eps * log2e

    if pid < blocks_f:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f_old = tl.load(
            f_ptr + offs_m * stride_f, mask=mask_m, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        for j0 in range(0, m, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < m

            g = tl.load(
                g_ptr + offs_n * stride_g, mask=mask_n, other=0.0
            ).to(tl.float32)
            logb = tl.load(
                logb_ptr + offs_n * stride_logb,
                mask=mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            if USE_EXP2:
                logb = logb * log2e
            y2 = tl.load(
                y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
            ).to(tl.float32)

            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                y = tl.load(
                    y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                    mask=mask_n[None, :] & mask_k[:, None],
                    other=0.0,
                )
                dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

            cost = x2[:, None] + y2[None, :] - 2.0 * dot
            if USE_EXP2:
                vals = tl.fma(g[None, :] - cost, inv_eps_log2, logb[None, :])
            else:
                vals = (g[None, :] - cost) * inv_eps + logb[None, :]
            vals = tl.where(mask_n[None, :], vals, -float("inf"))

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            if USE_EXP2:
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(
                    tl.exp2(vals - new_m[:, None]), axis=1
                )
            else:
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(
                    tl.exp(vals - new_m[:, None]), axis=1
                )
            m_i = new_m

        lse = (m_i + tl.log2(s_i)) * ln2 if USE_EXP2 else m_i + tl.log(s_i)
        cand = -eps * lse
        f_new = (1.0 - alpha) * f_old + alpha * cand
        tl.store(f_out_ptr + offs_m * stride_f_out, f_new, mask=mask_m)
        return

    pid_g = pid - blocks_f
    offs_n = pid_g * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    g_old = tl.load(
        g_ptr + offs_n * stride_g, mask=mask_n, other=0.0
    ).to(tl.float32)
    y2 = tl.load(
        y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
    ).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(
            f_ptr + offs_m * stride_f, mask=mask_m, other=0.0
        ).to(tl.float32)
        loga = tl.load(
            loga_ptr + offs_m * stride_loga,
            mask=mask_m,
            other=-float("inf"),
        ).to(tl.float32)
        if USE_EXP2:
            loga = loga * log2e
        x2 = tl.load(
            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
        ).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
        if USE_EXP2:
            vals = tl.fma(f[:, None] - cost, inv_eps_log2, loga[:, None])
        else:
            vals = (f[:, None] - cost) * inv_eps + loga[:, None]
        vals = tl.where(mask_m[:, None], vals, -float("inf"))

        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        if USE_EXP2:
            s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(
                tl.exp2(vals - new_m[None, :]), axis=0
            )
        else:
            s_j = s_j * tl.exp(m_j - new_m) + tl.sum(
                tl.exp(vals - new_m[None, :]), axis=0
            )
        m_j = new_m

    lse = (m_j + tl.log2(s_j)) * ln2 if USE_EXP2 else m_j + tl.log(s_j)
    cand = -eps * lse
    g_new = (1.0 - alpha) * g_old + alpha * cand
    tl.store(g_out_ptr + offs_n * stride_g_out, g_new, mask=mask_n)


def _default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    # Tuned default for strict fp32 (no TF32): favor smaller K and fewer warps.
    if dtype == torch.float32 and not allow_tf32:
        block_m = 128 if d >= 32 else 64
        block_n = 64 if d >= 32 else 64
        block_k = 32 if d >= 32 else 16
        num_warps = 4
        return block_m, block_n, block_k, num_warps

    block_m = 64
    block_n = 64
    if d >= 64:
        block_k = 64
    elif d >= 32:
        block_k = 32
    else:
        block_k = 16
    num_warps = 4 if block_k <= 32 else 8
    return block_m, block_n, block_k, num_warps


_geomloss_symmetric_step_sqeuclid_autotune = triton.autotune(
    configs=_geomloss_autotune_configs(),
    key=["D", "ALLOW_TF32", "DTYPE_ID"],
)(_geomloss_symmetric_step_sqeuclid_impl)

_geomloss_symmetric_step_sqeuclid = _geomloss_symmetric_step_sqeuclid_impl


def sinkhorn_geomloss_online_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float = 0.05,
    scaling: float = 0.5,
    use_epsilon_scaling: bool = True,
    last_extrapolation: bool = True,
    allow_tf32: bool = True,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    diameter: Optional[float] = None,
    eps_list: Optional[Sequence[float]] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
    use_exp2: bool = True,
    return_prelast: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if a.shape[0] != n or b.shape[0] != m:
        raise ValueError("a and b shapes must match x and y.")
    if a.device != x.device or b.device != x.device:
        raise ValueError("a and b must be on the same device as x and y.")

    if eps_list is None:
        if use_epsilon_scaling:
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = epsilon_schedule(diameter, blur, scaling, p=2.0)
        else:
            if eps is None or n_iters is None:
                raise ValueError(
                    "When use_epsilon_scaling=False, provide eps and n_iters."
                )
            eps_list = [float(eps)] * int(n_iters)

    if len(eps_list) == 0:
        raise ValueError("eps_list must be non-empty.")
    if n_iters is not None:
        eps_list = list(eps_list)[: int(n_iters)]
        if len(eps_list) == 0:
            raise ValueError("n_iters is 0 after slicing eps_list.")

    loga = log_weights(a).contiguous()
    logb = log_weights(b).contiguous()
    x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous()

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    f0 = torch.zeros((n,), device=x.device, dtype=torch.float32)
    g0 = torch.zeros((m,), device=x.device, dtype=torch.float32)
    f1 = torch.empty_like(f0)
    g1 = torch.empty_like(g0)

    def _launch_manual(f_in, g_in, f_out, g_out, step_eps: float, alpha: float):
        if block_m is None or block_n is None or block_k is None or num_warps is None:
            bm, bn, bk, nw = _default_block_sizes(d, x.dtype, allow_tf32)
            bm = bm if block_m is None else block_m
            bn = bn if block_n is None else block_n
            bk = bk if block_k is None else block_k
            nw = nw if num_warps is None else num_warps
        else:
            bm, bn, bk, nw = block_m, block_n, block_k, num_warps
        if bk < 16:
            bk = 16

        blocks_f = triton.cdiv(n, bm)
        blocks_g = triton.cdiv(m, bn)
        grid = (blocks_f + blocks_g,)

        _geomloss_symmetric_step_sqeuclid_impl[grid](
            x,
            y,
            f_in,
            g_in,
            loga,
            logb,
            x2,
            y2,
            f_out,
            g_out,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f_in.stride(0),
            g_in.stride(0),
            loga.stride(0),
            logb.stride(0),
            x2.stride(0),
            y2.stride(0),
            f_out.stride(0),
            g_out.stride(0),
            float(step_eps),
            float(alpha),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            USE_EXP2=use_exp2,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune(f_in, g_in, f_out, g_out, step_eps: float, alpha: float):
        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_M"])
                + triton.cdiv(m, meta["BLOCK_N"]),
            )

        _geomloss_symmetric_step_sqeuclid_autotune[grid](
            x,
            y,
            f_in,
            g_in,
            loga,
            logb,
            x2,
            y2,
            f_out,
            g_out,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f_in.stride(0),
            g_in.stride(0),
            loga.stride(0),
            logb.stride(0),
            x2.stride(0),
            y2.stride(0),
            f_out.stride(0),
            g_out.stride(0),
            float(step_eps),
            float(alpha),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )

    launch = _launch_autotune if use_autotune else _launch_manual

    # GeomLoss-style init at eps_list[0].
    launch(f0, g0, f1, g1, eps_list[0], alpha=1.0)
    f0, f1 = f1, f0
    g0, g1 = g1, g0

    # Symmetric updates, including eps_list[0] again (matches GeomLoss sinkhorn_loop).
    for step_eps in eps_list:
        launch(f0, g0, f1, g1, step_eps, alpha=0.5)
        f0, f1 = f1, f0
        g0, g1 = g1, g0

    if last_extrapolation:
        # Match GeomLoss's `sinkhorn_loop(last_extrapolation=True)` behavior:
        # do one final "full" update at the last epsilon value.
        launch(f0, g0, f1, g1, eps_list[-1], alpha=1.0)
        if return_prelast:
            return f1, g1, f0, g0
        return f1, g1

    if return_prelast:
        return f0, g0, f0, g0
    return f0, g0
