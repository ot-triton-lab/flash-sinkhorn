from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import log_weights


def _grad_autotune_configs() -> Sequence[triton.Config]:
    configs = []
    for block_m, block_n in ((64, 64), (128, 64), (64, 128)):
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
def _geomloss_grad_sqeuclid_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    a_ptr,
    b_ptr,
    x2_ptr,
    y2_ptr,
    grad_scale_ptr,
    grad_x_ptr,
    grad_y_ptr,
    pid_offset,
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
    stride_a,
    stride_b,
    stride_x2,
    stride_y2,
    stride_grad_x0,
    stride_grad_x1,
    stride_grad_y0,
    stride_grad_y1,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid = tl.program_id(0) + pid_offset
    inv_eps = 1.0 / eps
    grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
    blocks_x = tl.cdiv(n, BLOCK_M)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps_log2 = inv_eps * log2e

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # grad_x: for each x_i, compute y_bar_i = E_{P(.|x_i)}[y] and return
    # grad_x_i = 2 * a_i * (x_i - y_bar_i) * grad_scale.
    if pid < blocks_x:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        a = tl.load(a_ptr + offs_m * stride_a, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(
            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
        ).to(tl.float32)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        for j0 in range(0, m, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < m

            g = tl.load(
                g_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_g,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            logb = tl.load(
                logb_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_logb,
                mask=mask_m[:, None] & mask_n[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            if USE_EXP2:
                logb = logb * log2e

            y2 = tl.load(
                y2_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_y2,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)

            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                xk = tl.load(
                    x_ptr
                    + offs_m[:, None] * stride_x0
                    + offs_k[None, :] * stride_x1,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                yk = tl.load(
                    y_ptr
                    + offs_n[None, :] * stride_y0
                    + offs_k[:, None] * stride_y1,
                    mask=mask_n[None, :] & mask_k[:, None],
                    other=0.0,
                )
                dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

            cost = x2[:, None] + y2 - 2.0 * dot
            if USE_EXP2:
                vals = tl.fma(g - cost, inv_eps_log2, logb)
            else:
                vals = (g - cost) * inv_eps + logb
            # Out-of-bounds columns already have logb=-inf from the masked load.

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            if USE_EXP2:
                alpha = tl.exp2(m_i - new_m)
                w = tl.exp2(vals - new_m[:, None])
            else:
                alpha = tl.exp(m_i - new_m)
                w = tl.exp(vals - new_m[:, None])

            l_i = l_i * alpha + tl.sum(w, axis=1)
            o = o * alpha[:, None]

            yv_t = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
                mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
                other=0.0,
            ).to(tl.float32)
            yv = tl.trans(yv_t)
            o += tl.dot(w, yv, allow_tf32=ALLOW_TF32)
            m_i = new_m

        y_bar = o / l_i[:, None]
        scale = (2.0 * grad_scale) * a
        grad = (x - y_bar) * scale[:, None]
        tl.store(
            grad_x_ptr + offs_m[:, None] * stride_grad_x0 + offs_d[None, :] * stride_grad_x1,
            grad,
            mask=mask_m[:, None] & mask_d[None, :],
        )
        return

    # grad_y: for each y_j, compute x_bar_j = E_{P(.|y_j)}[x] and return
    # grad_y_j = 2 * b_j * (y_j - x_bar_j) * grad_scale.
    pid_y = pid - blocks_x
    offs_n = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    b = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(
        y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
    ).to(tl.float32)

    y_t = tl.load(
        y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
        mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
        other=0.0,
    ).to(tl.float32)
    y = tl.trans(y_t)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    l_j = tl.zeros([BLOCK_N], tl.float32)
    o = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(
            f_ptr + offs_m[:, None] * stride_f + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        loga = tl.load(
            loga_ptr + offs_m[:, None] * stride_loga + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        if USE_EXP2:
            loga = loga * log2e
        x2 = tl.load(
            x2_ptr + offs_m[:, None] * stride_x2 + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            xk = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            yk = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)

        cost = x2 + y2[None, :] - 2.0 * dot
        if USE_EXP2:
            vals = tl.fma(f - cost, inv_eps_log2, loga)
        else:
            vals = (f - cost) * inv_eps + loga
        # Out-of-bounds rows already have loga=-inf from the masked load.

        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        if USE_EXP2:
            alpha = tl.exp2(m_j - new_m)
            w = tl.exp2(vals - new_m[None, :])
        else:
            alpha = tl.exp(m_j - new_m)
            w = tl.exp(vals - new_m[None, :])

        l_j = l_j * alpha + tl.sum(w, axis=0)
        o = o * alpha[:, None]

        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        o += tl.dot(tl.trans(w), xv, allow_tf32=ALLOW_TF32)
        m_j = new_m

    x_bar = o / l_j[:, None]
    scale = (2.0 * grad_scale) * b
    grad = (y - x_bar) * scale[:, None]
    tl.store(
        grad_y_ptr + offs_n[:, None] * stride_grad_y0 + offs_d[None, :] * stride_grad_y1,
        grad,
        mask=mask_n[:, None] & mask_d[None, :],
    )


def _default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    if dtype == torch.float32 and not allow_tf32:
        block_m = 64
        block_n = 64
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


_geomloss_grad_sqeuclid_autotune = triton.autotune(
    configs=_grad_autotune_configs(),
    key=["D", "ALLOW_TF32", "DTYPE_ID"],
)(_geomloss_grad_sqeuclid_impl)


def sinkhorn_geomloss_online_grad_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
    grad_scale: Optional[torch.Tensor] = None,
    compute_grad_x: bool = True,
    compute_grad_y: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D tensors.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if a.shape[0] != n or f.shape[0] != n:
        raise ValueError("a and f shapes must match x.")
    if b.shape[0] != m or g.shape[0] != m:
        raise ValueError("b and g shapes must match y.")

    x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous()
    loga = log_weights(a).contiguous()
    logb = log_weights(b).contiguous()

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

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

    # Triton requires tl.arange ranges to be powers of 2. Pad the feature axis.
    block_d = max(16, 1 << (int(d) - 1).bit_length())

    blocks_x = triton.cdiv(n, bm)
    blocks_y = triton.cdiv(m, bn)

    compute_grad_x = bool(compute_grad_x)
    compute_grad_y = bool(compute_grad_y)
    if not compute_grad_x and not compute_grad_y:
        return None, None

    if compute_grad_x and compute_grad_y:
        pid_offset = 0
        grid = (blocks_x + blocks_y,)
    elif compute_grad_x and not compute_grad_y:
        pid_offset = 0
        grid = (blocks_x,)
    else:
        # grad_y only: requires pid offset by blocks_x. Autotune can't vary this
        # with meta["BLOCK_M"], so keep a single (manual) config.
        if autotune:
            autotune = False
        pid_offset = blocks_x
        grid = (blocks_y,)

    grad_x = (
        torch.empty((n, d), device=x.device, dtype=torch.float32)
        if compute_grad_x
        else torch.empty((1, 1), device=x.device, dtype=torch.float32)
    )
    grad_y = (
        torch.empty((m, d), device=x.device, dtype=torch.float32)
        if compute_grad_y
        else torch.empty((1, 1), device=x.device, dtype=torch.float32)
    )

    if grad_scale is None:
        grad_scale = torch.ones((), device=x.device, dtype=torch.float32)
    else:
        if grad_scale.numel() != 1:
            raise ValueError("grad_scale must be a scalar tensor.")
        if grad_scale.device != x.device:
            raise ValueError("grad_scale must be on the same device as x.")
        grad_scale = grad_scale.to(dtype=torch.float32)

    kernel = _geomloss_grad_sqeuclid_autotune if autotune else _geomloss_grad_sqeuclid_impl
    if autotune:
        # Autotune uses a callable grid.
        def grid_fn(meta):
            if compute_grad_x and compute_grad_y:
                return (
                    triton.cdiv(n, meta["BLOCK_M"]) + triton.cdiv(m, meta["BLOCK_N"]),
                )
            # grad_x only, pid_offset=0.
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        kernel[grid_fn](
            x,
            y,
            f,
            g,
            loga,
            logb,
            a,
            b,
            x2,
            y2,
            grad_scale,
            grad_x,
            grad_y,
            pid_offset,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            loga.stride(0),
            logb.stride(0),
            a.stride(0),
            b.stride(0),
            x2.stride(0),
            y2.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_y.stride(0),
            grad_y.stride(1),
            float(eps),
            D=d,
            BLOCK_D=block_d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )
    else:
        kernel[grid](
            x,
            y,
            f,
            g,
            loga,
            logb,
            a,
            b,
            x2,
            y2,
            grad_scale,
            grad_x,
            grad_y,
            pid_offset,
            n,
            m,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            loga.stride(0),
            logb.stride(0),
            a.stride(0),
            b.stride(0),
            x2.stride(0),
            y2.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_y.stride(0),
            grad_y.stride(1),
            float(eps),
            D=d,
            BLOCK_D=block_d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            USE_EXP2=use_exp2,
            num_warps=nw,
            num_stages=num_stages,
        )

    out_x = grad_x if compute_grad_x else None
    out_y = grad_y if compute_grad_y else None
    return out_x, out_y
