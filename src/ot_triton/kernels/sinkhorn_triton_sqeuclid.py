import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _lse_axis1_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    sgn_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    stride_sgn,
    eps,
    D: tl.constexpr,
    HAS_VEC: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # FlashAttention-style: do the online reduction in log2 space using exp2/log2.
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

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
        logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        if HAS_VEC:
            vec = tl.load(vec_ptr + offs_n * stride_vec, mask=mask_n, other=0.0).to(
                tl.float32
            )
            vec_abs = tl.abs(vec)
            if USE_EXP2:
                log_vec = tl.where(vec_abs > 0, tl.log2(vec_abs), -float("inf"))
                vals = logits * log2e + log_vec[None, :]
            else:
                log_vec = tl.where(vec_abs > 0, tl.log(vec_abs), -float("inf"))
                vals = logits + log_vec[None, :]

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            vec_sign = tl.where(
                vec > 0, 1.0, tl.where(vec < 0, -1.0, 0.0)
            )
            if USE_EXP2:
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(
                    vec_sign[None, :] * tl.exp2(vals - new_m[:, None]), axis=1
                )
            else:
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(
                    vec_sign[None, :] * tl.exp(vals - new_m[:, None]), axis=1
                )
            m_i = new_m
        else:
            if USE_EXP2:
                logits2 = logits * log2e
                block_max = tl.max(logits2, axis=1)
                new_m = tl.maximum(m_i, block_max)
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(
                    tl.exp2(logits2 - new_m[:, None]), axis=1
                )
            else:
                block_max = tl.max(logits, axis=1)
                new_m = tl.maximum(m_i, block_max)
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(
                    tl.exp(logits - new_m[:, None]), axis=1
                )
            m_i = new_m

    if HAS_VEC:
        s_abs = tl.abs(s_i)
        lse = m_i + (tl.log2(s_abs) if USE_EXP2 else tl.log(s_abs))
        sgn = tl.where(s_i > 0, 1.0, tl.where(s_i < 0, -1.0, 0.0))
        lse = tl.where(s_abs > 0, lse, -float("inf"))
    else:
        lse = m_i + (tl.log2(s_i) if USE_EXP2 else tl.log(s_i))
        sgn = tl.full([BLOCK_M], 1.0, tl.float32)

    out = (eps * ln2) * lse if USE_EXP2 else eps * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)
    tl.store(sgn_ptr + offs_m * stride_sgn, sgn, mask=mask_m)


@triton.jit
def _lse_axis0_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    sgn_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    stride_sgn,
    eps,
    D: tl.constexpr,
    HAS_VEC: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    inv_eps = 1.0 / eps

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

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
        logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_m[:, None], logits, -float("inf"))

        if HAS_VEC:
            vec = tl.load(vec_ptr + offs_m * stride_vec, mask=mask_m, other=0.0).to(
                tl.float32
            )
            vec_abs = tl.abs(vec)
            if USE_EXP2:
                log_vec = tl.where(vec_abs > 0, tl.log2(vec_abs), -float("inf"))
                vals = logits * log2e + log_vec[:, None]
            else:
                log_vec = tl.where(vec_abs > 0, tl.log(vec_abs), -float("inf"))
                vals = logits + log_vec[:, None]

            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            vec_sign = tl.where(
                vec > 0, 1.0, tl.where(vec < 0, -1.0, 0.0)
            )
            if USE_EXP2:
                s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(
                    vec_sign[:, None] * tl.exp2(vals - new_m[None, :]), axis=0
                )
            else:
                s_j = s_j * tl.exp(m_j - new_m) + tl.sum(
                    vec_sign[:, None] * tl.exp(vals - new_m[None, :]), axis=0
                )
            m_j = new_m
        else:
            if USE_EXP2:
                logits2 = logits * log2e
                block_max = tl.max(logits2, axis=0)
                new_m = tl.maximum(m_j, block_max)
                s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(
                    tl.exp2(logits2 - new_m[None, :]), axis=0
                )
            else:
                block_max = tl.max(logits, axis=0)
                new_m = tl.maximum(m_j, block_max)
                s_j = s_j * tl.exp(m_j - new_m) + tl.sum(
                    tl.exp(logits - new_m[None, :]), axis=0
                )
            m_j = new_m

    if HAS_VEC:
        s_abs = tl.abs(s_j)
        lse = m_j + (tl.log2(s_abs) if USE_EXP2 else tl.log(s_abs))
        sgn = tl.where(s_j > 0, 1.0, tl.where(s_j < 0, -1.0, 0.0))
        lse = tl.where(s_abs > 0, lse, -float("inf"))
    else:
        lse = m_j + (tl.log2(s_j) if USE_EXP2 else tl.log(s_j))
        sgn = tl.full([BLOCK_N], 1.0, tl.float32)

    out = (eps * ln2) * lse if USE_EXP2 else eps * lse
    tl.store(out_ptr + offs_n * stride_out, out, mask=mask_n)
    tl.store(sgn_ptr + offs_n * stride_sgn, sgn, mask=mask_n)


def _default_block_sizes(n, m, d):
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    elif n >= 32:
        block_m = 32
    else:
        block_m = 16

    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    elif m >= 32:
        block_n = 32
    else:
        block_n = 16

    if d >= 64:
        block_k = 64
    elif d >= 32:
        block_k = 32
    else:
        block_k = 16

    return block_m, block_n, block_k


def _lse_autotune_configs_axis1() -> list[triton.Config]:
    # Guided (small) config set; tuned to work well across D in {32,64,128} on A100.
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ]


def _lse_autotune_configs_axis0() -> list[triton.Config]:
    # Keep BLOCK_N smaller here: axis=0 stores m_j/s_j of length BLOCK_N in registers.
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ]


_lse_axis1_kernel_autotune = triton.autotune(
    configs=_lse_autotune_configs_axis1(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "HAS_VEC", "USE_EXP2"],
)(_lse_axis1_kernel)

_lse_axis0_kernel_autotune = triton.autotune(
    configs=_lse_autotune_configs_axis0(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "HAS_VEC", "USE_EXP2"],
)(_lse_axis0_kernel)


def apply_lse_kernel_sqeuclid(
    x,
    y,
    f,
    g,
    eps,
    axis,
    vec=None,
    x2=None,
    y2=None,
    block_m=None,
    block_n=None,
    block_k=None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    backend: str = "triton",
    autotune: bool = False,
):
    if not x.is_cuda or not y.is_cuda or not f.is_cuda or not g.is_cuda:
        raise ValueError("apply_lse_kernel_sqeuclid requires CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D tensors.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")
    if vec is not None:
        if vec.ndim != 1:
            raise ValueError("vec must be a 1D tensor.")
        if axis == 1 and vec.shape[0] != m:
            raise ValueError("vec shape must match axis=1 reduction (m).")
        if axis == 0 and vec.shape[0] != n:
            raise ValueError("vec shape must match axis=0 reduction (n).")

    eps = float(eps)
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1)
    else:
        if x2.shape != (n,):
            raise ValueError("x2 must have shape (n,).")
        if x2.device != x.device:
            raise ValueError("x2 must be on the same device as x.")
        x2 = x2.float()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1)
    else:
        if y2.shape != (m,):
            raise ValueError("y2 must have shape (m,).")
        if y2.device != y.device:
            raise ValueError("y2 must be on the same device as y.")
        y2 = y2.float()
    f = f.float()
    g = g.float()

    backend = backend.lower()
    if backend != "triton":
        raise ValueError(
            f"Unknown/unsupported backend={backend!r}. The cuBLAS baseline was abandoned; "
            "use backend='triton'."
        )

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    user_specified_tiles = (
        block_m is not None or block_n is not None or block_k is not None
    )
    user_specified_launch = num_warps is not None or num_stages is not None
    use_autotune = bool(autotune) and not user_specified_tiles and not user_specified_launch

    if not use_autotune:
        if num_warps is None:
            num_warps = 4
        if num_stages is None:
            num_stages = 2

        # Tuned defaults for strict fp32 (no TF32): use smaller K and deeper pipelining.
        # Only apply when the caller didn't specify any tiling / launch params.
        if (
            x.dtype == torch.float32
            and not allow_tf32
            and d >= 32
            and not user_specified_tiles
            and not user_specified_launch
        ):
            block_m = 128
            block_k = 32
            num_stages = 3
            if axis == 1:
                block_n = 128
                num_warps = 8
            else:
                block_n = 64
                num_warps = 4

        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16

    if axis == 1:
        out = torch.empty((n,), device=x.device, dtype=torch.float32)
        sgn = torch.empty((n,), device=x.device, dtype=torch.float32)
        if vec is None:
            vec_ptr = x2
            has_vec = False
        else:
            vec = vec.float().contiguous()
            vec_ptr = vec
            has_vec = True

        if use_autotune:
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            _lse_axis1_kernel_autotune[grid](
                x,
                y,
                f,
                g,
                x2,
                y2,
                vec_ptr,
                out,
                sgn,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                x2.stride(0),
                y2.stride(0),
                vec_ptr.stride(0),
                out.stride(0),
                sgn.stride(0),
                eps,
                D=d,
                HAS_VEC=has_vec,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
            )
        else:
            grid = (triton.cdiv(n, block_m),)
            _lse_axis1_kernel[grid](
                x,
                y,
                f,
                g,
                x2,
                y2,
                vec_ptr,
                out,
                sgn,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                x2.stride(0),
                y2.stride(0),
                vec_ptr.stride(0),
                out.stride(0),
                sgn.stride(0),
                eps,
                D=d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                HAS_VEC=has_vec,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        remove = f
    else:
        out = torch.empty((m,), device=x.device, dtype=torch.float32)
        sgn = torch.empty((m,), device=x.device, dtype=torch.float32)
        if vec is None:
            vec_ptr = x2
            has_vec = False
        else:
            vec = vec.float().contiguous()
            vec_ptr = vec
            has_vec = True

        if use_autotune:
            def grid(meta):
                return (triton.cdiv(m, meta["BLOCK_N"]),)

            _lse_axis0_kernel_autotune[grid](
                x,
                y,
                f,
                g,
                x2,
                y2,
                vec_ptr,
                out,
                sgn,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                x2.stride(0),
                y2.stride(0),
                vec_ptr.stride(0),
                out.stride(0),
                sgn.stride(0),
                eps,
                D=d,
                HAS_VEC=has_vec,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
            )
        else:
            grid = (triton.cdiv(m, block_n),)
            _lse_axis0_kernel[grid](
                x,
                y,
                f,
                g,
                x2,
                y2,
                vec_ptr,
                out,
                sgn,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                f.stride(0),
                g.stride(0),
                x2.stride(0),
                y2.stride(0),
                vec_ptr.stride(0),
                out.stride(0),
                sgn.stride(0),
                eps,
                D=d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                HAS_VEC=has_vec,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        remove = g

    safe_remove = torch.where(torch.isfinite(remove), remove, torch.zeros_like(remove))
    out = out - safe_remove
    return out, sgn


def update_potential(x, y, f, g, log_marginal, eps, axis, **kwargs):
    lse, _ = apply_lse_kernel_sqeuclid(x, y, f, g, eps, axis, **kwargs)
    safe_lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    return eps * log_marginal - safe_lse


def sinkhorn_potentials_sqeuclid(x, y, loga, logb, eps, n_iters, **kwargs):
    n = x.shape[0]
    m = y.shape[0]
    f = torch.zeros((n,), device=x.device, dtype=torch.float32)
    g = torch.zeros((m,), device=y.device, dtype=torch.float32)
    loga = loga.float()
    logb = logb.float()
    x2 = (x.float() * x.float()).sum(dim=1)
    y2 = (y.float() * y.float()).sum(dim=1)

    for _ in range(n_iters):
        g = update_potential(
            x, y, f, g, logb, eps, axis=0, x2=x2, y2=y2, **kwargs
        )
        f = update_potential(
            x, y, f, g, loga, eps, axis=1, x2=x2, y2=y2, **kwargs
        )
    return f, g


def apply_transport_from_potentials_sqeuclid(
    x, y, f, g, vec, eps, axis, **kwargs
):
    lse_res, lse_sgn = apply_lse_kernel_sqeuclid(
        x, y, f, g, eps, axis, vec=vec, **kwargs
    )
    remove = f if axis == 1 else g
    lse_res = lse_res + remove
    return lse_sgn * torch.exp(lse_res / eps)
