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
    """Default block sizes for OTT forward kernel.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.

    The key constraint is: BLOCK_K < D (must have at least 2 k iterations).
    - d >= 64: block_k = 32 → at least 2 iterations
    - d >= 32: block_k = 16 → at least 2 iterations
    - d < 32:  block_k = 16 → minimum for tl.dot
    """
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

    # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

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


# ==============================================================================
# FUSED UPDATE POTENTIAL KERNELS
# ==============================================================================
# These kernels output the final potential directly, eliminating PyTorch overhead.
#
# Formula for f update (axis=1): f_new[i] = eps * loga[i] - eps * logsumexp_j((g[j] - C_ij)/eps)
# Formula for g update (axis=0): g_new[j] = eps * logb[j] - eps * logsumexp_i((f[i] - C_ij)/eps)
#
# Note: The log marginals (loga, logb) are NOT inside the logsumexp - they only
# appear in the output formula. This is because when the original code computes
# logsumexp((f+g-C)/eps) and then subtracts g (or f), the g (or f) factors out.


@triton.jit
def _update_potential_axis1_kernel(
    x_ptr,
    y_ptr,
    g_ptr,              # Input: g potential (NOT f - we only need g for f update)
    log_marginal_ptr,   # loga: log marginal for OUTPUT only
    x2_ptr,
    y2_ptr,
    out_ptr,            # Output: f_new potential (NOT raw lse)
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_g,
    stride_log_marginal,
    stride_x2,
    stride_y2,
    stride_out,
    eps,
    D: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute f_new[i] = eps * loga[i] - eps * logsumexp_j((g[j] - C_ij)/eps)

    Note: log marginals (loga, logb) are NOT used inside the logsumexp.
    They only appear in the output formula.
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    # Load x² for this block of rows
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    # Online max-shift accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453

    # Iterate over all j (columns)
    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load g potential for this block (NO log marginal in logsumexp!)
        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

        # Compute dot product x·y
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

        # Compute cost C_ij = ||x_i||² + ||y_j||² - 2*x_i·y_j
        cost = x2[:, None] + y2[None, :] - 2.0 * dot

        # Logits for f update: (g[j] - C_ij) / eps
        # Note: NO log marginal here - it only appears in the output formula
        logits = (g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        # Online max-shift update
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

    # Compute final logsumexp
    if USE_EXP2:
        lse = m_i + tl.log2(s_i)
        lse = lse * ln2  # Convert from log2 to ln
    else:
        lse = m_i + tl.log(s_i)

    # Load log marginal (loga) and compute final potential
    log_marginal = tl.load(log_marginal_ptr + offs_m * stride_log_marginal, mask=mask_m, other=0.0).to(tl.float32)

    # f_new = eps * loga - eps * lse
    # Handle -inf case: if s_i is 0 (no valid entries), output 0
    potential = tl.where(s_i > 0, eps * log_marginal - eps * lse, tl.zeros([BLOCK_M], tl.float32))
    tl.store(out_ptr + offs_m * stride_out, potential, mask=mask_m)


@triton.jit
def _update_potential_axis0_kernel(
    x_ptr,
    y_ptr,
    f_ptr,              # Input: f potential (NOT g - we only need f for g update)
    log_marginal_ptr,   # logb: log marginal for OUTPUT only
    x2_ptr,
    y2_ptr,
    out_ptr,            # Output: g_new potential (NOT raw lse)
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_log_marginal,
    stride_x2,
    stride_y2,
    stride_out,
    eps,
    D: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute g_new[j] = eps * logb[j] - eps * logsumexp_i((f[i] - C_ij)/eps)

    Note: log marginals (loga, logb) are NOT used inside the logsumexp.
    They only appear in the output formula.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    inv_eps = 1.0 / eps

    # Load y² for this block of columns
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    # Online max-shift accumulators
    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453

    # Iterate over all i (rows)
    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        # Load f potential for this block (NO log marginal in logsumexp!)
        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        # Compute dot product x·y
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

        # Compute cost C_ij = ||x_i||² + ||y_j||² - 2*x_i·y_j
        cost = x2[:, None] + y2[None, :] - 2.0 * dot

        # Logits for g update: (f[i] - C_ij) / eps
        # Note: NO log marginal here - it only appears in the output formula
        logits = (f[:, None] - cost) * inv_eps
        logits = tl.where(mask_m[:, None], logits, -float("inf"))

        # Online max-shift update (reduce over axis=0, i.e., over i)
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

    # Compute final logsumexp
    if USE_EXP2:
        lse = m_j + tl.log2(s_j)
        lse = lse * ln2  # Convert from log2 to ln
    else:
        lse = m_j + tl.log(s_j)

    # Load log marginal (logb) and compute final potential
    log_marginal = tl.load(log_marginal_ptr + offs_n * stride_log_marginal, mask=mask_n, other=0.0).to(tl.float32)

    # g_new = eps * logb - eps * lse
    # Handle -inf case: if s_j is 0 (no valid entries), output 0
    potential = tl.where(s_j > 0, eps * log_marginal - eps * lse, tl.zeros([BLOCK_N], tl.float32))
    tl.store(out_ptr + offs_n * stride_out, potential, mask=mask_n)


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


# ==============================================================================
# AUTOTUNING CONFIGS FOR FUSED UPDATE POTENTIAL KERNELS
# ==============================================================================
# These configs are optimized for the fused kernels which output potential directly.
# No shared memory constraint (output is scalar per row/col), so we can use larger blocks.

def _update_potential_autotune_configs_axis1() -> list[triton.Config]:
    """Autotuning configs for fused f-update kernel (axis=1).

    Key insight: axis=1 reduces over j (columns), so BLOCK_N is the "reduction dimension".
    Larger BLOCK_N = fewer loop iterations but more registers for dot accumulation.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.
    """
    configs = []

    # Standard configs for small-to-medium d
    # IMPORTANT: Use BLOCK_K <= 32 to avoid BLOCK_K >= D bug
    for block_m in (128, 64):
        for block_n in (128, 64):
            for block_k in (32, 16):  # Removed 64 for safety
                for num_warps in (8, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_warps=num_warps,
                            num_stages=3,
                        )
                    )

    # Configs for large d (512+): smaller BLOCK_K to fit more k-iterations
    for block_m in (64, 32):
        for block_n in (64, 32):
            for block_k in (32, 16):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                        num_warps=4,
                        num_stages=2,
                    )
                )

    return configs


def _update_potential_autotune_configs_axis0() -> list[triton.Config]:
    """Autotuning configs for fused g-update kernel (axis=0).

    Key insight: axis=0 reduces over i (rows), so BLOCK_M is the "reduction dimension".
    Keep BLOCK_N moderate since we store m_j/s_j accumulators of length BLOCK_N.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.
    """
    configs = []

    # Standard configs: keep BLOCK_N <= 64 for register pressure
    # IMPORTANT: Use BLOCK_K <= 32 to avoid BLOCK_K >= D bug
    for block_m in (128, 64):
        for block_n in (64, 32):
            for block_k in (32, 16):  # Removed 64 for safety
                for num_warps in (4, 8):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_warps=num_warps,
                            num_stages=3,
                        )
                    )

    # Configs for large d (512+): smaller blocks
    for block_m in (64, 32):
        for block_n in (32, 64):
            for block_k in (32, 16):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                        num_warps=4,
                        num_stages=2,
                    )
                )

    return configs


# Create autotuned versions of fused update kernels
_update_potential_axis1_kernel_autotune = triton.autotune(
    configs=_update_potential_autotune_configs_axis1(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_update_potential_axis1_kernel)

_update_potential_axis0_kernel_autotune = triton.autotune(
    configs=_update_potential_autotune_configs_axis0(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_update_potential_axis0_kernel)


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
    """Original update_potential (kept for backward compatibility).

    Uses apply_lse_kernel_sqeuclid + PyTorch ops. Prefer update_potential_fused
    for better performance.
    """
    lse, _ = apply_lse_kernel_sqeuclid(x, y, f, g, eps, axis, **kwargs)
    safe_lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    return eps * log_marginal - safe_lse


def update_potential_fused(
    x: torch.Tensor,
    y: torch.Tensor,
    potential_in: torch.Tensor,
    log_marginal: torch.Tensor,
    eps: float,
    axis: int,
    *,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,  # Enable autotuning by default for optimal performance
) -> torch.Tensor:
    """Compute potential update in a single fused Triton kernel.

    This eliminates ~10 PyTorch kernel launches per update by computing
    the final potential directly in the Triton kernel.

    Args:
        x: Source points, shape (n, d)
        y: Target points, shape (m, d)
        potential_in: Input potential (f for axis=0/g-update, g for axis=1/f-update)
        log_marginal: Log marginal for OUTPUT (logb for axis=0, loga for axis=1)
        eps: Regularization strength
        axis: 0 for g update, 1 for f update
        x2: Precomputed ||x||², optional
        y2: Precomputed ||y||², optional
        allow_tf32: Enable TF32 for matmul
        autotune: Enable autotuning for optimal block sizes (default: True)

    Returns:
        Updated potential:
        - axis=1: f_new[i] = eps*loga[i] - eps*logsumexp_j((g[j]-C_ij)/eps)
        - axis=0: g_new[j] = eps*logb[j] - eps*logsumexp_i((f[i]-C_ij)/eps)

    Note:
        The log marginals are NOT used inside the logsumexp (they factor out).
        They only appear in the output formula.
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("update_potential_fused requires CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")

    eps = float(eps)

    # Precompute squared norms if not provided
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1)
    else:
        x2 = x2.float()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1)
    else:
        y2 = y2.float()

    # Ensure inputs are float32 and contiguous
    potential_in = potential_in.float().contiguous()
    log_marginal = log_marginal.float().contiguous()

    # Determine dtype_id for kernel
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    # Determine whether to use autotuning
    user_specified_tiles = (
        block_m is not None or block_n is not None or block_k is not None
    )
    user_specified_launch = num_warps is not None or num_stages is not None
    use_autotune = bool(autotune) and not user_specified_tiles and not user_specified_launch

    if axis == 1:
        # f update: output shape is (n,)
        out = torch.empty((n,), device=x.device, dtype=torch.float32)

        if use_autotune:
            # Use autotuned kernel
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            _update_potential_axis1_kernel_autotune[grid](
                x,
                y,
                potential_in,  # g potential
                log_marginal,  # loga (output marginal only)
                x2,
                y2,
                out,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                potential_in.stride(0),
                log_marginal.stride(0),
                x2.stride(0),
                y2.stride(0),
                out.stride(0),
                eps,
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
            )
        else:
            # Manual block sizes
            if num_warps is None:
                num_warps = 4
            if num_stages is None:
                num_stages = 2

            # Tuned defaults for strict fp32 (no TF32)
            if (
                x.dtype == torch.float32
                and not allow_tf32
                and d >= 32
                and not user_specified_tiles
                and not user_specified_launch
            ):
                block_m = 128
                block_n = 128
                block_k = 32
                num_stages = 3
                num_warps = 8

            if block_m is None or block_n is None or block_k is None:
                block_m, block_n, block_k = _default_block_sizes(n, m, d)
            if block_k < 16:
                block_k = 16

            grid = (triton.cdiv(n, block_m),)
            _update_potential_axis1_kernel[grid](
                x,
                y,
                potential_in,  # g potential
                log_marginal,  # loga (output marginal only)
                x2,
                y2,
                out,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                potential_in.stride(0),
                log_marginal.stride(0),
                x2.stride(0),
                y2.stride(0),
                out.stride(0),
                eps,
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        # g update: output shape is (m,)
        out = torch.empty((m,), device=x.device, dtype=torch.float32)

        if use_autotune:
            # Use autotuned kernel
            def grid(meta):
                return (triton.cdiv(m, meta["BLOCK_N"]),)

            _update_potential_axis0_kernel_autotune[grid](
                x,
                y,
                potential_in,  # f potential
                log_marginal,  # logb (output marginal only)
                x2,
                y2,
                out,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                potential_in.stride(0),
                log_marginal.stride(0),
                x2.stride(0),
                y2.stride(0),
                out.stride(0),
                eps,
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
            )
        else:
            # Manual block sizes
            if num_warps is None:
                num_warps = 4
            if num_stages is None:
                num_stages = 2

            # Tuned defaults for strict fp32 (no TF32)
            if (
                x.dtype == torch.float32
                and not allow_tf32
                and d >= 32
                and not user_specified_tiles
                and not user_specified_launch
            ):
                block_m = 128
                block_n = 64
                block_k = 32
                num_stages = 3
                num_warps = 4

            if block_m is None or block_n is None or block_k is None:
                block_m, block_n, block_k = _default_block_sizes(n, m, d)
            if block_k < 16:
                block_k = 16

            grid = (triton.cdiv(m, block_n),)
            _update_potential_axis0_kernel[grid](
                x,
                y,
                potential_in,  # f potential
                log_marginal,  # logb (output marginal only)
                x2,
                y2,
                out,
                n,
                m,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                potential_in.stride(0),
                log_marginal.stride(0),
                x2.stride(0),
                y2.stride(0),
                out.stride(0),
                eps,
                D=d,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return out


def sinkhorn_potentials_sqeuclid(x, y, loga, logb, eps, n_iters, fused: bool = True, **kwargs):
    """Compute Sinkhorn potentials using alternating (Gauss-Seidel) updates.

    Args:
        x: Source points, shape (n, d)
        y: Target points, shape (m, d)
        loga: Log of source marginal, shape (n,)
        logb: Log of target marginal, shape (m,)
        eps: Regularization strength
        n_iters: Number of Sinkhorn iterations
        fused: If True (default), use fused kernels that eliminate PyTorch overhead.
               If False, use original implementation (for debugging/comparison).
        **kwargs: Additional arguments passed to kernels (allow_tf32, etc.)

    Returns:
        f, g: Converged potentials
    """
    n = x.shape[0]
    m = y.shape[0]
    f = torch.zeros((n,), device=x.device, dtype=torch.float32)
    g = torch.zeros((m,), device=y.device, dtype=torch.float32)
    loga = loga.float().contiguous()
    logb = logb.float().contiguous()
    x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous()

    if fused:
        # Use fused kernels: each update is a single Triton kernel launch
        # (eliminates ~10 PyTorch ops per update)
        for _ in range(n_iters):
            # g update: g_new = eps*logb - eps*logsumexp_i((f-C)/eps)
            g = update_potential_fused(
                x, y, f, logb, eps, axis=0, x2=x2, y2=y2, **kwargs
            )
            # f update: f_new = eps*loga - eps*logsumexp_j((g-C)/eps)
            f = update_potential_fused(
                x, y, g, loga, eps, axis=1, x2=x2, y2=y2, **kwargs
            )
    else:
        # Original implementation (for backward compatibility / debugging)
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
