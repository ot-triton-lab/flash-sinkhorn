"""FlashSinkhorn: Sinkhorn OT with FlashAttention-style streaming and shifted potentials.

This module implements a reformulated Sinkhorn algorithm that aligns exactly with
FlashAttention's interface, enabling potential future integration with optimized
FlashAttention kernels.

Mathematical Reformulation
--------------------------

Standard Sinkhorn f-update:
    f_i = -ε · LSE_j[(g_j - C_ij)/ε + log(b_j)]

where C_ij = ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2·x_i·y_j = α_i + β_j - 2·x_i·y_j

Reformulation with shifted potentials:
1. Define: Q = √(2·cost_scale)·X, K = √(2·cost_scale)·Y
   So: Q_i·K_j = 2·cost_scale·x_i·y_j

2. Define shifted potentials:
   f̂ = f - α  where α_i = cost_scale·||x_i||²
   ĝ = g - β  where β_j = cost_scale·||y_j||²

3. Define pre-scaled biases:
   δ = ε·log(b)  (for f-update)
   γ = ε·log(a)  (for g-update)

4. The f-update becomes:
   f̂_i = -ε · LSE_j[(Q_i·K_j + ĝ_j + δ_j)/ε]
       = -ε · LSE_j[Q_i·K_j/ε + u_j]

   where u_j = (ĝ_j + δ_j)/ε is the pre-scaled bias

This matches FlashAttention exactly:
- Score matrix: S = Q·K^T
- Softmax scale: 1/ε
- Additive bias: u = (ĝ + δ)/ε

Benefits over current implementation:
- 67% fewer bias vector loads per tile (1 vs 3)
- ~78% fewer elementwise operations per element
- Direct FlashAttention interface alignment
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ot_triton.kernels._common import (
    _cache_key_bucket,
    dampening,
    epsilon_schedule,
    log_weights,
    max_diameter,
)


# =============================================================================
# PRECOMPUTATION UTILITIES
# =============================================================================

def precompute_flashsinkhorn_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cost_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute static bias components for FlashSinkhorn.

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full ||x-y||², 0.5 for half ||x-y||²/2)

    Returns:
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²
        gamma: Scaled log source weights [n] = eps * log(a) (for g-update)
        delta: Scaled log target weights [m] = eps * log(b) (for f-update)

    Notes:
        - For cost_scale=1.0: full squared Euclidean C = ||x-y||²
        - For cost_scale=0.5: half squared Euclidean C = ||x-y||²/2 (GeomLoss default)
        - Use flashsinkhorn_lse() with raw x, y coordinates (not pre-scaled Q, K)
    """
    # Squared norms with cost scaling
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)  # [n]
    beta = cost_scale * (y.float() ** 2).sum(dim=1)   # [m]

    # Scaled log marginals
    gamma = eps * log_weights(a)  # [n], for g-update
    delta = eps * log_weights(b)  # [m], for f-update

    return alpha, beta, gamma, delta


def compute_bias_f(
    g: torch.Tensor,
    beta: torch.Tensor,
    delta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute pre-scaled bias for f-update: u = (ĝ + δ)/ε.

    Args:
        g: Current g potential [m]
        beta: Target squared norms [m] = cost_scale * ||y||²
        delta: Scaled log target weights [m] = eps * log(b)
        eps: Regularization parameter

    Returns:
        u: Pre-scaled bias [m] = (g - beta + delta) / eps
    """
    g_hat = g - beta  # Shifted potential: ĝ = g - β
    return (g_hat + delta) / eps


def compute_bias_g(
    f: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute pre-scaled bias for g-update: v = (f̂ + γ)/ε.

    Args:
        f: Current f potential [n]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        gamma: Scaled log source weights [n] = eps * log(a)
        eps: Regularization parameter

    Returns:
        v: Pre-scaled bias [n] = (f - alpha + gamma) / eps
    """
    f_hat = f - alpha  # Shifted potential: f̂ = f - α
    return (f_hat + gamma) / eps


# =============================================================================
# TRITON KERNELS
# =============================================================================

def _flashsinkhorn_lse_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for FlashSinkhorn alternating LSE kernel.

    Key insight: Single pre-scaled bias vector per tile allows larger BLOCK_N.
    Similar to OTT-style _update_potential_autotune_configs_axis1().

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.
    """
    configs = []
    # Standard configs - include smaller block sizes for smaller n (like OTT-style)
    for block_m in (128, 64, 32):
        for block_n in (128, 64, 32):
            for block_k in (32, 16):  # Avoid 64 for BLOCK_K < D safety
                for num_warps in (8, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_warps=num_warps,
                            num_stages=3,
                        )
                    )

    # Additional configs for large d (512+) / small n: num_stages=2 reduces
    # register pressure and pipeline depth, better when launch overhead dominates
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


@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
    "EVEN_M": lambda args: args["n"] % args["BLOCK_M"] == 0,
})
@triton.jit
def _flashsinkhorn_lse_raw_kernel_impl(
    x_ptr,           # Source coordinates: [n, d] (NOT pre-scaled!)
    y_ptr,           # Target coordinates: [m, d] (NOT pre-scaled!)
    bias_ptr,        # Pre-scaled bias: [m] for f-update, [n] for g-update
    out_ptr,         # Output: shifted potential [n] or [m]
    n,               # Number of rows (sources for f-update, targets for g-update)
    m,               # Number of cols (targets for f-update, sources for g-update)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_bias,
    stride_out,
    coord_scale,     # 2 * cost_scale (to scale x @ y.T to match Q @ K.T)
    eps,             # Regularization parameter (for output scaling)
    damping,         # Unbalanced OT damping factor (1.0 for balanced)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Compute f̂_i = -ε · damping · LSE_j[coord_scale * x_i·y_j / ε + bias_j]

    This kernel computes a single shifted potential update using RAW coordinates.
    The coord_scale is applied INSIDE the kernel after the matmul, ensuring
    TF32 rounding behavior matches the fused kernel.

    KEY: Takes x, y directly (NOT pre-scaled Q, K). This ensures TF32 parity
    with the fused symmetric kernel which also uses raw coordinates.

    Args:
        x_ptr: Source coordinates [n, d] (NOT pre-scaled!)
        y_ptr: Target coordinates [m, d] (NOT pre-scaled!)
        bias_ptr: Pre-scaled bias [m] (or [n] for g-update)
        out_ptr: Output shifted potential [n] (or [m] for g-update)
        n, m: Dimensions
        coord_scale: 2 * cost_scale (to scale x·y to match Q·K)
        eps: Regularization parameter
        damping: Unbalanced OT factor (1.0 for balanced)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    # Online LSE accumulators (FlashAttention style)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # Constants for exp2/log2 optimization
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps = 1.0 / eps
    # Combined scale: coord_scale / eps (for x @ y.T -> scaled score / eps)
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Iterate over all columns (j dimension)
    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load ONLY the pre-scaled bias (instead of g, logb, y² in current impl)
        # This is the key memory savings: 1 vector instead of 3
        # CRITICAL: Must cast to float32 for consistent precision
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n * stride_bias, eviction_policy="evict_first").to(tl.float32)
        else:
            bias = tl.load(bias_ptr + offs_n * stride_bias, mask=mask_n, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
        # This ensures TF32 rounding matches the fused kernel
        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_block = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

        # Form logits: (coord_scale * x @ y.T) / eps + bias
        # = dot * (coord_scale / eps) + bias
        # This matches the fused kernel's computation exactly
        if USE_EXP2:
            # Use fma for better precision and potentially fused operations
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :] * log2e)
        else:
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # Online LSE update (standard FlashAttention algorithm)
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

    # Final LSE computation
    # Guard against s_i == 0 (can happen if all exp values underflow)
    s_i_safe = tl.maximum(s_i, 1e-40)
    if USE_EXP2:
        lse = (m_i + tl.log2(s_i_safe)) * ln2
    else:
        lse = m_i + tl.log(s_i_safe)

    # Output: f̂ = -ε * damping * LSE
    # For balanced OT, damping = 1.0
    # For unbalanced OT, damping = 1/(1 + eps/rho)
    out = -eps * damping * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


# Create autotuned version of raw LSE kernel (uses raw x, y coordinates)
_flashsinkhorn_lse_raw_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_lse_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_lse_raw_kernel_impl)


@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
    "EVEN_M": lambda args: args["n"] % args["BLOCK_M"] == 0,
})
@triton.jit
def _flashsinkhorn_lse_fused_kernel_impl(
    x_ptr,              # Source coordinates: [n, d]
    y_ptr,              # Target coordinates: [m, d]
    g_hat_ptr,          # Shifted potential: [m] for f-update (ĝ = g - β)
    log_w_ptr,          # Log marginal: [m] for f-update (log(b), NOT scaled!)
    out_ptr,            # Output: shifted potential [n]
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_out,
    coord_scale,        # 2 * cost_scale
    eps,                # Regularization parameter
    damping,            # Unbalanced OT damping factor (1.0 for balanced)
    CACHE_KEY_N,        # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,        # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Fused LSE kernel that computes bias in SRAM (matches symmetric kernel interface).

    This computes: f̂_i = -ε · damping · LSE_j[coord_scale * x_i·y_j / ε + ĝ_j/ε + log(w_j)]

    KEY OPTIMIZATION: Load ĝ and log(w) separately and compute bias = ĝ/ε + log(w) in SRAM.
    This matches the symmetric kernel interface and eliminates Python kernel launch overhead.

    Interface matches symmetric kernel:
    - Takes log_w directly (not eps*log(w))
    - Computes: bias = g_hat * inv_eps + log_w (same as symmetric)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    # Online LSE accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # Constants
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps = 1.0 / eps
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Iterate over columns
    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # FUSED BIAS COMPUTATION IN SRAM (matches symmetric kernel exactly):
        # Load ĝ and log(w), compute bias with exp2 pre-scaling if enabled
        if EVEN_N:
            g_hat = tl.load(g_hat_ptr + offs_n, eviction_policy="evict_first").to(tl.float32)
            log_w = tl.load(log_w_ptr + offs_n, eviction_policy="evict_first").to(tl.float32)
        else:
            g_hat = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first").to(tl.float32)
            log_w = tl.load(log_w_ptr + offs_n, mask=mask_n, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul
        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_block = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

        # Form logits: (coord_scale * dot)/ε + ĝ/ε + log(w)
        # PRE-SCALE bias for exp2 (matches symmetric kernel - avoids extra multiply)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = g_hat * inv_eps_log2 + log_w * log2e  # Pre-scaled for exp2
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :])  # Use directly
        else:
            bias = g_hat * inv_eps + log_w
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # Online LSE update
        block_max = tl.max(vals, axis=1)
        new_m = tl.maximum(m_i, block_max)
        if USE_EXP2:
            s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(tl.exp2(vals - new_m[:, None]), axis=1)
        else:
            s_i = s_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(vals - new_m[:, None]), axis=1)
        m_i = new_m

    # Final LSE
    s_i_safe = tl.maximum(s_i, 1e-40)
    if USE_EXP2:
        lse = (m_i + tl.log2(s_i_safe)) * ln2
    else:
        lse = m_i + tl.log(s_i_safe)

    out = -eps * damping * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


# Create autotuned version of fused LSE kernel
_flashsinkhorn_lse_fused_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_lse_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_lse_fused_kernel_impl)


# =============================================================================
# FUSED SYMMETRIC STEP KERNEL (single kernel for both f and g updates)
# =============================================================================

@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
    "EVEN_M": lambda args: args["n"] % args["BLOCK_M"] == 0,
})
@triton.jit
def _flashsinkhorn_symmetric_step_kernel(
    x_ptr,           # Source coordinates: [n, d] (NOT pre-scaled!)
    y_ptr,           # Target coordinates: [m, d] (NOT pre-scaled!)
    f_hat_ptr,       # Current shifted f potential: [n]
    g_hat_ptr,       # Current shifted g potential: [m]
    log_a_ptr,       # Log source weights: [n]
    log_b_ptr,       # Log target weights: [m]
    f_out_ptr,       # Output shifted f potential: [n]
    g_out_ptr,       # Output shifted g potential: [m]
    # OTDD label cost parameters (optional)
    label_x_ptr,     # int32 labels for x: [n] or dummy if not used
    label_y_ptr,     # int32 labels for y: [m] or dummy if not used
    W_ptr,           # Label cost matrix: [V, V] flattened or dummy
    n,
    m,
    V,               # Number of classes (1 if no label cost)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    eps,             # Regularization parameter
    alpha,           # Symmetric averaging weight (0.5 for symmetric, 1.0 for full update)
    damping_f,       # Unbalanced OT damping for f (1.0 for balanced)
    damping_g,       # Unbalanced OT damping for g (1.0 for balanced)
    coord_scale,     # 2 * cost_scale (to scale x @ y.T to match Q @ K.T)
    half_cost_scale, # cost_scale (for label cost: cost_scale * W)
    lambda_x,        # Weight for Euclidean cost (1.0 default)
    lambda_y,        # Weight for label cost (0.0 if no labels)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,  # For autotune cache consistency (harmonized with alternating)
    USE_EXP2: tl.constexpr,
    USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Fused symmetric Sinkhorn step: computes both f and g updates in one kernel.

    Grid structure: (blocks_f + blocks_g,) where blocks_f = cdiv(n, BLOCK_M)
    - Programs 0..blocks_f-1: compute f-update (LSE over m dimension)
    - Programs blocks_f..end: compute g-update (LSE over n dimension)

    This matches the GeomLoss kernel structure for maximum efficiency.

    KEY OPTIMIZATION: Uses x, y directly (no Q, K pre-allocation needed).
    The scale factor (2*cost_scale) is passed as coord_scale and applied in-kernel.

    Bias computation is done inline:
    - f-update: u = ĝ/ε + log(b)
    - g-update: v = f̂/ε + log(a)
    """
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps
    blocks_f = tl.cdiv(n, BLOCK_M)

    # Constants for exp2/log2 optimization
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    # Combined scale: coord_scale / eps (for xy @ xy.T -> QK^T / eps)
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # =========================================================================
    # F-UPDATE: first blocks_f programs
    # =========================================================================
    if pid < blocks_f:
        offs_i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_i = offs_i < n

        # Load current f_hat for symmetric averaging
        f_old = tl.load(f_hat_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)

        # Load labels for this block if using label cost
        if USE_LABEL_COST:
            label_i = tl.load(label_x_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)

        # Online LSE accumulators
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        # Iterate over all j (target points)
        for j0 in range(0, m, BLOCK_N):
            j0 = tl.multiple_of(j0, BLOCK_N)
            offs_j = j0 + tl.arange(0, BLOCK_N)
            mask_j = offs_j < m

            # Load ĝ and log(b) to compute bias: u = ĝ/ε + log(b)
            if EVEN_N:
                g_hat = tl.load(g_hat_ptr + offs_j, eviction_policy="evict_first").to(tl.float32)
                log_b = tl.load(log_b_ptr + offs_j, eviction_policy="evict_first").to(tl.float32)
            else:
                g_hat = tl.load(g_hat_ptr + offs_j, mask=mask_j, other=0.0, eviction_policy="evict_first").to(tl.float32)
                log_b = tl.load(log_b_ptr + offs_j, mask=mask_j, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

            # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                k0 = tl.multiple_of(k0, BLOCK_K)
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D
                x_block = tl.load(
                    x_ptr + offs_i[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                    mask=mask_i[:, None] & mask_k[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                y_block = tl.load(
                    y_ptr + offs_j[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                    mask=mask_j[None, :] & mask_k[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

            # Compute label cost if enabled
            # The cost term 2*cs*x·y contributes POSITIVELY to the exponent (lower cost = higher prob)
            # Label cost must be SUBTRACTED to increase cost for mismatched labels
            if USE_LABEL_COST:
                if EVEN_N:
                    label_j = tl.load(label_y_ptr + offs_j, eviction_policy="evict_first").to(tl.int32)
                else:
                    label_j = tl.load(label_y_ptr + offs_j, mask=mask_j, other=0, eviction_policy="evict_first").to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Label cost is SUBTRACTED (divided by eps) from the shifted dot product
                # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
                # Scale factors: coord_scale = 2*cost_scale, half_cost_scale = cost_scale
                effective_dot = lambda_x * dot * coord_scale - lambda_y * half_cost_scale * w_cost
            else:
                effective_dot = dot * coord_scale

            # Form logits: effective_dot/ε + ĝ/ε + log(b)
            if USE_EXP2:
                inv_eps_log2 = inv_eps * log2e
                bias = g_hat * inv_eps_log2 + log_b * log2e
                vals = tl.fma(effective_dot, inv_eps * log2e, bias[None, :])
            else:
                bias = g_hat * inv_eps + log_b
                vals = effective_dot * inv_eps + bias[None, :]
            vals = tl.where(mask_j[None, :], vals, -float("inf"))

            # Online LSE update
            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            if USE_EXP2:
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(tl.exp2(vals - new_m[:, None]), axis=1)
            else:
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(vals - new_m[:, None]), axis=1)
            m_i = new_m

        # Final LSE and output
        s_i_safe = tl.maximum(s_i, 1e-40)
        lse = (m_i + tl.log2(s_i_safe)) * ln2 if USE_EXP2 else m_i + tl.log(s_i_safe)
        f_cand = -eps * damping_f * lse
        f_new = (1.0 - alpha) * f_old + alpha * f_cand
        tl.store(f_out_ptr + offs_i, f_new, mask=mask_i)
        return

    # =========================================================================
    # G-UPDATE: remaining programs (pid >= blocks_f)
    # =========================================================================
    pid_g = pid - blocks_f
    offs_j = pid_g * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_j = offs_j < m

    # Load current g_hat for symmetric averaging
    g_old = tl.load(g_hat_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)

    # Load labels for this block if using label cost
    if USE_LABEL_COST:
        label_j = tl.load(label_y_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)

    # Online LSE accumulators
    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    # Iterate over all i (source points)
    for i0 in range(0, n, BLOCK_M):
        i0 = tl.multiple_of(i0, BLOCK_M)
        offs_i = i0 + tl.arange(0, BLOCK_M)
        mask_i = offs_i < n

        # Load f̂ and log(a) to compute bias: v = f̂/ε + log(a)
        if EVEN_M:
            f_hat = tl.load(f_hat_ptr + offs_i, eviction_policy="evict_first").to(tl.float32)
            log_a = tl.load(log_a_ptr + offs_i, eviction_policy="evict_first").to(tl.float32)
        else:
            f_hat = tl.load(f_hat_ptr + offs_i, mask=mask_i, other=0.0, eviction_policy="evict_first").to(tl.float32)
            log_a = tl.load(log_a_ptr + offs_i, mask=mask_i, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x_block = tl.load(
                x_ptr + offs_i[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_i[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_block = tl.load(
                y_ptr + offs_j[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_j[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

        # Compute label cost if enabled (same formula as f-update)
        if USE_LABEL_COST:
            if EVEN_M:
                label_i = tl.load(label_x_ptr + offs_i, eviction_policy="evict_first").to(tl.int32)
            else:
                label_i = tl.load(label_x_ptr + offs_i, mask=mask_i, other=0, eviction_policy="evict_first").to(tl.int32)
            # Compute flattened indices into W: W[label_i, label_j]
            w_idx = label_i[:, None] * V + label_j[None, :]
            # Gather from W (label cost matrix)
            w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
            # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
            effective_dot = lambda_x * dot * coord_scale - lambda_y * half_cost_scale * w_cost
        else:
            effective_dot = dot * coord_scale

        # Form logits: effective_dot/ε + f̂/ε + log(a) - reduce over i dimension
        # vals[i, j] = effective_dot_ij/ε + f̂_i/ε + log(a_i)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = f_hat * inv_eps_log2 + log_a * log2e
            vals = tl.fma(effective_dot, inv_eps * log2e, bias[:, None])
        else:
            bias = f_hat * inv_eps + log_a
            vals = effective_dot * inv_eps + bias[:, None]
        vals = tl.where(mask_i[:, None], vals, -float("inf"))

        # Online LSE update (reduce over axis=0, i.e., over i dimension)
        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        if USE_EXP2:
            s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(tl.exp2(vals - new_m[None, :]), axis=0)
        else:
            s_j = s_j * tl.exp(m_j - new_m) + tl.sum(tl.exp(vals - new_m[None, :]), axis=0)
        m_j = new_m

    # Final LSE and output
    s_j_safe = tl.maximum(s_j, 1e-40)
    lse = (m_j + tl.log2(s_j_safe)) * ln2 if USE_EXP2 else m_j + tl.log(s_j_safe)
    g_cand = -eps * damping_g * lse
    g_new = (1.0 - alpha) * g_old + alpha * g_cand
    tl.store(g_out_ptr + offs_j, g_new, mask=mask_j)


def _flashsinkhorn_symmetric_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for FlashSinkhorn symmetric fused kernel.

    Key insight: Fused f+g kernel. Keep BLOCK_N moderate for g-update accumulators.
    Similar to OTT-style _update_potential_autotune_configs_axis0().

    The g-update reduces over axis=0 (rows), storing accumulators of length BLOCK_N.

    NOTE: num_stages=3 is critical for large d (512+). Pipeline staging helps
    hide memory latency during k-iterations. Without stages=3, large d shows
    ~35% regression vs the OLD GeomLoss kernel.

    IMPORTANT: Include BLOCK_N=128 for large n (50k+) to match OLD kernel's configs.
    At n=50k, larger BLOCK_N reduces tile count and improves tensor core utilization.

    HARMONIZED with alternating kernel:
    - BLOCK_M: (128, 64, 32) - includes 32 for small n
    - BLOCK_N: (64, 32) standard + 128 for large n
    - DTYPE_ID in autotune key for consistent cache behavior
    """
    configs = []
    # Standard configs with moderate BLOCK_N for register pressure
    # Include BLOCK_M=32 for small n (harmonized with alternating kernel)
    for block_m in (128, 64, 32):
        for block_n in (64, 32):
            for block_k in (32, 16):
                for num_warps in (4, 8):
                    for num_stages in (2, 3):
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
    # Large-n configs with BLOCK_N=128 (matches OLD GeomLoss kernel)
    # Critical for n=50k+ where larger tiles reduce launch overhead
    for block_m in (128, 64):
        for block_k in (32, 16):
            for num_warps in (4, 8):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": 128, "BLOCK_K": block_k},
                        num_warps=num_warps,
                        num_stages=3,  # stages=3 for latency hiding
                    )
                )
    return configs


# Create autotuned version of symmetric step kernel
# NOTE: DTYPE_ID in key ensures separate autotune cache per dtype (harmonized with alternating)
_flashsinkhorn_symmetric_step_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_symmetric_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_symmetric_step_kernel)


def _flashsinkhorn_symmetric_autotune_configs_label() -> Sequence[triton.Config]:
    """Autotune configs optimized for label cost workloads (OTDD).

    Label cost requires scatter-gather access to W[label_x, label_y] every iteration.
    Larger BLOCK_K reduces k-loop iterations -> better W matrix cache reuse.
    Only valid for d >= 128 (need BLOCK_K < d for correctness).
    """
    # Use the fastest label config (same as old kernel)
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        )
    ]


# Separate autotune kernel for label workloads (with BLOCK_K=64 only)
_flashsinkhorn_symmetric_step_kernel_autotune_label = triton.autotune(
    configs=_flashsinkhorn_symmetric_autotune_configs_label(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_symmetric_step_kernel)


def _default_block_sizes_alternating(n: int, m: int, d: int) -> Tuple[int, int, int, int]:
    """Default block sizes for alternating LSE kernel (like OTT-style).

    Adapts to problem dimensions. Single bias load per tile allows larger BLOCK_N.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D causes incorrect results.
    """
    # BLOCK_M adapts to n
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    else:
        block_m = 32

    # BLOCK_N adapts to m - favor larger BN for single bias load
    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    else:
        block_n = 32

    # BLOCK_K based on d (BLOCK_K < D for correctness)
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

    num_warps = 4
    return block_m, block_n, block_k, num_warps


def _default_block_sizes_symmetric(n: int, m: int, d: int) -> Tuple[int, int, int, int]:
    """Default block sizes for symmetric fused kernel (like OTT-style axis=0).

    Keeps BLOCK_N smaller because g-update stores accumulators of length BLOCK_N.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    """
    # BLOCK_M adapts to n
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    else:
        block_m = 32

    # BLOCK_N smaller (g-update stores accumulators of length BLOCK_N)
    if m >= 64:
        block_n = 64
    else:
        block_n = 32

    # BLOCK_K based on d (BLOCK_K < D for correctness)
    if d >= 64:
        block_k = 32
    elif d >= 32:
        block_k = 16
    else:
        block_k = 16

    num_warps = 4
    return block_m, block_n, block_k, num_warps


def flashsinkhorn_lse(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    damping: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
) -> torch.Tensor:
    """Compute shifted potential using FlashSinkhorn kernel.

    This computes: out_i = -ε * damping * LSE_j[coord_scale * x_i·y_j / ε + bias_j]

    The kernel applies coord_scale = 2 * cost_scale inside the kernel to scale
    the dot product, ensuring consistent TF32 rounding between fused and separate
    kernel paths.

    Args:
        x: Source coordinates [n, d]
        y: Target coordinates [m, d]
        bias: Pre-scaled bias [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full ||x-y||², 0.5 for half ||x-y||²/2)
        damping: Unbalanced OT damping (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 for better numerical stability
        block_m, block_n, block_k: Manual block sizes (disables autotune)
        num_warps: Number of warps (disables autotune)
        num_stages: Number of pipeline stages
        autotune: Enable autotuning

    Returns:
        out: Shifted potential [n]
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if bias.ndim != 1:
        raise ValueError("bias must be 1D tensor")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension")
    if bias.shape[0] != m:
        raise ValueError(f"bias must have length {m}, got {bias.shape[0]}")

    # Ensure contiguous and float32
    x = x.contiguous().float()
    y = y.contiguous().float()
    bias = bias.contiguous().float()

    # Dtype ID for autotuning
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # coord_scale = 2 * cost_scale (to match QK^T = 2*cs*xy^T)
    coord_scale = 2.0 * cost_scale
    out = torch.empty((n,), device=x.device, dtype=torch.float32)

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    # Launcher helper functions
    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_alternating(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        grid = (triton.cdiv(n, bm),)
        _flashsinkhorn_lse_raw_kernel_impl[grid](
            x, y, bias, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        _flashsinkhorn_lse_raw_kernel_autotune[grid](
            x, y, bias, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )

    # Select and execute launcher
    launch = _launch_autotune if use_autotune else _launch_manual
    launch()

    return out


def flashsinkhorn_lse_fused(
    x: torch.Tensor,
    y: torch.Tensor,
    g_hat: torch.Tensor,
    log_w: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    damping: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
) -> torch.Tensor:
    """Fused LSE kernel that computes bias in SRAM (matches symmetric kernel interface).

    This computes: out_i = -ε * damping * LSE_j[coord_scale * x_i·y_j / ε + ĝ_j/ε + log(w_j)]

    KEY OPTIMIZATION: Load ĝ and log(w) separately and compute bias = ĝ/ε + log(w) in SRAM.
    This matches the symmetric kernel interface and eliminates Python kernel launch overhead.

    Args:
        x: Source coordinates [n, d]
        y: Target coordinates [m, d]
        g_hat: Shifted potential [m] (ĝ = g - β for f-update)
        log_w: Log marginal [m] (log(b) for f-update, NOT scaled by eps!)
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        damping: Unbalanced OT damping (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 for numerical stability
        block_m, block_n, block_k, num_warps: Manual block sizes
        num_stages: Pipeline stages
        autotune: Enable autotuning

    Returns:
        out: Shifted potential [n]
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension")
    if g_hat.shape[0] != m or log_w.shape[0] != m:
        raise ValueError(f"g_hat and log_w must have length {m}")

    # Ensure contiguous and float32
    x = x.contiguous().float()
    y = y.contiguous().float()
    g_hat = g_hat.contiguous().float()
    log_w = log_w.contiguous().float()

    # Dtype ID for autotuning
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    coord_scale = 2.0 * cost_scale
    out = torch.empty((n,), device=x.device, dtype=torch.float32)

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_alternating(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        grid = (triton.cdiv(n, bm),)
        _flashsinkhorn_lse_fused_kernel_impl[grid](
            x, y, g_hat, log_w, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        _flashsinkhorn_lse_fused_kernel_autotune[grid](
            x, y, g_hat, log_w, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )

    launch = _launch_autotune if use_autotune else _launch_manual
    launch()

    return out


def flashsinkhorn_symmetric_step(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    alpha: float = 0.5,
    damping_f: float = 1.0,
    damping_g: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    # OTDD label cost parameters
    label_x: Optional[torch.Tensor] = None,
    label_y: Optional[torch.Tensor] = None,
    label_cost_matrix: Optional[torch.Tensor] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused symmetric Sinkhorn step: computes both f and g updates in ONE kernel.

    This is the key optimization over separate f/g kernel calls - reduces kernel
    launch overhead by 50% and improves GPU occupancy.

    KEY: Uses x, y directly (no Q, K pre-allocation). The coord_scale = 2*cost_scale
    is applied inside the kernel, avoiding memory allocation overhead.

    Args:
        x: Source coordinates [n, d] (NOT pre-scaled!)
        y: Target coordinates [m, d] (NOT pre-scaled!)
        f_hat: Current shifted f potential [n]
        g_hat: Current shifted g potential [m]
        log_a: Log source weights [n]
        log_b: Log target weights [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full, 0.5 for half cost)
        alpha: Averaging weight (0.5 for symmetric, 1.0 for full update)
        damping_f: Unbalanced OT damping for f (1.0 for balanced)
        damping_g: Unbalanced OT damping for g (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable Triton autotuning (recommended)
        block_m, block_n, block_k: Manual block sizes (disables autotune)
        num_warps: Number of warps (disables autotune)
        num_stages: Pipeline stages
        label_x: int32/int64 labels for x [n] (OTDD)
        label_y: int32/int64 labels for y [m] (OTDD)
        label_cost_matrix: W [V, V] label distance matrix (OTDD)
        lambda_x: Weight for Euclidean cost (default 1.0)
        lambda_y: Weight for label cost (default 0.0 = no label cost)

    Returns:
        f_hat_new, g_hat_new: Updated shifted potentials
    """
    n, d = x.shape
    m = y.shape[0]

    # Ensure contiguous and float32
    x = x.contiguous().float()
    y = y.contiguous().float()
    f_hat = f_hat.contiguous().float()
    g_hat = g_hat.contiguous().float()
    log_a = log_a.contiguous().float()
    log_b = log_b.contiguous().float()

    # Output tensors
    f_out = torch.empty((n,), device=x.device, dtype=torch.float32)
    g_out = torch.empty((m,), device=x.device, dtype=torch.float32)

    # coord_scale = 2 * cost_scale (to match QK^T = 2*cost_scale * x·y)
    coord_scale = 2.0 * cost_scale
    half_cost_scale = cost_scale  # For label cost term

    # Dtype ID for autotuning (harmonized with alternating kernel)
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # OTDD label cost setup
    use_label_cost = (
        label_x is not None
        and label_y is not None
        and label_cost_matrix is not None
        and lambda_y != 0.0
    )

    if use_label_cost:
        # Validate label tensors
        if label_x.shape[0] != n:
            raise ValueError(f"label_x must have length n={n}, got {label_x.shape[0]}")
        if label_y.shape[0] != m:
            raise ValueError(f"label_y must have length m={m}, got {label_y.shape[0]}")
        if label_cost_matrix.ndim != 2:
            raise ValueError("label_cost_matrix must be 2D (V, V)")
        if label_cost_matrix.shape[0] != label_cost_matrix.shape[1]:
            raise ValueError("label_cost_matrix must be square (V, V)")

        V = label_cost_matrix.shape[0]

        # Validate label ranges
        if label_x.max() >= V or label_x.min() < 0:
            raise ValueError(f"label_x values must be in [0, {V}), got range [{label_x.min()}, {label_x.max()}]")
        if label_y.max() >= V or label_y.min() < 0:
            raise ValueError(f"label_y values must be in [0, {V}), got range [{label_y.min()}, {label_y.max()}]")

        # Ensure contiguous and correct dtype
        label_x_t = label_x.to(dtype=torch.int32, device=x.device).contiguous()
        label_y_t = label_y.to(dtype=torch.int32, device=x.device).contiguous()
        W = label_cost_matrix.to(dtype=torch.float32, device=x.device).contiguous()
    else:
        # Provide dummy tensors (size 1) to satisfy Triton's pointer requirements
        V = 1
        label_x_t = torch.zeros(1, dtype=torch.int32, device=x.device)
        label_y_t = torch.zeros(1, dtype=torch.int32, device=x.device)
        W = torch.zeros(1, dtype=torch.float32, device=x.device)

    # Check if manual block sizes were specified
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune_flag = autotune and not manual_blocks

    # Launcher helper functions (like GeomLoss-style)
    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_symmetric(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        # Grid: blocks_f + blocks_g (fused kernel handles both)
        blocks_f = triton.cdiv(n, bm)
        blocks_g = triton.cdiv(m, bn)
        grid = (blocks_f + blocks_g,)

        _flashsinkhorn_symmetric_step_kernel[grid](
            x, y,
            f_hat, g_hat,
            log_a, log_b,
            f_out, g_out,
            # OTDD label cost pointers
            label_x_t, label_y_t, W,
            n, m, V,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            float(eps),
            float(alpha),
            float(damping_f),
            float(damping_g),
            float(coord_scale),
            float(half_cost_scale),
            float(lambda_x),
            float(lambda_y),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            blocks_f = triton.cdiv(n, meta["BLOCK_M"])
            blocks_g = triton.cdiv(m, meta["BLOCK_N"])
            return (blocks_f + blocks_g,)

        # Select autotune kernel based on label cost usage
        kernel = (_flashsinkhorn_symmetric_step_kernel_autotune_label if use_label_cost
                  else _flashsinkhorn_symmetric_step_kernel_autotune)

        kernel[grid](
            x, y,
            f_hat, g_hat,
            log_a, log_b,
            f_out, g_out,
            # OTDD label cost pointers
            label_x_t, label_y_t, W,
            n, m, V,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            float(eps),
            float(alpha),
            float(damping_f),
            float(damping_g),
            float(coord_scale),
            float(half_cost_scale),
            float(lambda_x),
            float(lambda_y),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
        )

    # Select and execute launcher
    launch = _launch_autotune if use_autotune_flag else _launch_manual
    launch()

    return f_out, g_out


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def shifted_to_standard_potentials(
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert shifted potentials back to standard form.

    Args:
        f_hat: Shifted source potential [n]
        g_hat: Shifted target potential [m]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²

    Returns:
        f: Standard source potential [n] = f_hat + alpha
        g: Standard target potential [m] = g_hat + beta
    """
    return f_hat + alpha, g_hat + beta


def standard_to_shifted_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert standard potentials to shifted form.

    Args:
        f: Standard source potential [n]
        g: Standard target potential [m]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²

    Returns:
        f_hat: Shifted source potential [n] = f - alpha
        g_hat: Shifted target potential [m] = g - beta
    """
    return f - alpha, g - beta


# =============================================================================
# HIGH-LEVEL SOLVER FUNCTIONS
# =============================================================================

def sinkhorn_flashstyle_alternating(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 0.1,
    n_iters: int = 100,
    cost_scale: float = 1.0,
    # Unbalanced OT parameters (supports semi-unbalanced with separate x/y)
    rho: Optional[float] = None,  # Deprecated: use rho_x/rho_y for semi-unbalanced
    reach: Optional[float] = None,  # Deprecated: use reach_x/reach_y
    rho_x: Optional[float] = None,  # Source marginal penalty (None = strict/balanced)
    rho_y: Optional[float] = None,  # Target marginal penalty (None = strict/balanced)
    reach_x: Optional[float] = None,  # Alternative: reach_x^2 = rho_x
    reach_y: Optional[float] = None,  # Alternative: reach_y^2 = rho_y
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    ott_convention: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, int]:
    """FlashSinkhorn with alternating (Gauss-Seidel) updates.

    This uses the shifted potential formulation for reduced memory traffic
    and better FlashAttention alignment.

    Alternating Sinkhorn uses Gauss-Seidel updates where the g-update uses
    the NEWLY computed f potential. This requires 2 kernel launches per
    iteration (unlike symmetric which can fuse both updates).

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        eps: Regularization parameter
        n_iters: Number of Sinkhorn iterations
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        rho: Unbalanced OT marginal penalty (None for balanced) - DEPRECATED
        reach: Alternative to rho (reach^2 = rho) - DEPRECATED
        rho_x: Source marginal KL penalty (None = balanced/strict)
        rho_y: Target marginal KL penalty (None = balanced/strict)
        reach_x: Alternative to rho_x (reach_x^2 = rho_x)
        reach_y: Alternative to rho_y (reach_y^2 = rho_y)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        threshold: Early stopping threshold (None = no early stopping)
        check_every: Check convergence every N iterations
        return_n_iters: If True, also return number of iterations used
        ott_convention: If True, return potentials in OTT convention where
            log marginals are absorbed into potentials:
            - OTT: f = eps*log(a) - eps*LSE[(g-C)/eps], P = exp((f+g-C)/eps)
            If False (default), use GeomLoss convention:
            - GeomLoss: f = -eps*LSE[(g-C)/eps + log(b)], P = a*b*exp((f+g-C)/eps)

    Returns:
        f, g: Converged potentials (convention depends on ott_convention flag)
        n_iters_used: (optional) Number of iterations if return_n_iters=True

    Note:
        Both conventions produce EQUIVALENT transport plans. The only difference
        is whether log marginals are absorbed into the potentials or kept separate.

        OTT convention: P_ij = exp((f_i + g_j - C_ij) / eps)
        GeomLoss convention: P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)

        Semi-unbalanced OT:
        - Use rho_x, rho_y to set different marginal penalties for source/target
        - rho_x=None, rho_y=float gives strict source constraint, relaxed target
        - damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
    """
    n, d = x.shape
    m = y.shape[0]

    # Handle rho/reach for unbalanced OT (supports semi-unbalanced with separate x/y)
    # Priority: rho_x/rho_y > reach_x/reach_y > rho > reach
    if rho is not None and reach is not None:
        raise ValueError("Specify either rho or reach, not both.")
    if reach is not None:
        rho = reach ** 2  # GeomLoss convention: rho = reach^p where p=2

    # Handle legacy rho/reach: apply to both sides if new params not specified
    if rho is not None:
        if rho_x is None and reach_x is None:
            rho_x = rho
        if rho_y is None and reach_y is None:
            rho_y = rho

    # Handle reach_x/reach_y to rho_x/rho_y conversion
    if reach_x is not None:
        if rho_x is not None:
            raise ValueError("Specify either rho_x or reach_x, not both.")
        rho_x = reach_x ** 2
    if reach_y is not None:
        if rho_y is not None:
            raise ValueError("Specify either rho_y or reach_y, not both.")
        rho_y = reach_y ** 2

    # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
    # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
    #               rho_y controls TARGET marginal → damps g potential
    damp_f = dampening(eps, rho_x)
    damp_g = dampening(eps, rho_y)

    # Ensure float32 and contiguous for kernel
    x_f32 = x.float().contiguous()
    y_f32 = y.float().contiguous()

    # Precompute static bias components (NO Q, K allocation - use raw x, y!)
    # This ensures TF32 parity with the fused kernel
    alpha = cost_scale * (x_f32 ** 2).sum(dim=1)  # [n]
    beta = cost_scale * (y_f32 ** 2).sum(dim=1)   # [m]
    log_a = log_weights(a)  # [n], for g-update (NOT scaled by eps for fused kernel)
    log_b = log_weights(b)  # [m], for f-update (NOT scaled by eps for fused kernel)
    # For OTT convention (non-fused kernel), we need scaled versions
    gamma = eps * log_a  # [n], scaled for OTT
    delta = eps * log_b  # [m], scaled for OTT

    # Initialize shifted potentials: f̂ = f - α, ĝ = g - β
    # Standard init f=0, g=0 means f̂ = -α, ĝ = -β
    f_hat = -alpha.clone()
    g_hat = -beta.clone()

    prev_f_hat = f_hat.clone() if threshold is not None else None
    prev_g_hat = g_hat.clone() if threshold is not None else None

    n_iters_used = 0

    if ott_convention:
        # =================================================================
        # OTT CONVENTION: log marginals OUTSIDE the logsumexp
        # f = eps*log(a) - eps*LSE[(g-C)/eps]
        # g = eps*log(b) - eps*LSE[(f-C)/eps]
        #
        # Key difference from GeomLoss: potentials do NOT include ||x||², ||y||²
        # The squared norms only appear in the cost C, not in f or g.
        #
        # Math derivation (for f-update):
        #   (g - C)/eps = (g - alpha - beta + 2*cs*x·y)/eps
        #               = (g_hat + 2*cs*x·y)/eps - alpha/eps  [since g_hat = g - beta]
        #
        # Since alpha_i is constant w.r.t. j, it factors out of LSE:
        #   LSE_j[(g-C)/eps] = LSE_j[(g_hat + 2*cs*x·y)/eps] - alpha/eps
        #
        # Therefore:
        #   f = eps*log(a) - eps*LSE_j[(g-C)/eps]
        #     = gamma - eps*(LSE_j[...] - alpha/eps)
        #     = gamma - eps*LSE_j[...] + alpha
        #     = gamma + f_hat_lse + alpha
        #
        # But we want f (standard OTT potential), not f_hat. In OTT convention,
        # f does NOT include alpha, so we should NOT add alpha at the end.
        # This means f_ott = gamma + f_hat_lse (no alpha).
        # =================================================================
        for i in range(n_iters):
            # IMPORTANT: OTT-style does g-update FIRST, then f-update (Gauss-Seidel)
            # This matches sinkhorn_potentials_sqeuclid iteration order.

            # g-update FIRST: g = δ + (-ε * LSE_i[y·x^T * coord_scale/ε + f̂/ε]) + β
            # g_ott = delta + g_hat_lse + beta
            # We store g_hat = g_ott - beta, so g_hat = delta + g_hat_lse
            v = f_hat / eps  # Use current f_hat (shifted)
            g_hat_lse = flashsinkhorn_lse(
                y_f32, x_f32, v, eps, cost_scale=cost_scale,
                damping=damp_g,  # Semi-unbalanced: damp_g = 1/(1+eps/rho_x)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )
            g_hat = delta + g_hat_lse

            # f-update SECOND: f = γ + (-ε * LSE_j[x·y^T * coord_scale/ε + ĝ/ε]) + α
            # Uses the NEWLY computed g_hat (Gauss-Seidel!)
            u = g_hat / eps  # Bias is updated g_hat divided by eps
            f_hat_lse = flashsinkhorn_lse(
                x_f32, y_f32, u, eps, cost_scale=cost_scale,
                damping=damp_f,  # Semi-unbalanced: damp_f = 1/(1+eps/rho_y)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )
            # f_ott = gamma + f_hat_lse + alpha
            # We store f_hat = f_ott - alpha, so f_hat = gamma + f_hat_lse
            f_hat = gamma + f_hat_lse

            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (i + 1) % check_every == 0:
                f_change = (f_hat - prev_f_hat).abs().max().item()
                g_change = (g_hat - prev_g_hat).abs().max().item()
                if max(f_change, g_change) < threshold:
                    break
                prev_f_hat.copy_(f_hat)
                prev_g_hat.copy_(g_hat)

    else:
        # =================================================================
        # GEOMLOSS CONVENTION (default): log marginals INSIDE the logsumexp
        # f = -eps*LSE[(g-C)/eps + log(b)]
        # g = -eps*LSE[(f-C)/eps + log(a)]
        #
        # KEY: Uses FUSED kernel that computes bias in SRAM:
        #   bias = g_hat/eps + log_b  (same formula as symmetric kernel)
        # This eliminates Python kernel launch overhead.
        # =================================================================
        for i in range(n_iters):
            # f-update: f̂ = -ε * LSE_j[x·y^T * coord_scale/ε + ĝ/ε + log(b)]
            # FUSED: kernel computes bias = g_hat/eps + log_b in SRAM
            f_hat = flashsinkhorn_lse_fused(
                x_f32, y_f32, g_hat, log_b, eps, cost_scale=cost_scale,
                damping=damp_f,  # Semi-unbalanced: damp_f = 1/(1+eps/rho_y)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )

            # g-update: ĝ = -ε * LSE_i[y·x^T * coord_scale/ε + f̂/ε + log(a)]
            # FUSED: kernel computes bias = f_hat/eps + log_a in SRAM
            g_hat = flashsinkhorn_lse_fused(
                y_f32, x_f32, f_hat, log_a, eps, cost_scale=cost_scale,
                damping=damp_g,  # Semi-unbalanced: damp_g = 1/(1+eps/rho_x)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )

            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (i + 1) % check_every == 0:
                f_change = (f_hat - prev_f_hat).abs().max().item()
                g_change = (g_hat - prev_g_hat).abs().max().item()
                if max(f_change, g_change) < threshold:
                    break
                prev_f_hat.copy_(f_hat)
                prev_g_hat.copy_(g_hat)

    # Convert back to standard potentials
    if ott_convention:
        # OTT convention: potentials do NOT include ||x||², ||y||²
        # f_hat already contains the correct value (gamma + lse_result)
        # g_hat already contains the correct value (delta + lse_result)
        # But we need to add alpha/beta to match the shifted_to_standard formula
        # Actually no - for OTT, f_ott = gamma + lse + alpha, and we stored f_hat = gamma + lse
        # So we DO need to add alpha and beta to get the final OTT potentials!
        f, g = f_hat + alpha, g_hat + beta
    else:
        # GeomLoss convention: potentials include ||x||², ||y||² via shifted formulation
        f, g = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    if return_n_iters:
        return f, g, n_iters_used
    return f, g


def sinkhorn_flashstyle_symmetric(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float = 0.05,
    scaling: float = 0.5,
    use_epsilon_scaling: bool = True,
    last_extrapolation: bool = True,
    cost_scale: float = 1.0,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    diameter: Optional[float] = None,
    eps_list: Optional[Sequence[float]] = None,
    # Unbalanced OT parameters (supports semi-unbalanced with separate x/y)
    rho: Optional[float] = None,  # Deprecated: use rho_x/rho_y for semi-unbalanced
    reach: Optional[float] = None,  # Deprecated: use reach_x/reach_y
    rho_x: Optional[float] = None,  # Source marginal penalty (None = strict/balanced)
    rho_y: Optional[float] = None,  # Target marginal penalty (None = strict/balanced)
    reach_x: Optional[float] = None,  # Alternative: reach_x^2 = rho_x
    reach_y: Optional[float] = None,  # Alternative: reach_y^2 = rho_y
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    fused: bool = True,
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    return_prelast: bool = False,
    # Warm-start parameters (standard potentials, not shifted)
    f_init: Optional[torch.Tensor] = None,  # Initial f potential [n]
    g_init: Optional[torch.Tensor] = None,  # Initial g potential [m]
    # OTDD label-augmented cost parameters
    label_x: Optional[torch.Tensor] = None,  # int32/int64 labels for x: [n]
    label_y: Optional[torch.Tensor] = None,  # int32/int64 labels for y: [m]
    label_cost_matrix: Optional[torch.Tensor] = None,  # W: [V, V] label distances
    lambda_x: float = 1.0,  # Weight for Euclidean cost
    lambda_y: float = 0.0,  # Weight for label cost (0 = Euclidean only)
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, int]:
    """FlashSinkhorn with GeomLoss-style symmetric updates.

    This uses the shifted potential formulation for reduced memory traffic
    and epsilon scaling for numerical stability.

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        blur: Target blur (final eps = blur^2)
        scaling: Epsilon decay factor per iteration
        use_epsilon_scaling: If True, use exponential epsilon schedule
        last_extrapolation: If True, do final full update
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        eps: Fixed regularization (if not using epsilon scaling)
        n_iters: Number of iterations (if not using epsilon scaling)
        diameter: Point cloud diameter (auto-computed if None)
        eps_list: Explicit epsilon schedule (overrides other params)
        rho: Unbalanced OT marginal penalty (None for balanced) - DEPRECATED
        reach: Alternative to rho (reach^2 = rho) - DEPRECATED
        rho_x: Source marginal KL penalty (None = balanced/strict)
        rho_y: Target marginal KL penalty (None = balanced/strict)
        reach_x: Alternative to rho_x (reach_x^2 = rho_x)
        reach_y: Alternative to rho_y (reach_y^2 = rho_y)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        fused: If True (default), use single fused kernel for both f and g updates
               (1 kernel launch per iteration).
               If False, use separate kernels (2 launches per iteration).
               Auto-switches to False when n >= 30000 (separate is ~10% faster at scale).
        threshold: Early stopping threshold
        check_every: Check convergence every N iterations
        return_n_iters: If True, also return number of iterations used
        return_prelast: If True, also return pre-extrapolation potentials
        f_init: Initial f potential for warm-start (standard form, not shifted)
        g_init: Initial g potential for warm-start (standard form, not shifted)

    Returns:
        f, g: Converged potentials in standard form
        f_prelast, g_prelast: (optional) Pre-extrapolation potentials
        n_iters_used: (optional) Number of iterations

    Note:
        Fused vs Separate kernels:
        - Fused: 1 kernel launch per iteration, both f and g computed in parallel
        - Separate: 2 kernel launches per iteration
        - Both produce identical results (symmetric averaging uses old potentials)
        - Fused has 50% fewer kernel launches (better for small n < 30000)
        - Separate has better memory patterns at large n (auto-switches at n >= 30000)

        Semi-unbalanced OT:
        - Use rho_x, rho_y to set different marginal penalties for source/target
        - rho_x=None, rho_y=float gives strict source constraint, relaxed target
        - damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)

        OTDD Label Cost (requires fused=True):
        - label_x: int32/int64 labels for source points [n]
        - label_y: int32/int64 labels for target points [m]
        - label_cost_matrix: W [V, V] matrix of label distances
        - lambda_x: weight for Euclidean cost (default 1.0)
        - lambda_y: weight for label cost (default 0.0)
        - Combined cost: C = lambda_x * ||x-y||² + lambda_y * W[label_x, label_y]
    """
    n, d = x.shape
    m = y.shape[0]

    # Handle rho/reach for unbalanced OT (supports semi-unbalanced with separate x/y)
    # Priority: rho_x/rho_y > reach_x/reach_y > rho > reach
    if rho is not None and reach is not None:
        raise ValueError("Specify either rho or reach, not both.")
    if reach is not None:
        rho = reach ** 2  # GeomLoss convention: rho = reach^p where p=2

    # Handle legacy rho/reach: apply to both sides if new params not specified
    if rho is not None:
        if rho_x is None and reach_x is None:
            rho_x = rho
        if rho_y is None and reach_y is None:
            rho_y = rho

    # Handle reach_x/reach_y to rho_x/rho_y conversion
    if reach_x is not None:
        if rho_x is not None:
            raise ValueError("Specify either rho_x or reach_x, not both.")
        rho_x = reach_x ** 2
    if reach_y is not None:
        if rho_y is not None:
            raise ValueError("Specify either rho_y or reach_y, not both.")
        rho_y = reach_y ** 2

    # Build epsilon schedule
    if eps_list is None:
        if use_epsilon_scaling:
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = epsilon_schedule(diameter, blur, scaling, p=2.0)
        else:
            if eps is None or n_iters is None:
                raise ValueError("When use_epsilon_scaling=False, provide eps and n_iters")
            eps_list = [float(eps)] * int(n_iters)

    if len(eps_list) == 0:
        raise ValueError("eps_list must be non-empty")
    if n_iters is not None:
        eps_list = list(eps_list)[:int(n_iters)]

    # Early stopping state (will be set after first iteration)
    prev_f = None
    prev_g = None

    n_iters_used = 0

    # Precompute small vectors
    x_f32 = x.float().contiguous()
    y_f32 = y.float().contiguous()
    alpha = cost_scale * (x_f32 ** 2).sum(dim=1)
    beta = cost_scale * (y_f32 ** 2).sum(dim=1)
    log_a = log_weights(a)
    log_b = log_weights(b)

    # Work entirely in shifted potential space: f̂ = f - α, ĝ = g - β
    # Standard init is f=0, g=0, so f̂ = 0 - α = -α, ĝ = 0 - β = -β
    # With warm-start: f̂ = f_init - α, ĝ = g_init - β
    if f_init is not None:
        f_hat = f_init.float().contiguous() - alpha
    else:
        f_hat = -alpha.clone()
    if g_init is not None:
        g_hat = g_init.float().contiguous() - beta
    else:
        g_hat = -beta.clone()

    # Auto-switch to separate kernels for large n (benchmark shows ~10% speedup)
    # At n >= 30000, separate kernels have better memory access patterns
    if fused and n >= 30000:
        fused = False

    if fused:
        # =====================================================================
        # FUSED PATH: Single kernel launch per iteration
        # Both f and g computed in parallel using old potentials
        # =====================================================================

        # Initial step at eps_list[0] (alpha=1.0) - FUSED kernel
        eps_0 = eps_list[0]
        # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
        # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
        damp_f = dampening(eps_0, rho_x)
        damp_g = dampening(eps_0, rho_y)
        f_hat, g_hat = flashsinkhorn_symmetric_step(
            x_f32, y_f32, f_hat, g_hat, log_a, log_b, eps_0,
            cost_scale=cost_scale, alpha=1.0, damping_f=damp_f, damping_g=damp_g,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
            lambda_x=lambda_x, lambda_y=lambda_y,
        )
        # UNBALANCED OT CORRECTION for initial step (alpha=1.0)
        if damp_f < 1.0:
            f_hat = f_hat - 1.0 * alpha * (1.0 - damp_f)
        if damp_g < 1.0:
            g_hat = g_hat - 1.0 * beta * (1.0 - damp_g)
        n_iters_used += 1

        # Symmetric updates with alpha=0.5 - FUSED kernel
        for iter_idx, step_eps in enumerate(eps_list):
            damp_f = dampening(step_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(step_eps, rho_y)  # FIXED: rho_y → g

            # FUSED: both f and g updates in ONE kernel launch
            f_hat, g_hat = flashsinkhorn_symmetric_step(
                x_f32, y_f32, f_hat, g_hat, log_a, log_b, step_eps,
                cost_scale=cost_scale, alpha=0.5, damping_f=damp_f, damping_g=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
                label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
                lambda_x=lambda_x, lambda_y=lambda_y,
            )
            # UNBALANCED OT CORRECTION: The kernel incorrectly applies damping to the
            # shift term (α, β). The correct formula is: f̂ = damping*f_raw - α
            # but the kernel computes: f̂_bug = damping*f_raw - damping*α
            # Correction: subtract sym_alpha * α * (1 - damping) to fix the shift
            # (because f_correct - f_bug = -α*(1-damping))
            if damp_f < 1.0:
                f_hat = f_hat - 0.5 * alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_hat = g_hat - 0.5 * beta * (1.0 - damp_g)
            n_iters_used += 1

            # Early stopping check (in shifted space)
            if threshold is not None and (iter_idx + 1) % check_every == 0:
                if prev_f is None:
                    prev_f = f_hat.clone()
                    prev_g = g_hat.clone()
                else:
                    f_change = (f_hat - prev_f).abs().max().item()
                    g_change = (g_hat - prev_g).abs().max().item()
                    if max(f_change, g_change) < threshold:
                        break
                    prev_f.copy_(f_hat)
                    prev_g.copy_(g_hat)

        # Final extrapolation (alpha=1.0) - FUSED kernel
        if last_extrapolation:
            f_prelast = f_hat + alpha
            g_prelast = g_hat + beta

            final_eps = eps_list[-1]
            damp_f = dampening(final_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(final_eps, rho_y)  # FIXED: rho_y → g

            f_hat, g_hat = flashsinkhorn_symmetric_step(
                x_f32, y_f32, f_hat, g_hat, log_a, log_b, final_eps,
                cost_scale=cost_scale, alpha=1.0, damping_f=damp_f, damping_g=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
                label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
                lambda_x=lambda_x, lambda_y=lambda_y,
            )
            # UNBALANCED OT CORRECTION: alpha=1.0 for final step
            if damp_f < 1.0:
                f_hat = f_hat - 1.0 * alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_hat = g_hat - 1.0 * beta * (1.0 - damp_g)
            n_iters_used += 1

    else:
        # Note: SEPARATE PATH does not currently support OTDD label cost
        # For label cost, use fused=True (default)
        if label_x is not None and label_y is not None and label_cost_matrix is not None and lambda_y != 0.0:
            raise ValueError("OTDD label cost requires fused=True (default). Set fused=True or omit label_cost_matrix.")
        # =====================================================================
        # SEPARATE PATH: Two kernel launches per iteration
        # Uses flashsinkhorn_lse for each update with manual averaging
        #
        # KEY: Uses raw x, y coordinates (NOT pre-scaled Q, K) to ensure
        # TF32 rounding matches the fused kernel. This is critical for parity.
        # =====================================================================

        # Initial step at eps_list[0] (alpha=1.0)
        # CRITICAL: For initialization, GeomLoss uses softmin(log_a) and softmin(log_b)
        # Initial step at eps_list[0] (alpha=1.0)
        eps_0 = eps_list[0]
        # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
        # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
        damp_f = dampening(eps_0, rho_x)
        damp_g = dampening(eps_0, rho_y)
        gamma = eps_0 * log_a
        delta = eps_0 * log_b

        # Store old potentials for symmetric computation
        f_old = f_hat.clone()  # = -alpha
        g_old = g_hat.clone()  # = -beta

        # f-update (full, alpha=1.0) - uses OLD g
        u = (g_old + delta) / eps_0
        f_cand = flashsinkhorn_lse(
            x_f32, y_f32, u, eps_0, cost_scale=cost_scale, damping=damp_f,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
        )

        # g-update (full, alpha=1.0) - uses OLD f (symmetric!)
        v = (f_old + gamma) / eps_0
        g_cand = flashsinkhorn_lse(
            y_f32, x_f32, v, eps_0, cost_scale=cost_scale, damping=damp_g,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
        )

        # UNBALANCED OT CORRECTION for initial step
        if damp_f < 1.0:
            f_cand = f_cand - alpha * (1.0 - damp_f)
        if damp_g < 1.0:
            g_cand = g_cand - beta * (1.0 - damp_g)

        # alpha=1.0 means no averaging: new = candidate
        f_hat = f_cand
        g_hat = g_cand
        n_iters_used += 1

        # Symmetric updates with alpha=0.5
        for iter_idx, step_eps in enumerate(eps_list):
            damp_f = dampening(step_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(step_eps, rho_y)  # FIXED: rho_y → g
            delta = step_eps * log_b
            gamma = step_eps * log_a

            # Store old potentials for symmetric averaging
            f_old = f_hat.clone()
            g_old = g_hat.clone()

            # f-update candidate (uses old g)
            u = (g_old + delta) / step_eps
            f_cand = flashsinkhorn_lse(
                x_f32, y_f32, u, step_eps, cost_scale=cost_scale, damping=damp_f,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )

            # g-update candidate (uses old f, NOT f_cand - this is symmetric!)
            v = (f_old + gamma) / step_eps
            g_cand = flashsinkhorn_lse(
                y_f32, x_f32, v, step_eps, cost_scale=cost_scale, damping=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )

            # UNBALANCED OT CORRECTION: Fix damping over-application to shift term
            # f_correct - f_bug = -α*(1-damping), so subtract to correct
            if damp_f < 1.0:
                f_cand = f_cand - alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_cand = g_cand - beta * (1.0 - damp_g)

            # Symmetric averaging: new = 0.5 * old + 0.5 * candidate
            f_hat = 0.5 * f_old + 0.5 * f_cand
            g_hat = 0.5 * g_old + 0.5 * g_cand
            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (iter_idx + 1) % check_every == 0:
                if prev_f is None:
                    prev_f = f_hat.clone()
                    prev_g = g_hat.clone()
                else:
                    f_change = (f_hat - prev_f).abs().max().item()
                    g_change = (g_hat - prev_g).abs().max().item()
                    if max(f_change, g_change) < threshold:
                        break
                    prev_f.copy_(f_hat)
                    prev_g.copy_(g_hat)

        # Final extrapolation (alpha=1.0)
        if last_extrapolation:
            f_prelast = f_hat + alpha
            g_prelast = g_hat + beta

            final_eps = eps_list[-1]
            damp_f = dampening(final_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(final_eps, rho_y)  # FIXED: rho_y → g
            gamma = final_eps * log_a
            delta = final_eps * log_b

            # Store old for Jacobi-style (both use old potentials)
            f_old = f_hat.clone()
            g_old = g_hat.clone()

            # Full f-update (uses old g)
            u = (g_old + delta) / final_eps
            f_cand = flashsinkhorn_lse(
                x_f32, y_f32, u, final_eps, cost_scale=cost_scale, damping=damp_f,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )
            # Full g-update (uses old f - Jacobi style!)
            v = (f_old + gamma) / final_eps
            g_cand = flashsinkhorn_lse(
                y_f32, x_f32, v, final_eps, cost_scale=cost_scale, damping=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )
            # UNBALANCED OT CORRECTION: alpha=1.0 for final step
            if damp_f < 1.0:
                f_cand = f_cand - alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_cand = g_cand - beta * (1.0 - damp_g)
            # alpha=1.0: no averaging
            f_hat = f_cand
            g_hat = g_cand
            n_iters_used += 1

    # Convert to standard potentials at the end
    f = f_hat + alpha
    g = g_hat + beta

    if return_prelast and last_extrapolation:
        if return_n_iters:
            return f, g, f_prelast, g_prelast, n_iters_used
        return f, g, f_prelast, g_prelast

    if return_n_iters:
        return f, g, n_iters_used
    return f, g
