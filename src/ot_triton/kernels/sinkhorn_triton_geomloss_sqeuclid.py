from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

# Import from _common and re-export for backward compatibility
from ot_triton.kernels._common import dampening, epsilon_schedule, log_weights, max_diameter


def _compute_marginal_error(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cost_scale: float = 1.0,
    norm_error: int = 1,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
) -> float:
    """Compute marginal error like OTT-JAX.

    For balanced Sinkhorn, computes ||P @ 1 - b||_p where P is the transport plan.
    This is the same convergence criterion used by OTT-JAX.

    Args:
        x, y: Point clouds (n, d) and (m, d)
        f, g: Current potentials
        a, b: Target marginals
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full, 0.5 for half cost)
        norm_error: Norm to use (1=L1, 2=L2)
        x2, y2: Precomputed squared norms (optional)

    Returns:
        Marginal error (scalar)
    """
    # Lazy import to avoid circular dependency
    from ot_triton.kernels.sinkhorn_triton_apply_sqeuclid import apply_plan_vec_sqeuclid

    n = x.shape[0]
    m = y.shape[0]

    # Compute row marginal: P @ ones(m)
    ones_m = torch.ones(m, device=x.device, dtype=torch.float32)
    row_marginal = apply_plan_vec_sqeuclid(
        x, y, f, g, ones_m,
        eps=eps, axis=1, cost_scale=cost_scale,
        x2=x2, y2=y2,
        autotune=False,  # Use default config for speed
    )

    # Compute error: ||row_marginal - a||_p
    diff = row_marginal - a.float()
    if norm_error == 1:
        error = diff.abs().sum().item()
    elif norm_error == 2:
        error = diff.pow(2).sum().sqrt().item()
    else:
        error = diff.abs().max().item()  # L-inf

    return error


def _geomloss_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for GeomLoss forward kernel.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.

    We include BLOCK_K=64 for d >= 128 since it significantly reduces
    label cost overhead (18% vs 55%). The autotuner will skip configs
    that produce incorrect results.
    """
    configs = []
    for block_m, block_n in ((128, 64), (64, 128), (64, 64)):
        # Include block_k=64 for better label cost performance on large d
        # The autotuner key includes D, so d=64 will get separate tuning
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
    # OTDD label cost parameters (optional)
    label_x_ptr,  # int32 labels for x: [n] or None
    label_y_ptr,  # int32 labels for y: [m] or None
    W_ptr,        # Label cost matrix: [V, V] or None
    n,
    m,
    V,            # Number of classes (0 if no label cost)
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
    damping_f,  # Unbalanced OT damping for f: 1/(1+eps/rho_x), or 1.0 for balanced
    damping_g,  # Unbalanced OT damping for g: 1/(1+eps/rho_y), or 1.0 for balanced
    cost_scale,  # Cost scaling: 1.0 for full ||x-y||², 0.5 for half ||x-y||²/2
    lambda_x,    # Weight for Euclidean cost (default 1.0)
    lambda_y,    # Weight for label cost (default 0.0 if no labels)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
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

            # Compute Euclidean cost
            euclidean_cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)

            # Add label cost if enabled (OTDD-style augmented cost)
            if USE_LABEL_COST:
                # Load labels for this block
                label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
                label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Combined cost: lambda_x * euclidean + lambda_y * label
                # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
                cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
            else:
                cost = euclidean_cost

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

        # Guard against s_i == 0 (can happen if all exp values underflow)
        # Use 1e-40 as minimum to avoid log(0) = -inf
        s_i_safe = tl.maximum(s_i, 1e-40)
        lse = (m_i + tl.log2(s_i_safe)) * ln2 if USE_EXP2 else m_i + tl.log(s_i_safe)
        # Unbalanced OT: scale softmin by damping factor = 1/(1+eps/rho_x)
        cand = -eps * lse * damping_f
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

        # Compute Euclidean cost
        euclidean_cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot)

        # Add label cost if enabled (OTDD-style augmented cost)
        if USE_LABEL_COST:
            # Load labels for this block
            label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
            label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
            # Compute flattened indices into W: W[label_i, label_j]
            w_idx = label_i[:, None] * V + label_j[None, :]
            # Gather from W (label cost matrix)
            w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
            # Combined cost: lambda_x * euclidean + lambda_y * label
            # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
            cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
        else:
            cost = euclidean_cost

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

    # Guard against s_j == 0 (can happen if all exp values underflow)
    # Use 1e-40 as minimum to avoid log(0) = -inf
    s_j_safe = tl.maximum(s_j, 1e-40)
    lse = (m_j + tl.log2(s_j_safe)) * ln2 if USE_EXP2 else m_j + tl.log(s_j_safe)
    # Unbalanced OT: scale softmin by damping factor = 1/(1+eps/rho_y)
    cand = -eps * lse * damping_g
    g_new = (1.0 - alpha) * g_old + alpha * cand
    tl.store(g_out_ptr + offs_n * stride_g_out, g_new, mask=mask_n)


def _default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    """Default block sizes for GeomLoss forward kernel.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.

    The key constraint is: BLOCK_K < D (must have at least 2 k iterations).
    - d >= 128: block_k = 64 → at least 2 iterations (best for label cost)
    - d >= 64:  block_k = 32 → at least 2 iterations
    - d >= 32:  block_k = 16 → at least 2 iterations
    - d < 32:   block_k = 16 → minimum for tl.dot

    Note: Larger block_k significantly reduces label cost overhead
    (18% with block_k=64 vs 55% with block_k=16 for d=512).
    """
    # Tuned default for strict fp32 (no TF32): favor smaller K and fewer warps.
    if dtype == torch.float32 and not allow_tf32:
        block_m = 128 if d >= 32 else 64
        block_n = 64 if d >= 32 else 64
        # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
        if d >= 128:
            block_k = 64  # At least 2 k iterations, best for label cost
        elif d >= 64:
            block_k = 32  # Forces at least 2 k iterations for d >= 64
        elif d >= 32:
            block_k = 16  # Forces at least 2 k iterations for d >= 32
        else:
            block_k = 16  # Minimum for tl.dot
        num_warps = 4
        return block_m, block_n, block_k, num_warps

    # TF32 mode: use larger blocks but keep BLOCK_K safe
    block_m = 64
    block_n = 64
    # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
    if d >= 128:
        block_k = 64  # At least 2 k iterations, best for label cost
    elif d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot
    num_warps = 4
    return block_m, block_n, block_k, num_warps


# Create separate autotune kernels for label vs no-label workloads.
# This is necessary because the autotune cache doesn't include USE_LABEL_COST in the key,
# so label and no-label workloads would otherwise share the same (suboptimal) config.

def _geomloss_autotune_configs_label() -> Sequence[triton.Config]:
    """Autotune configs optimized for label cost workloads (OTDD).

    Label cost requires scatter-gather access to W[label_x, label_y] every Sinkhorn iteration.
    Benchmarking (d=512, A100-80GB) showed M64_N64_K64_W4 is FASTEST for label:

    | n     | M64_N64_K64_W4 label | M64_N128_K64_W4 label |
    |-------|----------------------|-----------------------|
    | 2000  | 8.3 ms               | 10.5 ms               |
    | 5000  | 21.3 ms              | 27.4 ms               |
    | 10000 | 80.3 ms              | 83.9 ms               |
    | 20000 | 265.0 ms             | 271.2 ms              |

    Overhead varies 11-31% depending on problem size, but absolute speed is more
    important than overhead for paper benchmarks (comparing against GeomLoss).

    BLOCK_K=64 is critical for good performance:
    - Reduces k-loop iterations → better W matrix cache reuse
    - Only valid for d >= 128 (need BLOCK_K < d for correctness)
    """
    # Use the fastest label config
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        )
    ]


# Standard autotune kernel (for no-label workloads)
_geomloss_symmetric_step_sqeuclid_autotune = triton.autotune(
    configs=_geomloss_autotune_configs(),
    key=["D", "ALLOW_TF32", "DTYPE_ID"],
)(_geomloss_symmetric_step_sqeuclid_impl)

# Separate autotune kernel for label workloads (with BLOCK_K=64 only)
_geomloss_symmetric_step_sqeuclid_autotune_label = triton.autotune(
    configs=_geomloss_autotune_configs_label(),
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
    rho: Optional[float] = None,  # Deprecated: use rho_x/rho_y for semi-unbalanced
    reach: Optional[float] = None,  # Deprecated: use reach_x/reach_y
    rho_x: Optional[float] = None,  # Unbalanced OT: source marginal penalty (None = strict)
    rho_y: Optional[float] = None,  # Unbalanced OT: target marginal penalty (None = strict)
    reach_x: Optional[float] = None,  # Alternative: reach_x^2 = rho_x
    reach_y: Optional[float] = None,  # Alternative: reach_y^2 = rho_y
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
    use_exp2: bool = True,
    return_prelast: bool = False,
    cost_scale: float = 1.0,
    # OTDD label-augmented cost parameters
    label_x: Optional[torch.Tensor] = None,  # int32/int64 labels for x: [n]
    label_y: Optional[torch.Tensor] = None,  # int32/int64 labels for y: [m]
    label_cost_matrix: Optional[torch.Tensor] = None,  # W: [V, V] label distances
    lambda_x: float = 1.0,  # Weight for Euclidean cost
    lambda_y: float = 0.0,  # Weight for label cost (0 = Euclidean only)
    # Early stopping parameters (like OTT-JAX)
    threshold: Optional[float] = None,  # Convergence threshold (None = no early stopping)
    check_every: int = 5,  # Check convergence every N iterations
    return_n_iters: bool = False,  # If True, also return number of iterations used
    # Warm-start parameters (use previous potentials as initial guess)
    f_init: Optional[torch.Tensor] = None,  # Initial source potential (None = zeros)
    g_init: Optional[torch.Tensor] = None,  # Initial target potential (None = zeros)
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
] | Tuple[torch.Tensor, torch.Tensor, int] | Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
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
        label_x = label_x.to(dtype=torch.int32, device=x.device).contiguous()
        label_y = label_y.to(dtype=torch.int32, device=x.device).contiguous()
        W = label_cost_matrix.to(dtype=torch.float32, device=x.device).contiguous()
    else:
        # Provide dummy tensors (size 1) to satisfy Triton's pointer requirements
        # These are never accessed when USE_LABEL_COST=False
        V = 1
        label_x = torch.zeros(1, dtype=torch.int32, device=x.device)
        label_y = torch.zeros(1, dtype=torch.int32, device=x.device)
        W = torch.zeros(1, dtype=torch.float32, device=x.device)

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

    # Initialize potentials - use warm-start if provided, otherwise zeros
    if f_init is not None:
        f0 = f_init.to(device=x.device, dtype=torch.float32).clone()
        if f0.shape[0] != n:
            raise ValueError(f"f_init has wrong shape: {f0.shape[0]} != {n}")
    else:
        f0 = torch.zeros((n,), device=x.device, dtype=torch.float32)

    if g_init is not None:
        g0 = g_init.to(device=x.device, dtype=torch.float32).clone()
        if g0.shape[0] != m:
            raise ValueError(f"g_init has wrong shape: {g0.shape[0]} != {m}")
    else:
        g0 = torch.zeros((m,), device=x.device, dtype=torch.float32)

    f1 = torch.empty_like(f0)
    g1 = torch.empty_like(g0)

    def _launch_manual(f_in, g_in, f_out, g_out, step_eps: float, alpha: float,
                       step_damping_f: float, step_damping_g: float):
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
            # OTDD label cost pointers (None if not used)
            label_x,
            label_y,
            W,
            n,
            m,
            V,
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
            float(step_damping_f),
            float(step_damping_g),
            float(cost_scale),
            float(lambda_x),
            float(lambda_y),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune(f_in, g_in, f_out, g_out, step_eps: float, alpha: float,
                         step_damping_f: float, step_damping_g: float):
        def grid(meta):
            return (
                triton.cdiv(n, meta["BLOCK_M"])
                + triton.cdiv(m, meta["BLOCK_N"]),
            )

        # Select autotune kernel based on label cost usage
        # Label workloads use separate kernel with BLOCK_K=64 for better cache locality
        kernel = (_geomloss_symmetric_step_sqeuclid_autotune_label if use_label_cost
                  else _geomloss_symmetric_step_sqeuclid_autotune)

        kernel[grid](
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
            # OTDD label cost pointers (None if not used)
            label_x,
            label_y,
            W,
            n,
            m,
            V,
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
            float(step_damping_f),
            float(step_damping_g),
            float(cost_scale),
            float(lambda_x),
            float(lambda_y),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
        )

    # Select launch function. Label workloads use a separate autotune kernel instance
    # with BLOCK_K=64 only configs (see _geomloss_autotune_configs_label).
    launch = _launch_autotune if use_autotune else _launch_manual

    # GeomLoss-style init at eps_list[0].
    # For unbalanced/semi-unbalanced OT, damping = 1/(1+eps/rho); for balanced, damping = 1.0
    init_damping_f = dampening(eps_list[0], rho_x)
    init_damping_g = dampening(eps_list[0], rho_y)
    launch(f0, g0, f1, g1, eps_list[0], alpha=1.0,
           step_damping_f=init_damping_f, step_damping_g=init_damping_g)
    f0, f1 = f1, f0
    g0, g1 = g1, g0

    # Symmetric updates, including eps_list[0] again (matches GeomLoss sinkhorn_loop).
    n_iters_used = 1  # Count initial step
    converged = False
    prev_f = f0.clone() if threshold is not None else None
    prev_g = g0.clone() if threshold is not None else None

    for iter_idx, step_eps in enumerate(eps_list):
        step_damping_f = dampening(step_eps, rho_x)
        step_damping_g = dampening(step_eps, rho_y)
        launch(f0, g0, f1, g1, step_eps, alpha=0.5,
               step_damping_f=step_damping_f, step_damping_g=step_damping_g)
        f0, f1 = f1, f0
        g0, g1 = g1, g0
        n_iters_used += 1

        # Early stopping: check convergence every check_every iterations
        # Use potential change as convergence metric (cheap: just max reduction)
        if threshold is not None and (iter_idx + 1) % check_every == 0:
            # Compute max potential change since last check
            f_change = (f0 - prev_f).abs().max().item()
            g_change = (g0 - prev_g).abs().max().item()
            error = max(f_change, g_change)

            if error < threshold:
                converged = True
                break

            # Update prev for next check
            prev_f.copy_(f0)
            prev_g.copy_(g0)

    if last_extrapolation:
        # Match GeomLoss's `sinkhorn_loop(last_extrapolation=True)` behavior:
        # do one final "full" update at the last epsilon value.
        final_damping_f = dampening(eps_list[-1], rho_x)
        final_damping_g = dampening(eps_list[-1], rho_y)
        launch(f0, g0, f1, g1, eps_list[-1], alpha=1.0,
               step_damping_f=final_damping_f, step_damping_g=final_damping_g)
        n_iters_used += 1
        if return_prelast:
            if return_n_iters:
                return f1, g1, f0, g0, n_iters_used
            return f1, g1, f0, g0
        if return_n_iters:
            return f1, g1, n_iters_used
        return f1, g1

    if return_prelast:
        if return_n_iters:
            return f0, g0, f0, g0, n_iters_used
        return f0, g0, f0, g0
    if return_n_iters:
        return f0, g0, n_iters_used
    return f0, g0
