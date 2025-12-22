from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from ot_triton.hvp import geomloss_to_ott_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials
from ot_triton.hvp import hvp_x_sqeuclid_from_potentials_taskcsr
from ot_triton.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    _default_block_sizes as _dense_default_block_sizes,
    epsilon_schedule,
    log_weights,
    max_diameter,
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_multiscale_sqeuclid import (
    sinkhorn_geomloss_multiscale_potentials_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_ranges_sqeuclid import (
    blocksparse_build_tasks_from_csr,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_sqeuclid import (
    blocksparse_prepare_metadata,
    geomloss_blocksparse_grad_sqeuclid,
)
from ot_triton.kernels.sinkhorn_triton_geomloss_blocksparse_taskcsr_sqeuclid import (
    blocksparse_build_taskcsr,
    blocksparse_build_taskcsr_buckets,
)
from ot_triton.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)


class _SinkhornGradFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        f_grad: torch.Tensor,
        g_grad: torch.Tensor,
        eps: float,
        allow_tf32: bool,
        use_exp2: bool,
        autotune: bool,
        block_m: Optional[int],
        block_n: Optional[int],
        block_k: Optional[int],
        num_warps: Optional[int],
        num_stages: int,
        grad_scale: torch.Tensor,
        hvp_tau2: float,
        hvp_max_cg_iter: int,
        hvp_cg_rtol: float,
        hvp_cg_atol: float,
        hvp_cg_stabilise_every: int,
        hvp_preconditioner: str,
        hvp_precond_terms: int,
        hvp_use_preconditioner: bool,
        compute_grad_x: bool,
        compute_grad_y: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gx, gy = sinkhorn_geomloss_online_grad_sqeuclid(
            x,
            y,
            a,
            b,
            f_grad,
            g_grad,
            eps=float(eps),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=int(num_stages),
            autotune=bool(autotune),
            grad_scale=grad_scale,
            compute_grad_x=bool(compute_grad_x),
            compute_grad_y=bool(compute_grad_y),
        )

        ctx.save_for_backward(x, y, a, b, f_grad, g_grad, grad_scale)
        ctx.eps = float(eps)
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.use_exp2 = bool(use_exp2)
        ctx.autotune = bool(autotune)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.block_k = block_k
        ctx.num_warps = num_warps
        ctx.num_stages = int(num_stages)
        ctx.hvp_tau2 = float(hvp_tau2)
        ctx.hvp_max_cg_iter = int(hvp_max_cg_iter)
        ctx.hvp_cg_rtol = float(hvp_cg_rtol)
        ctx.hvp_cg_atol = float(hvp_cg_atol)
        ctx.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        ctx.hvp_preconditioner = str(hvp_preconditioner)
        ctx.hvp_precond_terms = int(hvp_precond_terms)
        ctx.hvp_use_preconditioner = bool(hvp_use_preconditioner)
        return gx, gy

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_x: Optional[torch.Tensor], grad_grad_y: Optional[torch.Tensor]
    ):
        x, y, a, b, f_grad, g_grad, grad_scale = ctx.saved_tensors

        if grad_grad_y is not None:
            raise NotImplementedError("Double backward w.r.t y is not implemented yet.")

        out_x = None
        if grad_grad_x is not None:
            if grad_grad_x.shape != x.shape:
                raise ValueError("grad_grad_x must have the same shape as x.")
            if not grad_grad_x.is_cuda:
                raise ValueError("grad_grad_x must be a CUDA tensor.")

            f_hat, g_hat = geomloss_to_ott_potentials(
                f_grad, g_grad, a, b, eps=ctx.eps
            )
            hvp_x, _ = hvp_x_sqeuclid_from_potentials(
                x,
                y,
                f_hat,
                g_hat,
                grad_grad_x,
                eps=ctx.eps,
                tau2=ctx.hvp_tau2,
                max_cg_iter=ctx.hvp_max_cg_iter,
                cg_rtol=ctx.hvp_cg_rtol,
                cg_atol=ctx.hvp_cg_atol,
                cg_stabilise_every=ctx.hvp_cg_stabilise_every,
                preconditioner=ctx.hvp_preconditioner,
                precond_terms=ctx.hvp_precond_terms,
                use_preconditioner=ctx.hvp_use_preconditioner,
                allow_tf32=ctx.allow_tf32,
                use_exp2=ctx.use_exp2,
                block_m=ctx.block_m,
                block_n=ctx.block_n,
                block_k=ctx.block_k,
                num_warps=int(ctx.num_warps or 4),
                num_stages=ctx.num_stages,
            )
            out_x = hvp_x * grad_scale

        # Inputs: x,y,a,b,f_grad,g_grad,eps,allow_tf32,use_exp2,autotune,
        # block_m,block_n,block_k,num_warps,num_stages,grad_scale,
        # hvp_tau2,hvp_max_cg_iter,hvp_cg_rtol,hvp_cg_atol,hvp_cg_stabilise_every,hvp_preconditioner,hvp_precond_terms,hvp_use_preconditioner,
        # compute_grad_x, compute_grad_y
        return (
            out_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SinkhornGradFnMultiscale(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        f_grad: torch.Tensor,
        g_grad: torch.Tensor,
        perm_x: torch.Tensor,
        perm_y: torch.Tensor,
        offsets_x: torch.Tensor,
        offsets_y: torch.Tensor,
        row_ptr_x: torch.Tensor,
        col_idx_x: torch.Tensor,
        row_ptr_y: torch.Tensor,
        col_idx_y: torch.Tensor,
        eps: float,
        allow_tf32: bool,
        use_exp2: bool,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
        grad_scale: torch.Tensor,
        hvp_tau2: float,
        hvp_max_cg_iter: int,
        hvp_cg_rtol: float,
        hvp_cg_atol: float,
        hvp_cg_stabilise_every: int,
        hvp_preconditioner: str,
        hvp_precond_terms: int,
        hvp_use_preconditioner: bool,
        multiscale_blocksparse_backend: str,
        compute_grad_x: bool,
        compute_grad_y: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        (
            pid_x_cluster,
            pid_x_block,
            pid_y_cluster,
            pid_y_block,
            max_blocks_x,
            max_blocks_y,
            max_deg_x,
            max_deg_y,
        ) = blocksparse_prepare_metadata(
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            row_ptr_y=row_ptr_y,
            block_m=int(block_m),
            block_n=int(block_n),
        )

        x_s = x[perm_x]
        y_s = y[perm_y]
        a_s = a[perm_x].float().contiguous()
        b_s = b[perm_y].float().contiguous()
        f_s = f_grad[perm_x].float().contiguous()
        g_s = g_grad[perm_y].float().contiguous()
        x2_s = (x_s.float() * x_s.float()).sum(dim=1).contiguous()
        y2_s = (y_s.float() * y_s.float()).sum(dim=1).contiguous()
        loga_s = log_weights(a_s).contiguous()
        logb_s = log_weights(b_s).contiguous()

        gx_s, gy_s = geomloss_blocksparse_grad_sqeuclid(
            x_s,
            y_s,
            a_s,
            b_s,
            f_s,
            g_s,
            loga_s,
            logb_s,
            x2_s,
            y2_s,
            offsets_x=offsets_x,
            offsets_y=offsets_y,
            row_ptr_x=row_ptr_x,
            col_idx_x=col_idx_x,
            row_ptr_y=row_ptr_y,
            col_idx_y=col_idx_y,
            eps=float(eps),
            grad_scale=grad_scale,
            pid_x_cluster=pid_x_cluster,
            pid_x_block=pid_x_block,
            pid_y_cluster=pid_y_cluster,
            pid_y_block=pid_y_block,
            max_blocks_x=max_blocks_x,
            max_blocks_y=max_blocks_y,
            max_deg_x=max_deg_x,
            max_deg_y=max_deg_y,
            block_m=int(block_m),
            block_n=int(block_n),
            block_k=int(block_k),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            compute_grad_x=bool(compute_grad_x),
            compute_grad_y=bool(compute_grad_y),
        )

        gx = None
        if gx_s is not None:
            gx = torch.empty_like(x, dtype=torch.float32)
            gx[perm_x] = gx_s
        gy = None
        if gy_s is not None:
            gy = torch.empty_like(y, dtype=torch.float32)
            gy[perm_y] = gy_s

        ctx.save_for_backward(
            x,
            y,
            a,
            b,
            f_grad,
            g_grad,
            perm_x,
            perm_y,
            offsets_x,
            offsets_y,
            row_ptr_x,
            col_idx_x,
            row_ptr_y,
            col_idx_y,
            grad_scale,
        )
        ctx.eps = float(eps)
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.use_exp2 = bool(use_exp2)
        ctx.block_m = int(block_m)
        ctx.block_n = int(block_n)
        ctx.num_warps = int(num_warps)
        ctx.num_stages = int(num_stages)
        ctx.hvp_tau2 = float(hvp_tau2)
        ctx.hvp_max_cg_iter = int(hvp_max_cg_iter)
        ctx.hvp_cg_rtol = float(hvp_cg_rtol)
        ctx.hvp_cg_atol = float(hvp_cg_atol)
        ctx.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        ctx.hvp_preconditioner = str(hvp_preconditioner)
        ctx.hvp_precond_terms = int(hvp_precond_terms)
        ctx.hvp_use_preconditioner = bool(hvp_use_preconditioner)
        ctx.multiscale_blocksparse_backend = str(multiscale_blocksparse_backend)
        return gx, gy

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_x: Optional[torch.Tensor], grad_grad_y: Optional[torch.Tensor]
    ):
        (
            x,
            y,
            a,
            b,
            f_grad,
            g_grad,
            perm_x,
            perm_y,
            offsets_x,
            offsets_y,
            row_ptr_x,
            col_idx_x,
            row_ptr_y,
            col_idx_y,
            grad_scale,
        ) = ctx.saved_tensors

        if grad_grad_y is not None:
            raise NotImplementedError("Double backward w.r.t y is not implemented yet.")

        out_x = None
        if grad_grad_x is not None:
            if grad_grad_x.shape != x.shape:
                raise ValueError("grad_grad_x must have the same shape as x.")
            if not grad_grad_x.is_cuda:
                raise ValueError("grad_grad_x must be a CUDA tensor.")

            x_s = x[perm_x].contiguous()
            y_s = y[perm_y].contiguous()
            a_s = a[perm_x].float().contiguous()
            b_s = b[perm_y].float().contiguous()
            f_s = f_grad[perm_x].float().contiguous()
            g_s = g_grad[perm_y].float().contiguous()
            A_s = grad_grad_x[perm_x].contiguous()

            f_hat, g_hat = geomloss_to_ott_potentials(f_s, g_s, a_s, b_s, eps=ctx.eps)

            tasks = blocksparse_build_tasks_from_csr(
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                row_ptr_x=row_ptr_x,
                col_idx_x=col_idx_x,
                block_m=int(ctx.block_m),
                block_n=int(ctx.block_n),
            )
            taskcsr_x = blocksparse_build_taskcsr(tasks, by="x")
            taskcsr_y = blocksparse_build_taskcsr(tasks, by="y")

            buckets = None
            backend = ctx.multiscale_blocksparse_backend
            if backend == "auto":
                backend = "taskcsr_bucketed"
            if backend == "taskcsr_bucketed":
                buckets = blocksparse_build_taskcsr_buckets(taskcsr_x, taskcsr_y)

            hvp_s, _ = hvp_x_sqeuclid_from_potentials_taskcsr(
                x_s,
                y_s,
                f_hat,
                g_hat,
                A_s,
                eps=ctx.eps,
                offsets_x=offsets_x,
                offsets_y=offsets_y,
                taskcsr_x=taskcsr_x,
                taskcsr_y=taskcsr_y,
                buckets=buckets,
                tau2=ctx.hvp_tau2,
                max_cg_iter=ctx.hvp_max_cg_iter,
                cg_rtol=ctx.hvp_cg_rtol,
                cg_atol=ctx.hvp_cg_atol,
                cg_stabilise_every=ctx.hvp_cg_stabilise_every,
                preconditioner=ctx.hvp_preconditioner,
                precond_terms=ctx.hvp_precond_terms,
                use_preconditioner=ctx.hvp_use_preconditioner,
                block_m=int(ctx.block_m),
                block_n=int(ctx.block_n),
                num_warps=int(ctx.num_warps),
                num_stages=int(ctx.num_stages),
                use_exp2=bool(ctx.use_exp2),
            )

            out_x = torch.empty_like(x, dtype=torch.float32)
            out_x[perm_x] = hvp_s
            out_x = out_x * grad_scale

        # Inputs:
        # x,y,a,b,f_grad,g_grad,perm_x,perm_y,offsets_x,offsets_y,row_ptr_x,col_idx_x,row_ptr_y,col_idx_y,
        # eps,allow_tf32,use_exp2,block_m,block_n,block_k,num_warps,num_stages,grad_scale,
        # hvp_tau2,hvp_max_cg_iter,hvp_cg_rtol,hvp_cg_atol,hvp_cg_stabilise_every,hvp_preconditioner,hvp_precond_terms,hvp_use_preconditioner,
        # multiscale_blocksparse_backend,compute_grad_x,compute_grad_y
        return (
            out_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@dataclass(frozen=True)
class _ParsedInputs:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    batched: bool
    a_view_shape: Tuple[int, ...]
    b_view_shape: Tuple[int, ...]


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("Expected a torch.Tensor.")
    if not x.is_floating_point():
        raise TypeError("Expected a floating-point tensor.")
    return x


def _normalize_weights(w: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    w = w.float()
    if eps > 0:
        w = w + eps
    z = w.sum(dim=-1, keepdim=True)
    return w / z


def _process_args(*args, normalize: bool) -> _ParsedInputs:
    if len(args) == 2:
        x, y = args
        a = None
        b = None
    elif len(args) == 4:
        a, x, b, y = args
    else:
        raise TypeError(
            "SamplesLoss expects either (x, y) or (a, x, b, y). "
            f"Got {len(args)} arguments."
        )

    x = _as_float_tensor(x)
    y = _as_float_tensor(y)

    if x.ndim == 2:
        batched = False
        n, dx = x.shape
        m, dy = y.shape
        if dx != dy:
            raise ValueError("x and y must have the same feature dimension.")

        if a is None:
            a = torch.full((n,), 1.0 / n, device=x.device, dtype=torch.float32)
            a_view_shape = (n,)
        else:
            a = _as_float_tensor(a)
            if a.ndim == 2 and a.shape == (n, 1):
                a = a[:, 0]
                a_view_shape = (n, 1)
            elif a.ndim == 1 and a.shape == (n,):
                a_view_shape = (n,)
            else:
                raise ValueError("a must have shape (n,) or (n,1) matching x.")

        if b is None:
            b = torch.full((m,), 1.0 / m, device=y.device, dtype=torch.float32)
            b_view_shape = (m,)
        else:
            b = _as_float_tensor(b)
            if b.ndim == 2 and b.shape == (m, 1):
                b = b[:, 0]
                b_view_shape = (m, 1)
            elif b.ndim == 1 and b.shape == (m,):
                b_view_shape = (m,)
            else:
                raise ValueError("b must have shape (m,) or (m,1) matching y.")

        if normalize:
            a = _normalize_weights(a)
            b = _normalize_weights(b)

        return _ParsedInputs(
            x=x,
            y=y,
            a=a,
            b=b,
            batched=batched,
            a_view_shape=a_view_shape,
            b_view_shape=b_view_shape,
        )

    if x.ndim == 3:
        batched = True
        if y.ndim != 3:
            raise ValueError("If x is batched (B,N,D), y must be batched too.")
        bsz, n, dx = x.shape
        bsz2, m, dy = y.shape
        if bsz != bsz2:
            raise ValueError("x and y must have the same batch size.")
        if dx != dy:
            raise ValueError("x and y must have the same feature dimension.")

        if a is None:
            a = torch.full((bsz, n), 1.0 / n, device=x.device, dtype=torch.float32)
            a_view_shape = (bsz, n)
        else:
            a = _as_float_tensor(a)
            if a.ndim == 3 and a.shape == (bsz, n, 1):
                a = a[:, :, 0]
                a_view_shape = (bsz, n, 1)
            elif a.ndim == 2 and a.shape == (bsz, n):
                a_view_shape = (bsz, n)
            else:
                raise ValueError(
                    "a must have shape (B,n) or (B,n,1) matching x."
                )

        if b is None:
            b = torch.full((bsz, m), 1.0 / m, device=y.device, dtype=torch.float32)
            b_view_shape = (bsz, m)
        else:
            b = _as_float_tensor(b)
            if b.ndim == 3 and b.shape == (bsz, m, 1):
                b = b[:, :, 0]
                b_view_shape = (bsz, m, 1)
            elif b.ndim == 2 and b.shape == (bsz, m):
                b_view_shape = (bsz, m)
            else:
                raise ValueError(
                    "b must have shape (B,m) or (B,m,1) matching y."
                )

        if normalize:
            a = _normalize_weights(a)
            b = _normalize_weights(b)

        return _ParsedInputs(
            x=x,
            y=y,
            a=a,
            b=b,
            batched=batched,
            a_view_shape=a_view_shape,
            b_view_shape=b_view_shape,
        )

    raise ValueError("x and y must be shaped (N,D) or (B,N,D).")


class SamplesLoss(torch.nn.Module):
    """GeomLoss-like API for (online) Sinkhorn OT using Triton.

    This is a minimal, CUDA-only subset of GeomLoss's `SamplesLoss`:
    - `loss` must be "sinkhorn".
    - Only squared Euclidean ground cost is supported.
    - Only the balanced (reach=None) setting is supported.

    The implementation uses a GeomLoss-style symmetric Sinkhorn loop
    (`sinkhorn_geomloss_online_potentials_sqeuclid`) and returns either:
    - a scalar OT cost (default), or
    - a pair of potentials (f, g) when `potentials=True`.

    Notes
    -----
    - Gradients are computed analytically (no backprop through Sinkhorn iterations),
      matching GeomLoss's `last_extrapolation` convention.
    - `potentials=True` returns (f, g) without autograd support.
    - For `backend="multiscale"`, `multiscale_blocksparse_backend` selects the fine-level
      sparse reduction layout: `auto` (heuristic), `padded`, `taskcsr`,
      `taskcsr_bucketed` (degree-bucketed task-CSR), or `ranges_atomic`.
    """

    def __init__(
        self,
        loss: str = "sinkhorn",
        *,
        p: int = 2,
        blur: float = 0.05,
        scaling: float = 0.5,
        debias: bool = False,
        potentials: bool = False,
        backend: str = "online",
        normalize: bool = True,
        use_epsilon_scaling: bool = True,
        last_extrapolation: bool = True,
        truncate: float = 5.0,
        cluster_scale: Optional[float] = None,
        max_coarse_levels: int = 1,
        multiscale_blocksparse_backend: str = "auto",
        allow_tf32: bool = True,
        use_exp2: bool = True,
        autotune: bool = True,
        eps: Optional[float] = None,
        n_iters: Optional[int] = None,
        diameter: Optional[float] = None,
        eps_list: Optional[Sequence[float]] = None,
        block_m: Optional[int] = None,
        block_n: Optional[int] = None,
        block_k: Optional[int] = None,
        num_warps: Optional[int] = None,
        num_stages: int = 2,
        hvp_tau2: float = 1e-5,
        hvp_max_cg_iter: int = 300,
        hvp_cg_rtol: float = 1e-6,
        hvp_cg_atol: float = 1e-6,
        hvp_cg_stabilise_every: int = 0,
        hvp_preconditioner: str = "none",
        hvp_precond_terms: int = 3,
        hvp_use_preconditioner: bool = True,
    ):
        super().__init__()

        if loss != "sinkhorn":
            raise ValueError('Only loss="sinkhorn" is supported.')
        if p != 2:
            raise ValueError("Only p=2 (squared Euclidean cost) is supported.")
        if debias:
            raise NotImplementedError(
                "Debiased Sinkhorn divergence is not implemented in ot_triton yet. "
                "Use debias=False." 
            )
        if backend not in ("online", "triton", "multiscale", "auto"):
            raise ValueError(
                'Only backend in {"online","triton","multiscale","auto"} is supported.'
            )
        if multiscale_blocksparse_backend not in (
            "auto",
            "padded",
            "ranges_atomic",
            "taskcsr",
            "taskcsr_bucketed",
        ):
            raise ValueError(
                "multiscale_blocksparse_backend must be one of "
                "{'auto','padded','ranges_atomic','taskcsr','taskcsr_bucketed'}."
            )

        self.loss = loss
        self.p = p
        self.blur = float(blur)
        self.scaling = float(scaling)
        self.debias = bool(debias)
        self.potentials = bool(potentials)
        self.backend = backend
        self.normalize = bool(normalize)

        self.use_epsilon_scaling = bool(use_epsilon_scaling)
        self.last_extrapolation = bool(last_extrapolation)
        self.truncate = float(truncate)
        self.cluster_scale = None if cluster_scale is None else float(cluster_scale)
        self.max_coarse_levels = int(max_coarse_levels)
        self.multiscale_blocksparse_backend = str(multiscale_blocksparse_backend)
        self.allow_tf32 = bool(allow_tf32)
        self.use_exp2 = bool(use_exp2)
        self.autotune = bool(autotune)

        self.eps = None if eps is None else float(eps)
        self.n_iters = None if n_iters is None else int(n_iters)
        self.diameter = None if diameter is None else float(diameter)
        self.eps_list = None if eps_list is None else list(map(float, eps_list))

        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = int(num_stages)

        self.hvp_tau2 = float(hvp_tau2)
        self.hvp_max_cg_iter = int(hvp_max_cg_iter)
        self.hvp_cg_rtol = float(hvp_cg_rtol)
        self.hvp_cg_atol = float(hvp_cg_atol)
        self.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        self.hvp_preconditioner = str(hvp_preconditioner)
        self.hvp_precond_terms = int(hvp_precond_terms)
        self.hvp_use_preconditioner = bool(hvp_use_preconditioner)

    def _eps_list_for_inputs(self, x: torch.Tensor, y: torch.Tensor) -> Sequence[float]:
        if self.eps_list is not None:
            eps_list = list(self.eps_list)
        elif self.use_epsilon_scaling:
            diameter = self.diameter
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = list(epsilon_schedule(diameter, self.blur, self.scaling, p=2.0))
        else:
            if self.eps is None or self.n_iters is None:
                raise ValueError(
                    "When use_epsilon_scaling=False, provide eps and n_iters."
                )
            eps_list = [float(self.eps)] * int(self.n_iters)

        if self.n_iters is not None:
            eps_list = eps_list[: int(self.n_iters)]
        if len(eps_list) == 0:
            raise ValueError("eps_list is empty after applying n_iters.")
        return eps_list

    def forward(
        self,
        *args: Union[torch.Tensor, float],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        parsed = _process_args(*args, normalize=self.normalize)

        if not parsed.x.is_cuda or not parsed.y.is_cuda:
            raise ValueError("ot_triton.SamplesLoss requires CUDA tensors.")

        class _SinkhornCostFn(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                x: torch.Tensor,
                y: torch.Tensor,
                a: torch.Tensor,
                b: torch.Tensor,
                eps_list: Tuple[float, ...],
                last_extrapolation: bool,
                allow_tf32: bool,
                use_exp2: bool,
                autotune: bool,
                block_m: Optional[int],
                block_n: Optional[int],
                block_k: Optional[int],
                num_warps: Optional[int],
                num_stages: int,
            ) -> torch.Tensor:
                backend = self.backend
                if backend == "auto":
                    if x.shape[1] <= 3 and x.shape[0] * y.shape[0] > 10000**2:
                        backend = "multiscale"
                    else:
                        backend = "online"
                if last_extrapolation:
                    if backend == "multiscale":
                        (
                            f_cost,
                            g_cost,
                            f_grad,
                            g_grad,
                            state,
                        ) = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=True,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            eps=None,
                            n_iters=None,
                            diameter=None,
                            eps_list=eps_list,
                            truncate=self.truncate,
                            cluster_scale=self.cluster_scale,
                            max_coarse_levels=self.max_coarse_levels,
                            blocksparse_backend=self.multiscale_blocksparse_backend,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            autotune=autotune,
                            return_prelast=True,
                            return_state=True,
                        )
                    else:
                        f_cost, g_cost, f_grad, g_grad = sinkhorn_geomloss_online_potentials_sqeuclid(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=True,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            eps=None,
                            n_iters=None,
                            diameter=None,
                            eps_list=eps_list,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            autotune=autotune,
                            return_prelast=True,
                        )
                else:
                    if backend == "multiscale":
                        (
                            f_cost,
                            g_cost,
                            f_grad,
                            g_grad,
                            state,
                        ) = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=False,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            eps=None,
                            n_iters=None,
                            diameter=None,
                            eps_list=eps_list,
                            truncate=self.truncate,
                            cluster_scale=self.cluster_scale,
                            max_coarse_levels=self.max_coarse_levels,
                            blocksparse_backend=self.multiscale_blocksparse_backend,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            autotune=autotune,
                            return_prelast=True,
                            return_state=True,
                        )
                    else:
                        f_cost, g_cost = sinkhorn_geomloss_online_potentials_sqeuclid(
                            x,
                            y,
                            a,
                            b,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=False,
                            allow_tf32=allow_tf32,
                            use_exp2=use_exp2,
                            eps=None,
                            n_iters=None,
                            diameter=None,
                            eps_list=eps_list,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            autotune=autotune,
                        )
                    if backend != "multiscale":
                        f_grad, g_grad = f_cost, g_cost

                if backend == "multiscale":
                    perm_x, perm_y, offsets_x, offsets_y, row_ptr_x, col_idx_x, row_ptr_y, col_idx_y = state
                    ctx.save_for_backward(
                        x,
                        y,
                        a,
                        b,
                        f_cost,
                        g_cost,
                        f_grad,
                        g_grad,
                        perm_x,
                        perm_y,
                        offsets_x,
                        offsets_y,
                        row_ptr_x,
                        col_idx_x,
                        row_ptr_y,
                        col_idx_y,
                    )
                else:
                    ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
                ctx.eps = float(eps_list[-1])
                ctx.allow_tf32 = bool(allow_tf32)
                ctx.use_exp2 = bool(use_exp2)
                ctx.autotune = bool(autotune)
                ctx.backend = backend
                ctx.block_m = block_m
                ctx.block_n = block_n
                ctx.block_k = block_k
                ctx.num_warps = num_warps
                ctx.num_stages = int(num_stages)
                return (a * f_cost).sum() + (b * g_cost).sum()

            @staticmethod
            def backward(ctx, grad_out):
                if ctx.backend == "multiscale":
                    (
                        x,
                        y,
                        a,
                        b,
                        f_cost,
                        g_cost,
                        f_grad,
                        g_grad,
                        perm_x,
                        perm_y,
                        offsets_x,
                        offsets_y,
                        row_ptr_x,
                        col_idx_x,
                        row_ptr_y,
                        col_idx_y,
                    ) = ctx.saved_tensors
                else:
                    x, y, a, b, f_cost, g_cost, f_grad, g_grad = ctx.saved_tensors
                grad_x = grad_y = grad_a = grad_b = None

                if x.requires_grad or y.requires_grad:
                    if ctx.backend == "multiscale":
                        d = x.shape[1]
                        bm = ctx.block_m
                        bn = ctx.block_n
                        bk = ctx.block_k
                        nw = ctx.num_warps
                        if bm is None or bn is None or bk is None or nw is None:
                            bm_d, bn_d, bk_d, nw_d = _dense_default_block_sizes(
                                d, x.dtype, ctx.allow_tf32
                            )
                            bm = bm_d if bm is None else bm
                            bn = bn_d if bn is None else bn
                            bk = bk_d if bk is None else bk
                            nw = nw_d if nw is None else nw
                        if bk < 16:
                            bk = 16
                        gx, gy = _SinkhornGradFnMultiscale.apply(
                            x,
                            y,
                            a,
                            b,
                            f_grad,
                            g_grad,
                            perm_x,
                            perm_y,
                            offsets_x,
                            offsets_y,
                            row_ptr_x,
                            col_idx_x,
                            row_ptr_y,
                            col_idx_y,
                            ctx.eps,
                            ctx.allow_tf32,
                            ctx.use_exp2,
                            int(bm),
                            int(bn),
                            int(bk),
                            int(nw),
                            ctx.num_stages,
                            grad_out,
                            self.hvp_tau2,
                            self.hvp_max_cg_iter,
                            self.hvp_cg_rtol,
                            self.hvp_cg_atol,
                            self.hvp_cg_stabilise_every,
                            self.hvp_preconditioner,
                            self.hvp_precond_terms,
                            self.hvp_use_preconditioner,
                            self.multiscale_blocksparse_backend,
                            x.requires_grad,
                            y.requires_grad,
                        )
                        grad_x = gx if x.requires_grad else None
                        grad_y = gy if y.requires_grad else None
                    else:
                        gx, gy = _SinkhornGradFn.apply(
                            x,
                            y,
                            a,
                            b,
                            f_grad,
                            g_grad,
                            ctx.eps,
                            ctx.allow_tf32,
                            ctx.use_exp2,
                            ctx.autotune,
                            ctx.block_m,
                            ctx.block_n,
                            ctx.block_k,
                            ctx.num_warps,
                            ctx.num_stages,
                            grad_out,
                            self.hvp_tau2,
                            self.hvp_max_cg_iter,
                            self.hvp_cg_rtol,
                            self.hvp_cg_atol,
                            self.hvp_cg_stabilise_every,
                            self.hvp_preconditioner,
                            self.hvp_precond_terms,
                            self.hvp_use_preconditioner,
                            x.requires_grad,
                            y.requires_grad,
                        )
                        grad_x = gx if x.requires_grad else None
                        grad_y = gy if y.requires_grad else None

                if a.requires_grad:
                    grad_a = grad_out * f_cost
                if b.requires_grad:
                    grad_b = grad_out * g_cost

                return (
                    grad_x,
                    grad_y,
                    grad_a,
                    grad_b,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        def _cost(xb: torch.Tensor, yb: torch.Tensor, ab: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
            eps_list = tuple(self._eps_list_for_inputs(xb, yb))
            return _SinkhornCostFn.apply(
                xb,
                yb,
                ab,
                bb,
                eps_list,
                self.last_extrapolation,
                self.allow_tf32,
                self.use_exp2,
                self.autotune,
                self.block_m,
                self.block_n,
                self.block_k,
                self.num_warps,
                self.num_stages,
            )

        # Batched inputs: match GeomLoss by returning a vector of size (B,).
        if parsed.batched:
            if self.potentials:
                f_list = []
                g_list = []
                for xb, yb, ab, bb in zip(parsed.x, parsed.y, parsed.a, parsed.b):
                    eps_list = tuple(self._eps_list_for_inputs(xb, yb))
                    backend = self.backend
                    if backend == "auto":
                        if xb.shape[1] <= 3 and xb.shape[0] * yb.shape[0] > 10000**2:
                            backend = "multiscale"
                        else:
                            backend = "online"
                    if backend == "multiscale":
                        fb, gb = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
                            xb,
                            yb,
                            ab,
                            bb,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=self.last_extrapolation,
                            allow_tf32=self.allow_tf32,
                            use_exp2=self.use_exp2,
                            eps_list=eps_list,
                            truncate=self.truncate,
                            cluster_scale=self.cluster_scale,
                            max_coarse_levels=self.max_coarse_levels,
                            blocksparse_backend=self.multiscale_blocksparse_backend,
                            block_m=self.block_m,
                            block_n=self.block_n,
                            block_k=self.block_k,
                            num_warps=self.num_warps,
                            num_stages=self.num_stages,
                            autotune=self.autotune,
                        )
                    else:
                        fb, gb = sinkhorn_geomloss_online_potentials_sqeuclid(
                            xb,
                            yb,
                            ab,
                            bb,
                            blur=self.blur,
                            scaling=self.scaling,
                            use_epsilon_scaling=False,
                            last_extrapolation=self.last_extrapolation,
                            allow_tf32=self.allow_tf32,
                            use_exp2=self.use_exp2,
                            eps_list=eps_list,
                            block_m=self.block_m,
                            block_n=self.block_n,
                            block_k=self.block_k,
                            num_warps=self.num_warps,
                            num_stages=self.num_stages,
                            autotune=self.autotune,
                        )
                    f_list.append(fb)
                    g_list.append(gb)
                f_b = torch.stack(f_list, dim=0).view(parsed.a_view_shape)
                g_b = torch.stack(g_list, dim=0).view(parsed.b_view_shape)
                return f_b, g_b

            costs = [_cost(xb, yb, ab, bb) for xb, yb, ab, bb in zip(parsed.x, parsed.y, parsed.a, parsed.b)]
            return torch.stack(costs, dim=0)

        if self.potentials:
            eps_list = tuple(self._eps_list_for_inputs(parsed.x, parsed.y))
            backend = self.backend
            if backend == "auto":
                if parsed.x.shape[1] <= 3 and parsed.x.shape[0] * parsed.y.shape[0] > 10000**2:
                    backend = "multiscale"
                else:
                    backend = "online"
            if backend == "multiscale":
                f, g = sinkhorn_geomloss_multiscale_potentials_sqeuclid(
                    parsed.x,
                    parsed.y,
                    parsed.a,
                    parsed.b,
                    blur=self.blur,
                    scaling=self.scaling,
                    use_epsilon_scaling=False,
                    last_extrapolation=self.last_extrapolation,
                    allow_tf32=self.allow_tf32,
                    use_exp2=self.use_exp2,
                    eps_list=eps_list,
                    truncate=self.truncate,
                    cluster_scale=self.cluster_scale,
                    max_coarse_levels=self.max_coarse_levels,
                    blocksparse_backend=self.multiscale_blocksparse_backend,
                    block_m=self.block_m,
                    block_n=self.block_n,
                    block_k=self.block_k,
                    num_warps=self.num_warps,
                    num_stages=self.num_stages,
                    autotune=self.autotune,
                )
            else:
                f, g = sinkhorn_geomloss_online_potentials_sqeuclid(
                    parsed.x,
                    parsed.y,
                    parsed.a,
                    parsed.b,
                    blur=self.blur,
                    scaling=self.scaling,
                    use_epsilon_scaling=False,
                    last_extrapolation=self.last_extrapolation,
                    allow_tf32=self.allow_tf32,
                    use_exp2=self.use_exp2,
                    eps_list=eps_list,
                    block_m=self.block_m,
                    block_n=self.block_n,
                    block_k=self.block_k,
                    num_warps=self.num_warps,
                    num_stages=self.num_stages,
                    autotune=self.autotune,
                )
            return f.view(parsed.a_view_shape), g.view(parsed.b_view_shape)

        return _cost(parsed.x, parsed.y, parsed.a, parsed.b)
