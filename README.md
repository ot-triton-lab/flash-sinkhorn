<p align="center">
  <img src="https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/FlashSinkhorn.png" alt="FlashSinkhorn" width="100%">
</p>

# FlashSinkhorn

[![PyPI](https://img.shields.io/pypi/v/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![Python](https://img.shields.io/pypi/pyversions/flash-sinkhorn)](https://pypi.org/project/flash-sinkhorn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Kernels](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kernels%20Hub-ffb000)](https://huggingface.co/kernels/yexf308/flash-sinkhorn)
[![Kernel build](https://github.com/ot-triton-lab/flash-sinkhorn/actions/workflows/build-kernel.yml/badge.svg?branch=main)](https://github.com/ot-triton-lab/flash-sinkhorn/actions/workflows/build-kernel.yml)

**Entropic optimal transport in PyTorch + Triton, without storing the pairwise cost matrix.**

This package provides the streaming kernels of [FlashSinkhorn](https://arxiv.org/abs/2602.03067)
and the multiscale, block-sparse method of [FlashSinkhorn 2](https://arxiv.org/abs/2610.02395).
Dense backends support balanced and relaxed-marginal OT, gradients, supported Hessian-vector products,
and transport-plan application. A streaming c-transform supports semi-dual objectives.

**v0.4.1:** OT/derivative corrections, TF32 coordinate rounding and 15 Jupyter notebooks.
[Release notes](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/CHANGELOG.md).

## Install

```bash
pip install --upgrade flash-sinkhorn
```

Requires Python ≥3.9, PyTorch ≥2.5, Triton ≥3.1 and an NVIDIA CUDA GPU.

## Quick Start

```python
import torch
from flash_sinkhorn import SamplesLoss

x = torch.randn(4096, 64, device="cuda", requires_grad=True)
y = torch.randn(4096, 64, device="cuda")

loss = SamplesLoss(blur=0.1, half_cost=True, debias=True, autotune=False)
value = loss(x, y)
grad_x = torch.autograd.grad(value, x)[0]
```

`half_cost=True` selects squared distance divided by two; `debias=True` gives the balanced Sinkhorn
divergence. For large balanced 3-D clouds, use `backend="multiscale"`; it currently returns forward costs
and potentials, with no gradients. See the
[multiscale notebook](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/scale/plot_multiscale_3d.ipynb).

By default, dense backends round centred FP32/FP64 coordinates to TF32 before cost evaluation;
FP32 arithmetic error remains. `truncate_tf32=None` follows `allow_tf32`; use `allow_tf32=False`
for FP32 dot products without default coordinate rounding.

The [API reference](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/API.md) covers weights, batching,
unbalanced OT, stopping criteria, precision controls, HVP limits and low-level kernels.

## Examples

[![Gradient flow of 30,000 particles](https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/examples/getting_started/images/gradient_flow.gif)](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/README.md)

Start with [From two point clouds to a differentiable loss](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/getting_started/first_steps.ipynb),
then [Choose parameters and check the result](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/choosing_parameters/guide.ipynb).
The [gallery](https://github.com/ot-triton-lab/flash-sinkhorn/blob/main/examples/README.md) contains 15 notebooks
covering gradients, transport maps, relaxed marginals, embeddings, batches and large 3-D clouds.

```bash
pip install --upgrade "flash-sinkhorn[notebooks]"
git clone https://github.com/ot-triton-lab/flash-sinkhorn.git
cd flash-sinkhorn
jupyter lab examples/getting_started/first_steps.ipynb
```

The examples use the installed package. Reference figures come from previous script runs; running the cells
produces fresh outputs. Each application also has a Python script; install `flash-sinkhorn[examples]`
for its plotting dependencies.

## Hugging Face Kernels Hub

Install `kernels` with pip, then load the [Hub package](https://huggingface.co/kernels/yexf308/flash-sinkhorn/tree/v1):

```python
from kernels import get_kernel

flash_sinkhorn = get_kernel("yexf308/flash-sinkhorn", version=1)
loss = flash_sinkhorn.SamplesLoss(blur=0.1, half_cost=True, debias=True)
```

Hub API version `1` contains Python package version `0.4.1`.

## Citation

Cite [FlashSinkhorn](https://arxiv.org/abs/2602.03067) for the dense streaming kernels and
[FlashSinkhorn 2](https://arxiv.org/abs/2610.02395) for the multiscale method.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{ye2026flashsinkhorn,
  title={FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU},
  author={Ye, Felix X.-F. and Li, Xingjie and Yu, An and Chang, Ming-Ching and Chu, Linsong and Wertheimer, Davis},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026},
  url={https://arxiv.org/abs/2602.03067}
}

@misc{ye2026flashsinkhorn2,
  title={FlashSinkhorn 2: Block-Sparse Entropic Optimal Transport},
  author={Ye, Felix X.-F. and Lim, Yu Chin Fabian and Wang, Naigang and Wertheimer, Davis},
  year={2026},
  eprint={2610.02395},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.02395}
}
```

</details>

## License

MIT
