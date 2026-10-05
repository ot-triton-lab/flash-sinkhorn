# Changelog

## 0.4.1

- Correct unbalanced and semi-unbalanced OT values and derivatives, including half-cost scaling and convergence
  reporting. Centre dense inputs in FP32 to reduce sensitivity to a common coordinate offset.
- Add `SamplesLoss(truncate_tf32=None)`, which follows `allow_tf32`. By default, the symmetric and alternating
  backends round centred FP32/FP64 input coordinates to TF32 before padding, so norms and dot products use the
  same rounded cloud. FP32 arithmetic error remains. Rounding has an identity derivative for gradients and
  supported Hessian-vector products; fp16/bf16 inputs receive no additional rounding.
- Set `truncate_tf32=False` to keep centred dense coordinates unrounded. Set `allow_tf32=False` for FP32
  dot products and no default rounding. Multiscale always rounds and rejects `truncate_tf32=False`.
- Round c-transform coordinates before TF32 dot products, and keep the fused Sinkhorn kernel when label costs
  are active.
- Update the API documentation and add 13 runnable examples covering costs, gradients, parameter choices,
  convergence, large point clouds, embeddings, label costs, Hessian-vector products, and semi-dual transport.
- Provide Jupyter notebooks for all 13 examples and two introductory lessons, with reference figures,
  per-section code cells, and a generator that keeps them synchronized with their Python and Markdown sources.

Dense costs and weight gradients assume probability input weights. With `normalize=False`, each input measure
must already sum to one; arbitrary input mass totals remain unsupported for these outputs.

Version 0.4.1 is available on [PyPI](https://pypi.org/project/flash-sinkhorn/0.4.1/), GitHub `main`, and the
[Hugging Face Kernels Hub](https://huggingface.co/kernels/yexf308/flash-sinkhorn/tree/v1).
