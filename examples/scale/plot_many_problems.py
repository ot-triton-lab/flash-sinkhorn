"""
Many problems: batches and changing sizes
=========================================

**When to use:** a training loop solves OT many times, on a batch of point clouds at once or on clouds whose size
changes from one step to the next.

**What you will see:** 16 problems of 10,000 points solved in one call and one by one, and the time of each call
over a sequence of changing sizes, with autotuning on and off.
"""

# %%
# Setup
# -----
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flash_sinkhorn import SamplesLoss

sys.path.append(str(Path(__file__).resolve().parents[1]))  # the gallery's shared helpers
from _plotting import dense_size, images_dir, require_cuda, save, table, timed  # noqa: E402

device = require_cuda()
torch.manual_seed(0)

# %%
# A batch of problems
# -------------------
# Inputs of shape (B, N, D) describe B problems, and the loss returns one value per problem. A batch is a
# convenience: compare the time of one batched call with the loop over the same problems.
B, N = 16, 10_000
xb = torch.rand(B, N, 2, device=device)
yb = torch.rand(B, N, 2, device=device) + 0.1
print(dense_size(N, N))
loss = SamplesLoss(blur=0.05, half_cost=True, debias=True, backend="symmetric", autotune=False)
batched = timed("16 problems in one call", lambda: loss(xb, yb))
looped = timed("16 problems one by one", lambda: torch.stack([loss(xb[i], yb[i]) for i in range(B)]))
print(f"values of shape {tuple(batched.shape)}; largest difference between the two: "
      f"{(batched - looped).abs().max().item():.1e}")

# %%
# Changing sizes and autotuning
# -----------------------------
# With ``autotune=True`` (the default), the first call for a new size bucket (key n // 32; here the first four calls
# meet new buckets) benchmarks kernel configurations, compiling those not yet in the cache; later calls in that
# bucket reuse the choice, within the same process. With ``autotune=False`` no call is tuned; compare the warm
# calls to judge the benefit on these sizes. If new size buckets keep appearing, turn tuning off.
# ``pad_to_multiple=k`` pads every cloud to a multiple of k with zero-weight points, so that fewer distinct sizes
# reach the tuner, at the cost of the padded points; measure it on your sizes before relying on it.
sizes = [10_000, 13_000, 17_000, 22_000] * 2
timeline = {}
for autotune in (True, False):
    loss = SamplesLoss(blur=0.05, half_cost=True, debias=True, backend="symmetric", autotune=autotune)
    times = []
    for size in sizes:
        x = torch.rand(size, 2, device=device)
        y = torch.rand(size, 2, device=device) + 0.1
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss(x, y)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    timeline[autotune] = times
    print(f"autotune={autotune}: " + ", ".join(f"{size:,} points {t:.2f} s" for size, t in zip(sizes, times)))

fig, ax = plt.subplots(figsize=(6.0, 3.9))
table(ax, ["call", "points", "autotune=True", "autotune=False"],
      [[str(i + 1), f"{size:,}", f"{on:.2f} s", f"{off:.2f} s"]
       for i, (size, on, off) in enumerate(zip(sizes, timeline[True], timeline[False]))],
      title=f"seconds per call ({torch.cuda.get_device_name()})")
save(fig, images_dir(__file__), "many_problems")
plt.show()
