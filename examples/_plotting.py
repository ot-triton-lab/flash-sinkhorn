"""Helpers shared by the flash-sinkhorn examples: figures, timing and sizes.

The palette follows the GeomLoss gallery: source points in red, target points in blue, arrows in green, panels
without ticks. Large clouds are drawn as small markers; arrows only for a random subsample.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SOURCE = (0.95, 0.55, 0.55)
TARGET = (0.55, 0.55, 0.95)
THIRD = (0.55, 0.80, 0.55)
ARROW = "#5CBF3A"


def require_cuda() -> torch.device:
    """The CUDA device. Exits with a short message when there is none: the solvers are Triton kernels."""
    if not torch.cuda.is_available():
        sys.exit("This example needs a CUDA GPU: flash-sinkhorn runs Triton kernels.")
    return torch.device("cuda")


def images_dir(script: str) -> Path:
    """The images/ folder next to the given script, created if missing."""
    folder = Path(script).resolve().parent / "images"
    folder.mkdir(exist_ok=True)
    return folder


def to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().double().cpu().numpy()
    return np.asarray(t, dtype=float)


def timed(label, fn):
    """Run fn twice; print the first wall time (compilation and autotuning included), the second, and the peak GPU
    memory of the second run. Returns the second result."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    warm = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{label}: first call {first:.2f} s, then {warm:.2f} s, peak memory {peak:.2f} GiB "
          f"on {torch.cuda.get_device_name()}")
    return result


def dense_size(n, m) -> str:
    """The memory of a dense float32 n x m matrix, which flash-sinkhorn never forms."""
    return f"a dense float32 {n:,} x {m:,} matrix would need {4 * n * m / 2**30:.1f} GiB"


def subsample(n, k, seed=0, device=None) -> torch.Tensor:
    """k distinct random indices out of n, the same for the same seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:k].to(device)


def cloud(ax, x, color, s=0.5, alpha=0.5, **kwargs):
    """A large cloud as small markers without edges."""
    x = to_numpy(x)
    return ax.scatter(x[:, 0], x[:, 1], s=s, color=color, alpha=alpha, edgecolors="none", rasterized=True,
                      **kwargs)


def arrows(ax, x, v, color=ARROW, width=0.004, **kwargs):
    """Displacements v drawn from the points x, in data units."""
    x, v = to_numpy(x), to_numpy(v)
    return ax.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1], color=color, angles="xy", scale_units="xy", scale=1,
                     width=width, **kwargs)


def stripes(x, frequency=10.0) -> np.ndarray:
    """RGBA colours from cos(f x) cos(f y): nearby points share a colour, so tearing and folding stay visible."""
    x = to_numpy(x)
    t = np.cos(frequency * x[:, 0]) * np.cos(frequency * x[:, 1])
    return plt.cm.hsv((t + 1.0) / 2.0)


def small_multiples(rows, cols, size=3.0, aspect=1.0):
    """A grid of equal-aspect panels without ticks, each `size` wide and `size * aspect` high: pass the data's
    height / width as `aspect` so that the panels fill their cells."""
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * aspect * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def fit_limits(ax, *point_sets, margin=0.05):
    """Axis limits that contain every given point, arrow tips included, with a margin relative to the extent."""
    points = np.concatenate([to_numpy(p)[:, :2] for p in point_sets])
    low, high = points.min(axis=0), points.max(axis=0)
    pad = margin * (high - low).max()
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)


def table(ax, header, rows, title=None, fontsize=10):
    """A small table drawn in place of a plot: a handful of numbers reads better as a table than as a curve."""
    ax.axis("off")
    cells = ax.table(cellText=[[str(value) for value in row] for row in rows], colLabels=list(header),
                     loc="center", cellLoc="center")
    cells.auto_set_font_size(False)
    cells.set_fontsize(fontsize)
    cells.auto_set_column_width(list(range(len(header))))
    cells.scale(1.0, 1.5)
    for (row, _), cell in cells.get_celld().items():
        cell.set_edgecolor("0.85")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.94")
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    return cells


def save(fig, folder, name) -> Path:
    path = Path(folder) / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


def figure_frame(fig) -> np.ndarray:
    """The rendered figure as an RGB uint8 array, for save_gif."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, folder, name, fps=10) -> Path:
    """An animated GIF from equally sized RGB uint8 frames."""
    from PIL import Image  # installed with matplotlib

    images = [Image.fromarray(frame) for frame in frames]
    path = Path(folder) / f"{name}.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return path
