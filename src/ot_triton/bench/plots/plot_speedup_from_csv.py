"""
Plot speedup curves from benchmark CSVs.

Given a CSV produced by the benchmark scripts in this repo with columns:
  Npoints geomloss_time ot_triton_time
this script creates:
  - <csv_stem>_speedup.csv (Npoints, geomloss/ot_triton)
  - <csv_stem>_speedup.png (log-log plot)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _load_csv(path: Path) -> np.ndarray:
    header = path.read_text().splitlines()[0]
    delimiter = "," if "," in header else None
    return np.loadtxt(path, skiprows=1, delimiter=delimiter)


def _save_speedup(path: Path) -> None:
    data = _load_csv(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns.")

    n = data[:, 0]
    geomloss = data[:, 1]
    triton = data[:, 2]
    speedup = geomloss / triton

    out_csv = path.with_name(path.stem + "_speedup.csv")
    out_png = path.with_name(path.stem + "_speedup.png")

    np.savetxt(
        out_csv,
        np.stack([n, speedup], axis=1),
        fmt="%-9.6f",
        delimiter=",",
        header="Npoints,speedup_geomloss_over_ot_triton",
        comments="",
    )

    plt.figure()
    plt.plot(n, speedup, "o-", linewidth=2)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.title(f"Speedup (GeomLoss / ot_triton): {path.name}")
    plt.xlabel("Number of samples per measure")
    plt.ylabel("Speedup (>1 means ot_triton faster)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved {out_csv}")
    print(f"Saved {out_png}")


def _expand_paths(args: List[str]) -> Iterable[Path]:
    for arg in args:
        p = Path(arg)
        if "*" in arg or "?" in arg or "[" in arg:
            for match in sorted(Path().glob(arg)):
                yield match
        else:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot speedup from benchmark CSVs.")
    parser.add_argument("csv", nargs="+", help="Input CSV path(s) or glob(s).")
    args = parser.parse_args()

    for path in _expand_paths(args.csv):
        _save_speedup(path)


if __name__ == "__main__":
    main()
