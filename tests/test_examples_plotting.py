"""Unit tests for the plotting helpers shared by the examples (CPU only)."""

import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import _plotting as P  # noqa: E402


def test_dense_size_states_the_float32_matrix_size():
    assert P.dense_size(100_000, 100_000) == "a dense float32 100,000 x 100,000 matrix would need 37.3 GiB"


def test_subsample_is_reproducible_and_has_no_repeats():
    first, again = P.subsample(1000, 300, seed=3), P.subsample(1000, 300, seed=3)
    assert torch.equal(first, again)
    assert len(set(first.tolist())) == 300
    assert int(first.max()) < 1000


def test_small_multiples_sizes_panels_to_the_data_aspect():
    fig, axes = P.small_multiples(2, 4, size=3.0, aspect=0.5)
    assert axes.shape == (2, 4)
    assert tuple(fig.get_size_inches()) == pytest.approx((12.0, 3.0))


def test_fit_limits_keeps_every_point_and_arrow_tip_inside():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    points = np.array([[0.0, 0.0], [1.0, 0.5]])
    tips = np.array([[2.0, -1.0]])
    P.fit_limits(ax, points, tips, margin=0.1)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert x0 < 0.0 and x1 > 2.0 and y0 < -1.0 and y1 > 0.5
    plt.close(fig)


def test_examples_request_autotune_false_except_the_tuning_demo():
    # Autotuning benchmarks kernel configurations whenever a process meets a new tuning key (size bucket,
    # dimension, precision): a script run once does not win that back. The examples say when to turn it on.
    # The multiscale backend tunes its own kernels and ignores the flag, so its calls must not pass it.
    import ast

    tuned = {"SamplesLoss", "apply_plan_mat_flashstyle"}
    exempt = {"plot_many_problems.py"}  # it compares autotuning on and off on purpose
    calls, offenders = 0, []
    for path in sorted((Path(__file__).resolve().parents[1] / "examples").rglob("*.py")):
        if path.name in exempt:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in tuned:
                continue
            calls += 1
            keywords = {k.arg: k.value for k in node.keywords}
            backend, flag = keywords.get("backend"), keywords.get("autotune")
            if isinstance(backend, ast.Constant) and backend.value == "multiscale":
                if flag is not None:
                    offenders.append(f"{path.name}:{node.lineno} (multiscale ignores autotune)")
            elif not (isinstance(flag, ast.Constant) and flag.value is False):
                offenders.append(f"{path.name}:{node.lineno}")
    assert calls > 0
    assert not offenders, offenders


def test_table_has_a_header_row_and_one_row_per_entry():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cells = P.table(ax, ["blur", "residual"], [["0.1", "1e-3"], ["1", "2e-4"], ["10", "5e-6"]])
    assert len(cells.get_celld()) == 4 * 2
    assert cells[0, 0].get_text().get_text() == "blur"
    assert cells[3, 1].get_text().get_text() == "5e-6"
    plt.close(fig)


def test_stripes_gives_one_rgba_colour_per_point():
    colours = P.stripes(np.random.default_rng(0).random((50, 2)))
    assert colours.shape == (50, 4)
    assert 0.0 <= colours.min() and colours.max() <= 1.0


def test_save_gif_writes_every_frame(tmp_path):
    frames = [np.full((8, 8, 3), 40 * k, dtype=np.uint8) for k in range(5)]
    assert Image.open(P.save_gif(frames, tmp_path, "frames")).n_frames == 5


def test_images_dir_is_next_to_the_script(tmp_path):
    script = tmp_path / "group" / "plot_example.py"
    script.parent.mkdir()
    script.write_text("")
    assert P.images_dir(str(script)) == script.parent / "images"
    assert (script.parent / "images").is_dir()
