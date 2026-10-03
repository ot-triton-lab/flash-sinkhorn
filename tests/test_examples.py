"""GPU tests for the gallery: the plan-application helpers, and every example run end to end.

They need a CUDA GPU and matplotlib. The file is outside the default test paths, so run it explicitly:
    python -m pytest tests/test_examples.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the examples need a CUDA GPU")

# Each script, relative to examples/, and the figures it must rewrite in its group's images/ folder.
EXAMPLES = {
    "getting_started/plot_sinkhorn_basics.py": ["basics.png"],
    "getting_started/plot_blur.py": ["blur_maps.png", "blur_translation.png"],
    "getting_started/plot_gradient_flow_2d.py": ["gradient_flow.png", "gradient_flow.gif"],
    "choosing_parameters/plot_conventions.py": ["conventions.png"],
    "choosing_parameters/plot_unbalanced.py": ["unbalanced.png", "unbalanced_reach.png"],
    "choosing_parameters/plot_convergence.py": ["convergence.png"],
    "scale/plot_multiscale_3d.py": ["multiscale_3d.png"],
    "scale/plot_many_problems.py": ["many_problems.png"],
    "scale/plot_embeddings.py": ["embeddings.png"],
    "advanced/plot_hvp_newton.py": ["hvp_newton.png"],
    "advanced/plot_semi_dual.py": ["semi_dual.png"],
    "advanced/plot_label_cost.py": ["label_cost.png"],
    "advanced/plot_attribute_transfer.py": ["colour_transfer.png", "label_transfer.png"],
}
DENSE_LINE = re.compile(r"dense float32 ([\d,]+) x ([\d,]+) matrix would need")


@pytest.fixture(scope="module")
def solved():
    """Potentials of a small problem and the dense float64 plan they define, as the reference."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, eps, cost_scale = 400, 500, 0.09, 0.5
    x = torch.randn(n, 2, device="cuda")
    y = torch.randn(m, 2, device="cuda") + 1.0
    a = torch.rand(n, device="cuda") + 0.5
    b = torch.rand(m, device="cuda") + 0.5
    a, b = a / a.sum(), b / b.sum()
    f, g = SamplesLoss(blur=eps**0.5, half_cost=True, debias=False, backend="symmetric", potentials=True,
                       allow_tf32=False)(a, x, b, y)
    C = cost_scale * torch.cdist(x.double(), y.double()) ** 2
    P = a.double()[:, None] * b.double()[None, :] * torch.exp((f.double()[:, None] + g.double()[None, :] - C) / eps)
    return dict(x=x, y=y, f=f, g=g, a=a, b=b, eps=eps, cost_scale=cost_scale, P=P)


def _args(s):
    return (s["x"], s["y"], s["f"], s["g"], s["a"], s["b"])


@needs_gpu
@pytest.mark.parametrize("width", [None, 2, 5])
def test_transport_apply_plan_matches_the_dense_plan(solved, width):
    import _transport as T

    m = len(solved["y"])
    v = torch.randn(m, device="cuda") if width is None else torch.randn(m, width, device="cuda")
    out = T.apply_plan(*_args(solved), v, eps=solved["eps"], cost_scale=solved["cost_scale"])
    torch.testing.assert_close(out.double(), solved["P"] @ v.double(), rtol=1e-4, atol=1e-7)


@needs_gpu
def test_transport_apply_plan_transpose_matches_the_dense_plan(solved):
    import _transport as T

    u = torch.randn(len(solved["x"]), device="cuda")
    out = T.apply_plan_transpose(*_args(solved), u, eps=solved["eps"], cost_scale=solved["cost_scale"])
    torch.testing.assert_close(out.double(), solved["P"].T @ u.double(), rtol=1e-4, atol=1e-7)


@needs_gpu
def test_transport_ignores_a_common_translation(solved):
    import _transport as T

    v = torch.randn(len(solved["y"]), device="cuda")
    shift = torch.tensor([100.0, -50.0], device="cuda")
    near = T.apply_plan(*_args(solved), v, eps=solved["eps"], cost_scale=solved["cost_scale"])
    x, y, f, g, a, b = _args(solved)
    far = T.apply_plan(x + shift, y + shift, f, g, a, b, v, eps=solved["eps"], cost_scale=solved["cost_scale"])
    torch.testing.assert_close(far, near, rtol=1e-3, atol=1e-7)


@needs_gpu
def test_transport_barycentric_targets_average_the_target_points(solved):
    import _transport as T

    out = T.barycentric_targets(*_args(solved), eps=solved["eps"], cost_scale=solved["cost_scale"])
    P = solved["P"]
    expected = (P @ solved["y"].double()) / P.sum(dim=1, keepdim=True)
    torch.testing.assert_close(out.double(), expected, rtol=1e-4, atol=1e-6)


def run(script, cwd, **env):
    return subprocess.run([sys.executable, str(ROOT / "examples" / script)], cwd=cwd, capture_output=True,
                          text=True, timeout=1800, env=dict(os.environ, MPLBACKEND="Agg", **env))


@needs_gpu
@pytest.mark.parametrize("script", sorted(EXAMPLES))
def test_example_runs_at_scale_and_rewrites_its_figures(script, tmp_path):
    images = (ROOT / "examples" / script).parent / "images"
    before = {name: (images / name).stat().st_mtime_ns if (images / name).exists() else None
              for name in EXAMPLES[script]}
    result = run(script, cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]
    for name, mtime in before.items():
        assert (images / name).exists(), f"{script} did not write images/{name}"
        assert mtime is None or (images / name).stat().st_mtime_ns > mtime, f"{script} did not rewrite {name}"
    assert "peak memory" in result.stdout, f"{script} does not report peak memory"
    sizes = [(int(n.replace(",", "")), int(m.replace(",", ""))) for n, m in DENSE_LINE.findall(result.stdout)]
    assert sizes, f"{script} does not print the size of the dense matrix it avoids"
    assert all(min(n, m) >= 10_000 for n, m in sizes), f"{script} runs below 10^4 points: {sizes}"


def test_examples_say_when_there_is_no_gpu(tmp_path):
    result = run("getting_started/plot_sinkhorn_basics.py", cwd=tmp_path, CUDA_VISIBLE_DEVICES="")
    assert result.returncode != 0
    assert "needs a CUDA GPU" in result.stderr
