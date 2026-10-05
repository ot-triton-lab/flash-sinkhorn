"""CPU checks for generated notebooks, including execution in a real Jupyter kernel.

Run explicitly with the notebook dependencies installed:
    python -m pytest tests/test_notebooks.py
"""

import ast
import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "examples").glob("*/plot_*.py")) + [
    ROOT / "examples/getting_started/first_steps.md",
    ROOT / "examples/choosing_parameters/guide.md",
]


@pytest.fixture
def converter():
    path = ROOT / "examples/_notebooks.py"
    assert path.is_file(), "The notebook generator is missing"
    spec = importlib.util.spec_from_file_location("gallery_notebooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_text(cell):
    return "".join(cell["source"])


def computational_code(notebook):
    return "\n\n".join(source_text(cell) for cell in notebook["cells"]
                       if "source-code" in cell["metadata"].get("tags", []))


def without_docstring(code):
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def test_python_sections_keep_strings_and_functions_intact(converter):
    code = '''"""Example title
=============
"""
# %%
# Setup
# -----
# Explain the first step.
text = """literal
# %%
still literal"""
def twice(value):
    # %% is a comment inside the function
    return value * 2
# %%
# Result
# ------
print(twice(2), text)
'''
    cells = converter.split_python(code)
    assert [kind for kind, _ in cells] == ["markdown", "markdown", "code", "markdown", "code"]
    assert "Explain the first step." in cells[1][1]
    assert "still literal" in cells[2][1]
    assert without_docstring(code) == without_docstring("\n".join(text for kind, text in cells if kind == "code"))


def test_markdown_splits_python_but_preserves_other_fences(converter):
    text = '# Title\n\n```bash\npip install something\n```\n\n```python\nx = 1\n```\n\nExpected:\n```text\n1\n```\n'
    cells = converter.split_markdown(text)
    assert [kind for kind, _ in cells] == ["markdown", "code", "markdown"]
    assert "```bash" in cells[0][1]
    assert cells[1][1].strip() == "x = 1"
    assert "```text" in cells[2][1]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebooks_preserve_code_and_label_reference_figures(converter, source):
    nbformat = pytest.importorskip("nbformat")
    generated = converter.make_notebook(source, ROOT)
    path = source.with_suffix(".ipynb")
    assert path.is_file(), path
    committed = json.loads(path.read_text())
    assert committed == generated, f"Regenerate {path.name} from its source"
    nbformat.validate(nbformat.read(path, as_version=4))
    assert f"Source: [{source.name}]({source.name})" in source_text(committed["cells"][1])
    code = computational_code(committed)
    if source.suffix == ".py":
        assert without_docstring(code) == without_docstring(source.read_text())
    else:
        blocks = re.findall(r"^```python\n(.*?)^```", source.read_text(), re.S | re.M)
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse("\n\n".join(blocks)))
    references = []
    for cell in committed["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(source_text(cell), source.name, "exec")
        if cell.get("attachments"):
            references.append(cell)
            assert "Reference figure" in source_text(cell)
            for data in cell["attachments"].values():
                assert all(base64.b64decode(value) for value in data.values())
        if cell["cell_type"] == "markdown":
            for link in re.findall(r"\]\(([^)]+)\)", source_text(cell)):
                if link.startswith("attachment:"):
                    assert link.removeprefix("attachment:") in cell["attachments"]
                elif not link.startswith(("https://", "http://", "#")):
                    assert (source.parent / link.split("#")[0]).is_file(), link
    assert references, "Keep the existing figures available before running the notebook"
    assert "previous script runs" in "\n".join(source_text(c) for c in committed["cells"])
    if source.stem == "plot_gradient_flow_2d":
        assert any("image/gif" in data for cell in references for data in cell["attachments"].values())
        assert any("gradient_flow.gif" in source_text(c) for c in committed["cells"] if c["cell_type"] == "code")
    if source.stem == "guide":
        assert not any(c["cell_type"] == "code" for c in committed["cells"])


def test_generator_check_detects_stale_sources(converter, tmp_path):
    root, source = plotting_checkout(tmp_path)
    converter.generate(root)
    assert converter.generate(root, check=True) == []
    source.write_text(source.read_text().replace("print('finished')", "print('updated')"))
    assert converter.generate(root, check=True) == [source.with_suffix(".ipynb")]


def plotting_checkout(tmp_path):
    """A small CPU example using the gallery's actual plotting and path helpers."""
    from PIL import Image

    root = tmp_path / "checkout"
    folder = root / "examples/getting_started"
    images = folder / "images"
    images.mkdir(parents=True)
    (root / "torch-ext/flash_sinkhorn").mkdir(parents=True)
    shutil.copyfile(ROOT / "examples/_plotting.py", root / "examples/_plotting.py")
    Image.new("RGB", (4, 4), "white").save(images / "result.png")
    Image.new("RGB", (4, 4), "white").save(images / "closed.png")
    Image.new("RGB", (4, 4), "white").save(images / "motion.gif")
    source = folder / "plot_cpu.py"
    source.write_text('''"""A CPU example
=============
"""
# %%
# Plot and print
# --------------
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from _plotting import images_dir, save, save_gif
assert Path.cwd() == Path(__file__).resolve().parents[2]
fig, ax = plt.subplots()
ax.plot([0, 1], [1, 0])
save(fig, images_dir(__file__), "result")
fig.savefig(images_dir(__file__) / "closed.png")
plt.close(fig)
frames = [np.asarray(Image.new("RGB", (8, 8), color)) for color in ("red", "blue")]
save_gif(frames, images_dir(__file__), "motion")
print('finished')
''')
    return root, source


@pytest.mark.parametrize("nested", [False, True], ids=["root", "notebook-folder"])
@pytest.mark.parametrize("closed", [False, True], ids=["unclosed", "already-closed"])
def test_real_kernel_displays_fresh_png_and_gif_and_can_run_twice(converter, tmp_path, nested, closed):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    root, source = plotting_checkout(tmp_path)
    if not closed:
        source.write_text(source.read_text().replace("plt.close(fig)", "plt.show()"))
    notebook = nbformat.reads(json.dumps(converter.make_notebook(source, root)), as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("assert not plt.get_fignums(), plt.get_fignums()"))
    # Repeating all cells in one kernel checks that setup and imports are reusable.
    notebook.cells += [nbformat.from_dict(json.loads(json.dumps(cell))) for cell in notebook.cells]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"cell-{index}"
    client = nbclient.NotebookClient(notebook, timeout=90, kernel_name="python3")
    client.execute(cwd=str(source.parent if nested else root))
    outputs = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs]
    stdout = "".join(out.get("text", "") for out in outputs)
    assert stdout.count("finished") == 2
    for mime in ("image/png", "image/gif"):
        displays = [out.data[mime] for out in outputs if mime in out.get("data", {})]
        assert len(displays) == (4 if mime == "image/png" else 2)
        filename = "closed.png" if mime == "image/png" else "motion.gif"
        assert base64.b64decode(displays[-1]) == (source.parent / "images" / filename).read_bytes()


@pytest.mark.parametrize("name", ["plot_sinkhorn_basics.py", "first_steps.md"])
def test_notebook_setup_uses_installed_package_then_reports_missing_cuda(converter, name, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    source = ROOT / "examples/getting_started" / name
    installed = tmp_path / "site-packages"
    shutil.copytree(ROOT / "torch-ext/flash_sinkhorn", installed / "flash_sinkhorn",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    notebook = nbformat.reads(json.dumps(converter.make_notebook(source, ROOT)), as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook.cells = code[:2]
    notebook.cells.insert(1, nbformat.v4.new_code_cell(
        "import flash_sinkhorn\n"
        f"assert Path(flash_sinkhorn.__file__).resolve().is_relative_to(Path({str(installed)!r}))"
    ))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(installed))
    client = nbclient.NotebookClient(notebook, timeout=90, kernel_name="python3", allow_errors=True)
    client.execute(cwd=str(source.parent), env=env)
    errors = [out for cell in notebook.cells for out in cell.get("outputs", []) if out.output_type == "error"]
    assert len(errors) == 1, errors
    assert "CUDA GPU" in errors[0].evalue


def test_index_offers_all_notebooks_and_keeps_script_links():
    index = (ROOT / "examples/README.md").read_text()
    for source in SOURCES:
        relative = source.relative_to(ROOT / "examples")
        assert f"]({relative.with_suffix('.ipynb').as_posix()})" in index
        if source.suffix == ".py":
            assert f"]({relative.as_posix()})" in index
    assert 'flash-sinkhorn[notebooks]' in index
    assert sorted((ROOT / "examples").glob("*/*.ipynb")) == sorted(s.with_suffix(".ipynb") for s in SOURCES)


def test_notebook_extra_installs_the_ui_and_execution_dependencies():
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert {"matplotlib", "jupyterlab", "nbclient"} <= set(extras["notebooks"])


def test_cross_notebook_fragments_resolve_in_rendered_html():
    nbformat = pytest.importorskip("nbformat")
    nbconvert = pytest.importorskip("nbconvert")
    exporter = nbconvert.HTMLExporter()
    rendered = {}
    checked = 0
    for source in SOURCES:
        notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "markdown":
                continue
            for target, fragment in re.findall(r"\]\(([^)#]+\.ipynb)#([^)]+)\)", cell.source):
                path = (source.parent / target).resolve()
                if path not in rendered:
                    html, _ = exporter.from_notebook_node(nbformat.read(path, as_version=4))
                    rendered[path] = set(re.findall(r'\bid="([^"]+)"', html))
                assert fragment in rendered[path], f"{target}#{fragment} has no matching rendered anchor"
                checked += 1
    assert checked > 0


def test_generator_cli_reports_current_artifacts():
    result = subprocess.run([sys.executable, "examples/_notebooks.py", "--check"], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "15 notebooks are up to date" in result.stdout
