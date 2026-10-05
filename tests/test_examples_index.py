"""CPU tests of the examples index (examples/README.md) against the scripts and the package API.

The file is outside the default test paths, so run it explicitly:
    python -m pytest tests/test_examples_index.py
"""

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
INDEX = EXAMPLES / "README.md"


def scripts():
    return sorted(p.relative_to(EXAMPLES).as_posix() for p in EXAMPLES.glob("*/plot_*.py"))


def table_rows():
    """Rows of the situation tables: (situation, link to the script, key call, thumbnail)."""
    rows = []
    for line in INDEX.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1].startswith("["):
            rows.append(cells)
    return rows


def repository_package():
    """This repository's flash_sinkhorn, not whichever version is installed."""
    pytest.importorskip("triton")
    source = str(ROOT / "torch-ext")
    sys.path.insert(0, source)
    try:
        import flash_sinkhorn
    finally:
        sys.path.remove(source)
    assert Path(flash_sinkhorn.__file__).resolve().is_relative_to(Path(source).resolve()), flash_sinkhorn.__file__
    return flash_sinkhorn


def test_index_has_one_row_per_script():
    linked = [re.search(r"\]\(([^)]+\.py)\)", row[1]).group(1) for row in table_rows()]
    assert sorted(linked) == scripts()


def test_index_images_exist():
    for row in table_rows():
        assert re.search(r'<img src="([^"]+)"', row[3]), row[0]
    text = INDEX.read_text()
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) + re.findall(r'<img src="([^"]+)"', text)
    assert images
    for image in images:
        assert (EXAMPLES / image).is_file(), image


def test_index_key_calls_name_real_parameters():
    flash_sinkhorn = repository_package()
    for row in table_rows():
        code = re.search(r"`([^`]+)`", row[2]).group(1)
        called = 0
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and hasattr(flash_sinkhorn, node.func.id):
                called += 1
                accepted = inspect.signature(getattr(flash_sinkhorn, node.func.id)).parameters
                for keyword in node.keywords:
                    assert keyword.arg in accepted, (node.func.id, keyword.arg)
        assert called, code


RAW = "https://raw.githubusercontent.com/ot-triton-lab/flash-sinkhorn/main/"


def test_readme_examples_section_uses_absolute_urls():
    """PyPI renders the README without the repository, so its images and links must be absolute."""
    section = (ROOT / "README.md").read_text().split("\n## Examples\n", 1)[1].split("\n## ", 1)[0]
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section)
    assert images and all(url.startswith(RAW) for url in images)
    for url in images:
        assert (ROOT / url[len(RAW):]).is_file(), url
    links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", section)
    assert links and all(url.startswith("https://") for url in links)


def test_pyproject_has_an_examples_extra():
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "matplotlib" in extras["examples"]
