"""Generate the gallery notebooks from Python scripts and Markdown lessons.

Run from the repository root: python examples/_notebooks.py [--check]
Uses only the Python standard library. Edit the sources, then regenerate.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import mimetypes
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[1]


def sources(root: Path) -> list[Path]:
    gallery = root / "examples"
    lessons = [gallery / "getting_started/first_steps.md", gallery / "choosing_parameters/guide.md"]
    return sorted(gallery.glob("*/plot_*.py")) + [path for path in lessons if path.exists()]


def split_python(text: str) -> list[tuple[str, str]]:
    """Promote section introductions to Markdown, preserving executable statements."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    cells = []
    start = 0
    if ast.get_docstring(tree):
        cells.append(("markdown", ast.get_docstring(tree)))
        start = tree.body[0].end_lineno
    boundaries = [start]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        row, column = token.start
        if token.type == tokenize.COMMENT and column == 0 and token.string.strip() == "# %%":
            if not any(node.lineno <= row <= node.end_lineno for node in tree.body):
                boundaries.append(row - 1)
    boundaries.append(len(lines))
    for first, last in zip(boundaries, boundaries[1:]):
        section = lines[first:last]
        while section and (not section[0].strip() or section[0].strip() == "# %%"):
            section.pop(0)
        introduction = []
        while section and (section[0].startswith("#") or not section[0].strip()):
            line = section.pop(0)
            introduction.append(re.sub(r"^# ?", "", line))
        if "".join(introduction).strip():
            cells.append(("markdown", "".join(introduction).strip()))
        if "".join(section).strip():
            cells.append(("code", "".join(section).strip()))
    return cells


def split_markdown(text: str) -> list[tuple[str, str]]:
    """Turn Python fences into cells and keep headings and other fences as prose."""
    cells, block = [], []
    python = False
    fence = False

    def flush():
        if "".join(block).strip():
            cells.append(("code" if python else "markdown", "".join(block).strip()))
        block.clear()

    for line in text.splitlines(keepends=True):
        if not fence and not python and line.strip() == "```python":
            flush()
            python = True
        elif python and line.strip() == "```":
            flush()
            python = False
        else:
            if not python and not fence and re.match(r"^#{1,6} ", line):
                flush()
            block.append(line)
            if not python and line.startswith("```"):
                fence = not fence
    flush()
    return cells


def bootstrap(relative: str) -> str:
    return f'''# Find this checkout when Jupyter starts at its root or in a notebook folder.
import os
import sys
from pathlib import Path

for _candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    if (_candidate / {relative!r}).is_file() and (_candidate / "torch-ext/flash_sinkhorn").is_dir():
        _notebook_root = _candidate
        break
else:
    raise RuntimeError("Start Jupyter inside the flash-sinkhorn repository checkout.")

os.chdir(_notebook_root)
# Locate shared example helpers; import the solver from the installed package.
for _path in reversed([_notebook_root / "examples", _notebook_root]):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))
__file__ = str(_notebook_root / {relative!r})

# Display the saved PNG/GIF files explicitly, including figures closed by the source.
import matplotlib
matplotlib.use("Agg")'''


def saved_images(code: str, source: Path) -> list[Path]:
    """Find the gallery's save/save_gif calls and the lessons' savefig calls."""
    images = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in ("save", "save_gif"):
            name = ast.literal_eval(node.args[2])
            suffix = ".gif" if node.func.id == "save_gif" else ".png"
            images.append(source.parent / "images" / (name + suffix))
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "savefig":
            path = node.args[0]
            if not isinstance(path, ast.BinOp) or not isinstance(path.op, ast.Div):
                raise ValueError("Expected savefig(image_dir / 'name.png')")
            images.append(source.parent / "images" / ast.literal_eval(path.right))
    return images


def make_notebook(source: Path, root: Path = ROOT) -> dict:
    text = source.read_text()
    relative = source.relative_to(root).as_posix()
    sections = split_python(text) if source.suffix == ".py" else split_markdown(text)
    notebook_sources = {path.resolve() for path in sources(root)}
    cells = []

    def add(kind, content, tag=None, attachments=None):
        metadata = {"tags": [tag]} if tag else {}
        cell = {"cell_type": kind, "metadata": metadata, "source": content.strip().splitlines(keepends=True)}
        cell["id"] = hashlib.sha256(f"{relative}:{len(cells)}:{kind}".encode()).hexdigest()[:12]
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        if attachments:
            cell["attachments"] = attachments
        cells.append(cell)

    def markdown(content):
        attachments = {}
        heading = re.match(r"^#{1,6} (.+)(?:\n|$)", content)
        if heading:
            # Keep Markdown chapter links valid in notebook renderers with different heading IDs.
            anchor = re.sub(r"[^\w -]", "", heading.group(1)).lower().replace(" ", "-")
            content = f'<a id="{anchor}"></a>\n\n' + content

        def link(match):
            label, url = match.group(1), match.group(2)
            target, separator, anchor = url.partition("#")
            path = (source.parent / target).resolve()
            if match.group(0).startswith("!") and path.is_file():
                mime = mimetypes.guess_type(path.name)[0]
                attachments[path.name] = {mime: base64.b64encode(path.read_bytes()).decode("ascii")}
                return f"![{label}](attachment:{path.name})"
            if path in notebook_sources:
                url = str(Path(target).with_suffix(".ipynb")) + (separator + anchor if separator else "")
            prefix = "!" if match.group(0).startswith("!") else ""
            return f"{prefix}[{label}]({url})"

        content = re.sub(r"!?\[([^\]]*)\]\(([^)]+)\)", link, content)
        if attachments:
            content = "**Reference figure from previous script runs.**\n\n" + content
        add("markdown", content, attachments=attachments)

    kind, intro = sections[0]
    markdown(intro)
    runnable = any(kind == "code" for kind, _ in sections)
    instructions = (
        f"Source: [{source.name}]({source.name}) · [Gallery](../README.md)\n\n"
        "Reference figures and quoted results come from previous script runs. "
        "They are reading aids, not outputs from an execution of this notebook."
    )
    if runnable:
        instructions += (
            "\n\nInstall `pip install --upgrade 'flash-sinkhorn[notebooks]'`, clone this repository for the "
            "example files, and start `jupyter lab` from the checkout. "
            "Select its Python environment, then run the cells in order on an NVIDIA CUDA GPU. "
            "The setup cell finds the checkout from the repository root or this notebook's folder. "
            "The solver comes from the installed package. "
            "Running all cells prints fresh results and displays new figures; it also replaces the "
            "source's saved images. The scale examples retain the script's full problem sizes."
        )
    else:
        instructions += "\n\nThis is a reading guide. Follow its example links for runnable notebooks."
    add("markdown", instructions)
    if runnable:
        add("code", bootstrap(relative), "setup")
    for kind, content in sections[1:]:
        if kind == "markdown":
            markdown(content)
            continue
        add("code", content, "source-code")
        images = saved_images(content, source)
        if images:
            display = ["from IPython.display import Image as _NotebookImage, display as _notebook_display"]
            for path in images:
                display.append(f"_notebook_display(_NotebookImage(filename=str(_notebook_root / {path.relative_to(root).as_posix()!r})))")
            display.extend([
                "# Release figure managers so repeated runs do not accumulate open figures.",
                "import matplotlib.pyplot as _notebook_plt",
                '_notebook_plt.close("all")',
            ])
            add("code", "\n".join(display), "display-figures")
            if source.suffix == ".py":
                for path in images:
                    markdown(f"![{path.stem.replace('_', ' ')}](images/{path.name})")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "gallery": {"source": relative, "source_sha256": hashlib.sha256(text.encode()).hexdigest()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate(root: Path = ROOT, check: bool = False) -> list[Path]:
    """Write notebooks, or return the stale paths without changing them."""
    changed = []
    for source in sources(root):
        path = source.with_suffix(".ipynb")
        contents = json.dumps(make_notebook(source, root), indent=1, ensure_ascii=False) + "\n"
        if not path.exists() or path.read_text() != contents:
            changed.append(path)
            if not check:
                path.write_text(contents)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if notebooks need regeneration")
    args = parser.parse_args()
    changed = generate(check=args.check)
    if args.check and changed:
        for path in changed:
            print(f"Regenerate {path.relative_to(ROOT)}")
        raise SystemExit(1)
    print(f"{len(sources(ROOT))} notebooks are up to date ({len(changed)} regenerated).")
