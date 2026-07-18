"""Sanity checks for the Hugging Face Kernels Hub `build.toml`.

These tests only check the packaging metadata used to publish
FlashSinkhorn on the Hugging Face Kernels Hub
(https://huggingface.co/docs/kernels/index) via `kernel-builder`. They do
not require a GPU and do not invoke `kernel-builder` itself.
"""

import pathlib

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

import flash_sinkhorn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BUILD_TOML_PATH = REPO_ROOT / "build.toml"


def _load_build_toml():
    with open(BUILD_TOML_PATH, "rb") as f:
        return tomllib.load(f)


def test_build_toml_exists():
    assert BUILD_TOML_PATH.is_file()


def test_build_toml_general_section():
    config = _load_build_toml()
    general = config["general"]

    assert general["name"] == "flash_sinkhorn"
    assert general["backends"] == ["triton"]
    assert isinstance(general["version"], int)
    assert general["license"] == "MIT"


def test_build_toml_hub_repo_id():
    config = _load_build_toml()
    assert config["general"]["hub"]["repo-id"] == "ot-triton-lab/flash-sinkhorn"


def test_build_toml_name_matches_package():
    config = _load_build_toml()
    # Per the Hugging Face kernel-builder directory convention, `general.name`
    # must be a valid Python module name matching the published package.
    assert config["general"]["name"] == flash_sinkhorn.__name__
