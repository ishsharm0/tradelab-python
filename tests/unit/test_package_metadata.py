"""Tests for runtime distribution metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_runtime_dependencies_declare_tzdata() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    assert any(dependency.startswith("tzdata>=") for dependency in project["dependencies"])
