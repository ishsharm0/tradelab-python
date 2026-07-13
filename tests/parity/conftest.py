"""Shared paths and JSON loaders for generated JavaScript parity fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Return the directory containing generated parity fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def load_fixture(fixture_dir: Path) -> Callable[[str], object]:
    """Load one generated fixture by its manifest filename."""

    def load(filename: str) -> object:
        return json.loads((fixture_dir / filename).read_text(encoding="utf-8"))

    return load
