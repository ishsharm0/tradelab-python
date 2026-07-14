"""Smoke an installed wheel in an isolated interpreter.

Set ``TRADELAB_RELEASE_PYTHON`` to the interpreter in the clean release virtual
environment. The normal source test suite skips this test intentionally.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    "TRADELAB_RELEASE_PYTHON" not in os.environ,
    reason="requires a clean wheel environment",
)
def test_installed_wheel_imports_and_console_scripts(tmp_path: Path) -> None:
    python = os.environ["TRADELAB_RELEASE_PYTHON"]
    probe = """
import importlib, json, tradelab
names = ['brokers', 'data', 'engine', 'live', 'mcp', 'metrics', 'reporting',
         'research', 'strategies', 'ta', 'utils']
for name in names:
    importlib.import_module(f'tradelab.{name}')
print(json.dumps({'version': tradelab.__version__, 'namespaces': names}))
"""
    result = subprocess.run(
        [python, "-I", "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["version"] == "1.3.1"
    assert len(payload["namespaces"]) == 11

    bin_dir = Path(python).parent
    cli = subprocess.run(
        [str(bin_dir / "tradelab"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "backtest" in cli.stdout

    mcp = subprocess.run(
        [str(bin_dir / "tradelab-mcp"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "Model Context Protocol" in mcp.stdout
