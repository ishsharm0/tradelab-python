# CLI Live Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add safe config-driven multi-system orchestration, loopback dashboards, and watch lifecycle semantics to the paper and live CLI commands.

**Architecture:** Keep parsing and lifecycle ownership in `tradelab.cli`; reuse the public `LiveOrchestrator` and dashboard factory without modifying live runtime modules. Convert each JSON system into Python engine options by resolving its registered or local strategy, and wrap every started resource in deterministic `finally` cleanup while preserving the existing live permission and certified-stream gates.

**Tech Stack:** Python 3.11+, Typer, asyncio, pytest, Ruff, mypy.

---

### Task 1: Config parsing and multi-system paper orchestration

**Files:**
- Modify: `src/tradelab/cli.py`
- Test: `tests/unit/test_cli.py`

- [x] Write a failing CLI test whose JSON config contains two systems using a registered strategy and a local Python strategy.
- [x] Run `uv run pytest tests/unit/test_cli.py -q` and confirm failure because `paper --config` is absent.
- [x] Add strict JSON-object/system-array parsing, strategy resolution, Python option normalization, and `LiveOrchestrator` construction with `PaperEngine`.
- [x] Replay each configured system's optional CSV only through its matching paper symbol/interval and report aggregate status.
- [x] Run the focused test and confirm both systems start, replay, report, and stop.

### Task 2: Dashboard and watch lifecycle

**Files:**
- Modify: `src/tradelab/cli.py`
- Test: `tests/unit/test_cli.py`

- [x] Write failing injected-factory tests asserting `--dashboard-port` is passed, the dashboard starts after its source exists, and dashboard/orchestrator or dashboard/engine close in reverse ownership order on success and failure.
- [x] Run the focused tests and confirm failure because dashboard options and lifecycle are absent.
- [x] Add `--dashboard`, `--dashboard-port`, and `--watch` to paper; add dashboard options to live.
- [x] Implement one async owner helper that starts the runtime, optionally starts the loopback dashboard, performs one-shot work or waits for cancellation, and always closes the dashboard and stops the runtime.
- [x] Re-run focused tests and confirm cleanup assertions pass.

### Task 3: Live config integration without weaker gates

**Files:**
- Modify: `src/tradelab/cli.py`
- Test: `tests/unit/test_cli.py`

- [x] Write failing help and fail-closed tests for `live --config`, dashboard flags, and config use with an uncertified bundled adapter.
- [x] Run those tests and confirm the new options are absent while the existing gate remains active.
- [x] Select single-engine versus orchestrator only after environment, explicit confirmation, and certified-stream validation; pass `confirm_live=True` and broker config to either path.
- [x] Re-run tests and confirm config never bypasses the adapter gate.

### Task 4: Documentation and release verification

**Files:**
- Modify: `README.md`

- [x] Document a Python JSON config example, dashboard/watch commands, cleanup behavior, and the unchanged fail-closed bundled-live limitation.
- [x] Run `uv run pytest tests/unit/test_cli.py -q`, then the complete `uv run pytest -q` suite.
- [x] Run `uv run ruff check src/tradelab/cli.py tests/unit/test_cli.py`, `uv run ruff format --check src/tradelab/cli.py tests/unit/test_cli.py`, and `uv run mypy --strict src examples scripts`.
- [x] Review the diff for accidental live-runtime changes, stage only CLI/tests/docs, and commit.
