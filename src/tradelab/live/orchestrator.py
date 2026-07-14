"""Multi-system live runtime with portfolio allocation and loss guardrails."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from tradelab.engine.execution import day_key_et
from tradelab.errors import ValidationError

from .engine import LiveEngine
from .events import EventBus
from .storage import JsonFileStorage, StorageProvider


def _system_now_ms() -> int:
    return time.time_ns() // 1_000_000


class RuntimeEngine(Protocol):
    event_bus: EventBus
    risk_manager: Any

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def get_status(self) -> dict[str, Any]: ...


EngineFactory = Callable[..., RuntimeEngine]


def _weight(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


class LiveOrchestrator:
    def __init__(
        self,
        *,
        systems: Sequence[Mapping[str, object]],
        broker: Any,
        allocation: str = "equal",
        equity: object = 10_000,
        max_daily_loss_pct: object = 0,
        feed: object | None = None,
        storage: StorageProvider | None = None,
        event_bus: EventBus | None = None,
        broker_config: Mapping[str, object] | None = None,
        confirm_live: bool = False,
        engine_factory: EngineFactory = LiveEngine,
        now_ms: Callable[[], int] = _system_now_ms,
    ) -> None:
        if not systems:
            raise ValidationError("orchestrator requires a non-empty systems array")
        if broker is None:
            raise ValidationError("orchestrator requires a broker adapter")
        if allocation not in {"equal", "weighted"}:
            raise ValidationError("allocation must be equal or weighted")
        total = _weight(equity)
        if total <= 0:
            raise ValidationError("equity must be finite and positive")
        maximum_loss = _weight(max_daily_loss_pct)
        self.systems = [dict(system) for system in systems]
        self.broker = broker
        self.allocation = allocation
        self.initial_equity = total
        self.max_daily_loss_pct = maximum_loss
        self.feed = feed
        self.storage = storage or JsonFileStorage()
        self.event_bus = event_bus or EventBus()
        self.broker_config = dict(broker_config or {})
        self.confirm_live = confirm_live
        self.engine_factory = engine_factory
        self._now_ms = now_ms
        self.engines: list[RuntimeEngine] = []
        self._event_unsubscribers: list[Callable[[], None]] = []
        self.running = False
        self.day_start_equity = total
        self.current_day: str | None = None
        self._portfolio_halted = False
        self._lifecycle_lock = asyncio.Lock()

    @staticmethod
    def _system_id(system: Mapping[str, object], index: int) -> str:
        return str(
            system.get("id") or f"{system.get('symbol')}-{system.get('interval', '1m')}-{index + 1}"
        )

    def _weights(self) -> list[float]:
        if self.allocation == "equal":
            return [1.0] * len(self.systems)
        return [_weight(system.get("weight")) for system in self.systems]

    def _allocated_equities(self, total: float) -> list[float]:
        weights = self._weights()
        divisor = sum(weights) or 1.0
        return [total * weight / divisor for weight in weights]

    def _forward(self, system_id: str, envelope: dict[str, Any]) -> None:
        event = str(envelope.get("event", ""))
        raw = envelope.get("payload")
        payload = dict(raw) if isinstance(raw, Mapping) else {}
        self.event_bus.emit_event(event, {"systemId": system_id, **payload})
        if event == "equity:update":
            self.check_portfolio_limits()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.running:
                return
            try:
                account = await self.broker.get_account()
                account_equity = _weight(account.get("equity"))
            except Exception:
                account_equity = 0
            allocations = self._allocated_equities(account_equity or self.initial_equity)
            started: list[RuntimeEngine] = []
            try:
                for index, system in enumerate(self.systems):
                    system_id = self._system_id(system, index)
                    engine_bus = EventBus()

                    def forward(envelope: dict[str, Any], system_id: str = system_id) -> None:
                        self._forward(system_id, envelope)

                    self._event_unsubscribers.append(engine_bus.on_any(forward))
                    options = {
                        key: value
                        for key, value in system.items()
                        if key not in {"id", "weight", "confirm_live"}
                    }
                    engine = self.engine_factory(
                        **options,
                        id=system_id,
                        broker=self.broker,
                        feed=self.feed,
                        storage=self.storage,
                        event_bus=engine_bus,
                        broker_config=self.broker_config,
                        confirm_live=self.confirm_live,
                        equity=allocations[index],
                        use_broker_account_equity=False,
                    )
                    await engine.start()
                    started.append(engine)
                    self.engines.append(engine)
            except BaseException:
                for engine in reversed(started):
                    await engine.stop()
                self.engines.clear()
                for unsubscribe in self._event_unsubscribers:
                    unsubscribe()
                self._event_unsubscribers.clear()
                raise
            self.running = True
            self.day_start_equity = float(self.get_status()["aggregateEquity"])
            self.current_day = day_key_et(self._now_ms())
            self._portfolio_halted = False

    def check_portfolio_limits(self) -> None:
        if self.max_daily_loss_pct <= 0:
            return
        today = day_key_et(self._now_ms())
        if self.current_day != today:
            self.current_day = today
            self.day_start_equity = float(self.get_status()["aggregateEquity"])
            self._portfolio_halted = False
            return
        if self._portfolio_halted:
            return
        equity = float(self.get_status()["aggregateEquity"])
        if equity <= self.day_start_equity * (1 - self.max_daily_loss_pct / 100):
            self._portfolio_halted = True
            for engine in self.engines:
                engine.risk_manager.halt("portfolio daily loss limit reached")
            self.event_bus.emit_event(
                "risk:halt",
                {
                    "reason": "portfolio daily loss limit reached",
                    "aggregateEquity": equity,
                },
            )

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if not self.running and not self.engines:
                return
            outcomes = await asyncio.gather(
                *(engine.stop() for engine in reversed(self.engines)),
                return_exceptions=True,
            )
            self.engines.clear()
            for unsubscribe in self._event_unsubscribers:
                unsubscribe()
            self._event_unsubscribers.clear()
            self.running = False
            failure = next(
                (outcome for outcome in outcomes if isinstance(outcome, BaseException)), None
            )
            if failure is not None:
                raise failure

    def get_status(self) -> dict[str, Any]:
        systems = [engine.get_status() for engine in self.engines]
        aggregate = sum(float(status.get("equity") or 0) for status in systems)
        return {
            "running": self.running,
            "systems": systems,
            "aggregateEquity": aggregate,
            "openPositions": sum(bool(status.get("openPosition")) for status in systems),
            "dayStartEquity": self.day_start_equity,
        }


def create_live_orchestrator(**options: Any) -> LiveOrchestrator:
    return LiveOrchestrator(**options)
