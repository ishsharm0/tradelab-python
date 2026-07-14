"""Exercise a paper session without credentials or a network connection."""

from __future__ import annotations

import asyncio

from tradelab.live import SessionManager


async def main() -> None:
    manager = SessionManager()
    session = await manager.create(
        id="paper-example",
        symbol="SPY",
        mode="paper",
        equity=25_000,
    )
    await session.push_bar(
        {"time": 1_735_828_200_000, "open": 100, "high": 101, "low": 99, "close": 100}
    )
    await session.place_order(side="long", risk_pct=0.5, stop=98, target=104)
    print(session.get_status())
    await manager.halt_all()


if __name__ == "__main__":
    asyncio.run(main())
