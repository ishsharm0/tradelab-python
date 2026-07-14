"""Run an offline CSV backtest and write self-contained artifacts."""

from __future__ import annotations

from pathlib import Path

from tradelab import backtest, load_candles_from_csv
from tradelab.reporting import export_backtest_artifacts
from tradelab.strategies import get_strategy


def main(csv_path: Path, output: Path = Path("output/csv-backtest")) -> None:
    candles = load_candles_from_csv(csv_path)
    signal = get_strategy("ema-cross")({"fast": 10, "slow": 30, "rr": 2})
    result = backtest(
        candles=candles,
        symbol=csv_path.stem.upper(),
        interval="1d",
        warmup_bars=30,
        signal=signal,
    )
    artifacts = export_backtest_artifacts(result, out_dir=output)
    print(result["metrics"])
    print(artifacts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/csv-backtest"))
    arguments = parser.parse_args()
    main(arguments.csv_path, arguments.output)
