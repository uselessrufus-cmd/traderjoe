import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from signal_engine import build_signals


@dataclass
class Trade:
    entry_ts: int
    entry: float
    exit_ts: int
    exit: float
    pnl: float


def backtest(df: pd.DataFrame):
    in_pos = False
    entry_price = 0.0
    entry_ts = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        if not in_pos and row["buy"]:
            in_pos = True
            entry_price = float(row["close"])
            entry_ts = int(row["timestamp"])
            continue

        if in_pos and row["sell"]:
            exit_price = float(row["close"])
            exit_ts = int(row["timestamp"])
            pnl = (exit_price - entry_price) / entry_price
            trades.append(Trade(entry_ts, entry_price, exit_ts, exit_price, pnl))
            in_pos = False

    return trades


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--signals", default="data/signals/btc_1h_signals.csv")
    p.add_argument("--report", default="data/signals/btc_1h_report.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = build_signals(df)
    trades = backtest(df)

    # Save signals
    Path(args.signals).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.signals, index=False)

    if not trades:
        report = "No trades generated."
        Path(args.report).write_text(report, encoding="ascii")
        print(report)
        return

    pnls = np.array([t.pnl for t in trades])
    equity = np.cumprod(1 + pnls)

    win_rate = float((pnls > 0).mean())
    avg_pnl = float(pnls.mean())
    total_return = float(equity[-1] - 1)
    dd = max_drawdown(equity)

    report = (
        f"Trades: {len(trades)}\n"
        f"Win rate: {win_rate:.2%}\n"
        f"Avg trade: {avg_pnl:.2%}\n"
        f"Total return: {total_return:.2%}\n"
        f"Max drawdown: {dd:.2%}\n"
    )

    Path(args.report).write_text(report, encoding="ascii")
    print(report)


if __name__ == "__main__":
    main()
