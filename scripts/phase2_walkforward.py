import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from signal_engine import build_signals
from progress_bar import render_progress


@dataclass
class EvalResult:
    trades: int
    win_rate: float
    avg_pnl: float
    total_return: float
    max_drawdown: float


def backtest(df: pd.DataFrame) -> EvalResult:
    in_pos = False
    entry_price = 0.0
    pnls = []

    for i in range(len(df)):
        row = df.iloc[i]
        if not in_pos and row["buy"]:
            in_pos = True
            entry_price = float(row["close"])
            continue

        if in_pos and row["sell"]:
            exit_price = float(row["close"])
            pnl = (exit_price - entry_price) / entry_price
            pnls.append(pnl)
            in_pos = False

    if not pnls:
        return EvalResult(0, 0.0, 0.0, 0.0, 0.0)

    pnls = np.array(pnls)
    equity = np.cumprod(1 + pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    return EvalResult(
        trades=len(pnls),
        win_rate=float((pnls > 0).mean()),
        avg_pnl=float(pnls.mean()),
        total_return=float(equity[-1] - 1),
        max_drawdown=float(dd.min()),
    )


def score(res: EvalResult) -> float:
    if res.trades == 0:
        return -1e9
    dd = abs(res.max_drawdown) if res.max_drawdown != 0 else 1e-6
    return res.total_return / dd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--report", default="data/signals/btc_1h_walkforward.txt")
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--max-windows", type=int, default=12)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    param_grid = []
    for mfi_lower in [40]:
        for mfi_upper in [70, 80]:
            for ob_lookback in [6, 12]:
                for trend_filter in [True]:
                    param_grid.append((mfi_lower, mfi_upper, ob_lookback, trend_filter))

    reports = []
    i = 0
    windows = 0
    total_windows = 0
    if len(df) >= args.train_bars + args.test_bars:
        total_windows = ((len(df) - args.train_bars - args.test_bars) // args.test_bars) + 1
    if args.max_windows and total_windows > 0:
        total_windows = min(total_windows, args.max_windows)
    while i + args.train_bars + args.test_bars <= len(df):
        train = df.iloc[i : i + args.train_bars]
        test = df.iloc[i + args.train_bars : i + args.train_bars + args.test_bars]

        best = None
        best_score = -1e9
        best_res = None

        for mfi_lower, mfi_upper, ob_lookback, trend_filter in param_grid:
            sig = build_signals(
                train,
                mfi_lower=mfi_lower,
                mfi_upper=mfi_upper,
                ob_lookback=ob_lookback,
                trend_filter=trend_filter,
            )
            res = backtest(sig)
            s = score(res)
            if s > best_score:
                best_score = s
                best = (mfi_lower, mfi_upper, ob_lookback, trend_filter)
                best_res = res

        test_sig = build_signals(
            test,
            mfi_lower=best[0],
            mfi_upper=best[1],
            ob_lookback=best[2],
            trend_filter=best[3],
        )
        test_res = backtest(test_sig)

        reports.append(
            {
                "start_ts": int(train.iloc[0]["timestamp"]),
                "end_ts": int(test.iloc[-1]["timestamp"]),
                "params": best,
                "train_return": best_res.total_return,
                "train_dd": best_res.max_drawdown,
                "test_return": test_res.total_return,
                "test_dd": test_res.max_drawdown,
                "test_trades": test_res.trades,
                "test_win_rate": test_res.win_rate,
            }
        )

        i += args.test_bars
        windows += 1
        if total_windows > 0:
            print(f"Progress {render_progress(windows, total_windows)}")
        if windows >= args.max_windows:
            break

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in reports:
        lines.append(
            "Window "
            f"{r['start_ts']}..{r['end_ts']} | params={r['params']} | "
            f"train_ret={r['train_return']:.2%} train_dd={r['train_dd']:.2%} | "
            f"test_ret={r['test_return']:.2%} test_dd={r['test_dd']:.2%} "
            f"trades={r['test_trades']} win={r['test_win_rate']:.2%}"
        )

    out.write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
