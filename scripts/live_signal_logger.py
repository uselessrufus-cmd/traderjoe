import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from signal_engine import build_signals


def append_signal(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "signal",
                    "close",
                    "macd_line",
                    "macd_signal",
                    "macd_hist",
                    "mfi_14",
                ],
            )
            w.writeheader()
            w.writerow(row)
        return

    rows = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if rows and int(rows[-1]["timestamp"]) == int(row["timestamp"]):
        return

    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "signal",
                "close",
                "macd_line",
                "macd_signal",
                "macd_hist",
                "mfi_14",
            ],
        )
        w.writerow(row)


def run_once(input_csv: Path, out_csv: Path):
    df = pd.read_csv(input_csv)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = build_signals(df)
    last = df.iloc[-1]

    signal = "hold"
    if bool(last.get("buy")):
        signal = "buy"
    elif bool(last.get("sell")):
        signal = "sell"

    row = {
        "timestamp": int(last["timestamp"]),
        "signal": signal,
        "close": float(last["close"]),
        "macd_line": float(last["macd_line"]),
        "macd_signal": float(last["macd_signal"]),
        "macd_hist": float(last["macd_hist"]),
        "mfi_14": float(last["mfi_14"]),
    }
    append_signal(out_csv, row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--output", default="data/signals/live_signals.csv")
    p.add_argument("--every", type=int, default=3600)
    args = p.parse_args()

    input_csv = Path(args.input)
    out_csv = Path(args.output)

    while True:
        run_once(input_csv, out_csv)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
