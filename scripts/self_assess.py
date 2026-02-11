import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def compute_outcomes(df: pd.DataFrame, horizon: int, threshold: float):
    future = df["close"].shift(-horizon)
    future_ret = (future - df["close"]) / df["close"]
    label = 0
    if future_ret > threshold:
        label = 1
    elif future_ret < -threshold:
        label = -1
    return label, future_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals", default="data/models/lgbm_mtf_latest_signal.txt")
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--db", default="data/ai_memory.db")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01)
    args = p.parse_args()

    # Read latest signal
    sig_path = Path(args.signals)
    if not sig_path.exists():
        print("No signal file found.")
        return

    lines = sig_path.read_text(encoding="ascii", errors="ignore").splitlines()
    sig = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            sig[k.strip()] = v.strip()

    ts = int(sig.get("timestamp", 0))
    signal = sig.get("signal", "hold")
    confidence = float(sig.get("confidence", 0))

    if ts == 0:
        print("Invalid signal timestamp")
        return

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "close"]]

    # Locate timestamp
    row = df[df["timestamp"] == ts]
    if row.empty:
        print("Timestamp not found in price data")
        return

    idx = row.index[0]
    if idx + args.horizon >= len(df):
        print("Not enough future data to label yet")
        return

    future = df.iloc[idx + args.horizon]["close"]
    current = df.iloc[idx]["close"]
    ret = (future - current) / current
    outcome = 0
    if ret > args.threshold:
        outcome = 1
    elif ret < -args.threshold:
        outcome = -1

    # Store in DB
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_history (
                timestamp INTEGER PRIMARY KEY,
                signal TEXT,
                confidence REAL,
                outcome INTEGER,
                ret REAL
            )
            """
        )
        con.execute(
            "INSERT OR REPLACE INTO signal_history (timestamp, signal, confidence, outcome, ret) VALUES (?, ?, ?, ?, ?)",
            (ts, signal, confidence, outcome, ret),
        )
        con.commit()

    print(f"Logged outcome for {ts}: signal={signal} outcome={outcome} ret={ret:.4f}")


if __name__ == "__main__":
    main()
