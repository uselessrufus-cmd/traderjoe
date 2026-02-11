import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features, label_outcomes


def add_timeframe_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    feats = build_features(df)
    # Keep timestamp unprefixed to avoid duplicates, then prefix others
    feats = feats.rename(columns={"timestamp": f"{prefix}_timestamp"})
    other_cols = [c for c in feats.columns if c != f"{prefix}_timestamp"]
    feats = feats.rename(columns={c: f"{prefix}_{c}" for c in other_cols})
    return feats


def align_on_timestamp(base: pd.DataFrame, other: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # Forward-fill other timeframe features to base timestamps
    # Ensure no duplicate timestamp column
    other = other.loc[:, ~other.columns.duplicated()]
    other = other[
        [f"{prefix}_timestamp"]
        + [c for c in other.columns if c.startswith(prefix + "_") and c != f"{prefix}_timestamp"]
    ]
    other = other.sort_values(f"{prefix}_timestamp")
    base = base.sort_values("h1_timestamp")
    base = pd.merge_asof(
        base,
        other,
        left_on="h1_timestamp",
        right_on=f"{prefix}_timestamp",
        direction="backward",
    )
    base = base.drop(columns=[f"{prefix}_timestamp"])
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--h4", default="data/bitstamp/ohlc/bitstamp_4h.csv")
    p.add_argument("--d1", default="data/bitstamp/ohlc/bitstamp_1d.csv")
    p.add_argument("--db", default="data/ai_memory.db")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01)
    args = p.parse_args()

    h1 = pd.read_csv(args.h1)
    h4 = pd.read_csv(args.h4)
    d1 = pd.read_csv(args.d1)

    for df in (h1, h4, d1):
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)

    h1 = h1[["timestamp", "open", "high", "low", "close", "volume"]]
    h4 = h4[["timestamp", "open", "high", "low", "close", "volume"]]
    d1 = d1[["timestamp", "open", "high", "low", "close", "volume"]]

    h1f = add_timeframe_features(h1, "h1")
    h4f = add_timeframe_features(h4, "h4")
    d1f = add_timeframe_features(d1, "d1")

    # Base on 1h timestamps
    base = h1f
    base = align_on_timestamp(base, h4f, "h4")
    base = align_on_timestamp(base, d1f, "d1")

    # Add labels from 1h close
    base = base.rename(columns={"h1_timestamp": "timestamp", "h1_close": "close"})
    base = label_outcomes(base, horizon=args.horizon, threshold=args.threshold)

    base = base.dropna().reset_index(drop=True)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as con:
        base.to_sql("features", con, if_exists="replace", index=False)

    print(f"Wrote multi-timeframe features+labels to {args.db}")


if __name__ == "__main__":
    main()
