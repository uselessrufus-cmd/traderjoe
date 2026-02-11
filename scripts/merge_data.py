import argparse
from pathlib import Path

import pandas as pd


def read_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def merge(history: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([history, live], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ts = df["timestamp"].astype("int64")
    dt = pd.to_datetime(ts, unit="s", utc=True)
    df = df.copy()
    df["_dt"] = dt
    df = df.set_index("_dt")

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    out = df.resample(rule).agg(ohlc).dropna()
    out["timestamp"] = out.index.astype("int64") // 10**9
    out = out.reset_index(drop=True)
    return out[["timestamp", "open", "high", "low", "close", "volume"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", default="data/bitstamp/ohlc")
    p.add_argument("--live-dir", default="data/bitunix/live")
    p.add_argument("--out-dir", default="data/combined")
    p.add_argument("--intervals", default="1m,5m,15m,1h,4h,12h,1d,1w,1M")
    p.add_argument("--keep-live-days", type=int, default=7)
    args = p.parse_args()

    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    history_dir = Path(args.history_dir)
    live_dir = Path(args.live_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for interval in intervals:
        hist_path = history_dir / f"bitstamp_{interval}.csv"
        if not hist_path.exists():
            print("Missing", hist_path)
            continue
        hist_df = read_ohlc(hist_path)

        live_path = live_dir / f"bitunix_{interval}.csv"
        if live_path.exists():
            live_df = read_ohlc(live_path)
            merged = merge(hist_df, live_df)
        else:
            merged = hist_df

        out_path = out_dir / f"btc_{interval}.csv"
        merged.to_csv(out_path, index=False)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
