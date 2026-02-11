import argparse
import gzip
import io
from pathlib import Path

import pandas as pd
import requests


HIST_URL = (
    "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/"
    "refs/heads/main/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
)
UPDATES_URL = (
    "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/"
    "refs/heads/main/data/updates/btcusd_bitstamp_1min_latest.csv"
)


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_hist_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        data = f.read()
    df = pd.read_csv(io.BytesIO(data))
    return df


def load_updates_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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
    p.add_argument("--out-dir", default="data/bitstamp")
    p.add_argument("--intervals", default="1m,5m,15m,1h,4h,12h,1d,1w,1mo")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    ohlc_dir = out_dir / "ohlc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ohlc_dir.mkdir(parents=True, exist_ok=True)

    hist_gz = raw_dir / "btcusd_bitstamp_1min_2012-2025.csv.gz"
    updates_csv = raw_dir / "btcusd_bitstamp_1min_latest.csv"

    print("Downloading historical + updates...")
    download(HIST_URL, hist_gz)
    download(UPDATES_URL, updates_csv)

    print("Loading data...")
    df_hist = load_hist_csv_gz(hist_gz)
    df_updates = load_updates_csv(updates_csv)

    df = pd.concat([df_hist, df_updates], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Save 1m base
    base_path = ohlc_dir / "bitstamp_1m.csv"
    df.to_csv(base_path, index=False)
    print("Wrote", base_path)

    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    rule_map = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
        "1mo": "MS",
    }

    for interval in intervals:
        if interval == "1m":
            continue
        rule = rule_map.get(interval)
        if not rule:
            continue
        out = resample_ohlc(df, rule)
        out_path = ohlc_dir / f"bitstamp_{interval}.csv"
        out.to_csv(out_path, index=False)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
