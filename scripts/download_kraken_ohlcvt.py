import argparse
import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd

try:
    import gdown
except Exception:  # pragma: no cover
    gdown = None

KRAKEN_OHLCVT_FILE_ID = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"

INTERVALS_MIN = {"1", "5", "15", "30", "60", "240", "720", "1440"}


def download_zip(out_path: Path):
    if out_path.exists():
        return
    if gdown is None:
        raise SystemExit("gdown is required to download the Kraken OHLCVT zip")
    url = f"https://drive.google.com/uc?id={KRAKEN_OHLCVT_FILE_ID}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(out_path), quiet=False)


def detect_files(zf: zipfile.ZipFile):
    files = []
    for n in zf.namelist():
        nl = n.lower()
        if not nl.endswith(".csv"):
            continue
        if "/__macosx/" in nl or "/._" in nl or nl.startswith("._"):
            continue
        files.append(n)
    pairs = {}

    # Try multiple filename patterns
    patterns = [
        re.compile(r"(?P<pair>[A-Z0-9]+)[_-](?P<intv>\d+)(?:m)?\.csv$"),
        re.compile(r".*/(?P<pair>[A-Z0-9]+)[_-](?P<intv>\d+)(?:m)?\.csv$"),
    ]

    for name in files:
        for pat in patterns:
            m = pat.search(name)
            if not m:
                continue
            pair = m.group("pair")
            intv = m.group("intv")
            if intv not in INTERVALS_MIN:
                continue
            pairs.setdefault(pair, {})[intv] = name
            break

    return pairs


def read_ohlcvt(fileobj: io.BufferedReader) -> pd.DataFrame:
    # Kraken OHLCVT columns typically: time,open,high,low,close,vwap,volume,count
    # Handle with or without headers; be robust to non-utf8 bytes.
    data = fileobj.read()
    head = data[:2048]
    has_header = b"time" in head.lower()

    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            bio = io.BytesIO(data)
            if has_header:
                df = pd.read_csv(bio, encoding=enc)
            else:
                df = pd.read_csv(bio, header=None, encoding=enc)
            break
        except UnicodeDecodeError:
            df = None
            continue
    if df is None:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode CSV")

    if df.shape[1] < 6:
        raise ValueError("Unexpected OHLCVT column count")

    # Map columns by index
    df = df.rename(
        columns={
            df.columns[0]: "timestamp",
            df.columns[1]: "open",
            df.columns[2]: "high",
            df.columns[3]: "low",
            df.columns[4]: "close",
            df.columns[6] if df.shape[1] > 6 else df.columns[5]: "volume",
        }
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/kraken/raw/kraken_ohlcvt.zip")
    p.add_argument("--pair", default="XBTUSD")
    p.add_argument("--intervals", default="1,5,15,60,240,720,1440")
    p.add_argument("--out-dir", default="data/kraken/ohlc")
    p.add_argument("--list-pairs", action="store_true")
    args = p.parse_args()

    zip_path = Path(args.zip)
    download_zip(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        pairs = detect_files(zf)

        if args.list_pairs:
            print("Available pairs (detected):")
            for p in sorted(pairs.keys()):
                print(p)
            return

        if args.pair not in pairs:
            raise SystemExit(
                f"Pair {args.pair} not found. Use --list-pairs to see available pairs."
            )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
        for intv in intervals:
            name = pairs[args.pair].get(intv)
            if not name:
                print(f"Skipping interval {intv} (not found)")
                continue

            with zf.open(name) as f:
                df = read_ohlcvt(f)

            out_path = out_dir / f"kraken_{intv}m.csv"
            df.to_csv(out_path, index=False)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
