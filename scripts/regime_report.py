import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--out", default="data/models/regime_report.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "close"]]
    ret = df["close"].pct_change()

    vol = ret.rolling(48).std()
    trend = df["close"].rolling(50).mean() - df["close"].rolling(200).mean()

    df["regime"] = "range"
    df.loc[trend > 0, "regime"] = "bull"
    df.loc[trend < 0, "regime"] = "bear"

    df["vol_regime"] = "low"
    df.loc[vol > vol.quantile(0.7), "vol_regime"] = "high"

    summary = df.groupby(["regime", "vol_regime"]).size().reset_index(name="count")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary.to_string(index=False), encoding="ascii")
    print(out.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
