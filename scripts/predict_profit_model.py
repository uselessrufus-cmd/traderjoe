import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--model", default="data/models/profit_model.txt")
    p.add_argument("--out", default="data/models/profit_signal.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_features(df).dropna().reset_index(drop=True)

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_12",
        "vol_12",
        "vol_48",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "mfi_14",
        "atr_14",
        "trend_up",
        "ob_bull_recent",
        "ob_bear_recent",
        "close_vs_sma50",
        "close_vs_sma200",
    ]

    last = df.iloc[-1]
    X = last[feature_cols].values.reshape(1, -1)

    model = lgb.Booster(model_file=args.model)
    pred = model.predict(X)[0]

    signal = "hold"
    if pred > 0.002:
        signal = "buy"
    elif pred < -0.002:
        signal = "sell"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"timestamp={int(last['timestamp'])}\n"
        f"pred_return={pred:.6f}\n"
        f"signal={signal}\n",
        encoding="ascii",
    )

    print(out.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
