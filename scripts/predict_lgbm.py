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
    p.add_argument("--model", default="data/models/lgbm_model.txt")
    p.add_argument("--output", default="data/models/lgbm_latest_signal.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_features(df).dropna()

    if df.empty:
        raise SystemExit("No data after feature engineering")

    last = df.iloc[-1]

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

    X = last[feature_cols].values.reshape(1, -1)

    model = lgb.Booster(model_file=args.model)
    preds = model.predict(X)[0]
    label = int(np.argmax(preds) - 1)
    confidence = float(np.max(preds))

    signal = "hold"
    if label == 1:
        signal = "buy"
    elif label == -1:
        signal = "sell"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"timestamp={int(last['timestamp'])}\n"
        f"signal={signal}\n"
        f"confidence={confidence:.4f}\n",
        encoding="ascii",
    )

    print(out.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
