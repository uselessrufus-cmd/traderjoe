import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features


def load_feature_list(meta_path: Path) -> list[str]:
    if not meta_path.exists():
        return []
    lines = meta_path.read_text(encoding="ascii", errors="ignore").splitlines()
    feats = [l for l in lines if l and not l.startswith("validation_accuracy=")]
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--h4", default="data/bitstamp/ohlc/bitstamp_4h.csv")
    p.add_argument("--d1", default="data/bitstamp/ohlc/bitstamp_1d.csv")
    p.add_argument("--model", default="data/models/lgbm_mtf_model.txt")
    p.add_argument("--meta", default="data/models/lgbm_mtf_meta.txt")
    p.add_argument("--output", default="data/models/lgbm_mtf_latest_signal.txt")
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

    # Build per-timeframe features
    h1f = build_features(h1).add_prefix("h1_")
    h4f = build_features(h4).add_prefix("h4_")
    d1f = build_features(d1).add_prefix("d1_")

    # Align on 1h
    base = h1f
    for other, prefix in [(h4f, "h4"), (d1f, "d1")]:
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

    base = base.dropna().reset_index(drop=True)
    if base.empty:
        raise SystemExit("No data after feature alignment")

    last = base.iloc[-1]

    feature_cols = load_feature_list(Path(args.meta))
    if not feature_cols:
        # Fallback to all numeric features
        feature_cols = [c for c in base.columns if c not in {"label"}]
        feature_cols = [c for c in feature_cols if not c.endswith("timestamp")]
    else:
        # Remove any columns that may not exist in the latest frame
        feature_cols = [c for c in feature_cols if c in base.columns]

    X = last[feature_cols].values.reshape(1, -1)

    model = lgb.Booster(model_file=args.model)
    preds = model.predict(X, predict_disable_shape_check=True)[0]
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
        f"timestamp={int(last['h1_timestamp'])}\n"
        f"signal={signal}\n"
        f"confidence={confidence:.4f}\n",
        encoding="ascii",
    )

    print(out.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
