import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features


def label_soft(df: pd.DataFrame, horizon: int = 12):
    future = df["close"].shift(-horizon)
    future_ret = (future - df["close"]) / df["close"]
    # Soft label: use return as target for regression
    df["label_reg"] = future_ret
    return df


def profit_objective(preds: np.ndarray, train_data: lgb.Dataset):
    # Custom objective to push returns up; gradient of -pred*y
    y = train_data.get_label()
    grad = -y
    hess = np.ones_like(y)
    return grad, hess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--out", default="data/models/profit_model.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = build_features(df)
    df = label_soft(df, horizon=12)
    df = df.dropna().reset_index(drop=True)

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

    X = df[feature_cols]
    y = df["label_reg"]

    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "metric": "l2",
        "verbosity": -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
