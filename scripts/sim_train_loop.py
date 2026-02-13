import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features, label_outcomes
from progress_bar import render_progress


def train_eval(df, train_end, test_end, feature_cols):
    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end]

    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "metric": "multi_logloss",
        "verbosity": -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train + 1)
    lgb_val = lgb.Dataset(X_test, label=y_test + 1)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1) - 1
    acc = float((pred_labels == y_test.values).mean())

    return acc, len(test)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--out", default="data/models/sim_report.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = build_features(df)
    df = label_outcomes(df, horizon=args.horizon, threshold=args.threshold)
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

    reports = []
    i = 0
    windows = 0
    max_possible = 0
    if len(df) >= args.train_bars + args.test_bars:
        max_possible = ((len(df) - args.train_bars - args.test_bars) // args.test_bars) + 1
    total_windows = max_possible
    if args.max_windows and total_windows > 0:
        total_windows = min(total_windows, args.max_windows)
    while i + args.train_bars + args.test_bars <= len(df):
        train_end = i + args.train_bars
        test_end = train_end + args.test_bars
        acc, n = train_eval(df, train_end, test_end, feature_cols)
        reports.append(f"window {i}-{test_end}: acc={acc:.2%} n={n}")
        i += args.test_bars
        windows += 1
        if total_windows > 0:
            print(f"Progress {render_progress(windows, total_windows)}")
        # Throttle to reduce CPU usage
        import time
        time.sleep(args.sleep)
        if args.max_windows and windows >= args.max_windows:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(reports), encoding="ascii")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
