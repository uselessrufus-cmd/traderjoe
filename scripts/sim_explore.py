import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features, label_outcomes
from progress_bar import render_progress


def run_windows(df, train_bars, test_bars):
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

    def train_eval(train, test):
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
        return acc

    accs = []
    total_windows = 0
    if len(df) >= train_bars + test_bars:
        total_windows = ((len(df) - train_bars - test_bars) // test_bars) + 1
    i = 0
    widx = 0
    while i + train_bars + test_bars <= len(df):
        train = df.iloc[i : i + train_bars]
        test = df.iloc[i + train_bars : i + train_bars + test_bars]
        accs.append(train_eval(train, test))
        i += test_bars
        widx += 1
        if total_windows > 0 and (widx % 5 == 0 or widx == total_windows):
            print(f"  windows {render_progress(widx, total_windows)}")

    if not accs:
        return None

    arr = np.array(accs)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "windows": len(arr),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--out", default="data/models/sim_explore.csv")
    p.add_argument("--summary", default="data/models/sim_explore_summary.txt")
    p.add_argument("--sleep", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Parameter grid (keep tight for speed)
    horizons = [6, 12]
    thresholds = [0.005, 0.01]
    train_windows = [24 * 365]
    test_windows = [24 * 90]

    rows = []
    combos = list(itertools.product(horizons, thresholds, train_windows, test_windows))
    total = len(combos)
    for idx, (horizon, threshold, train_bars, test_bars) in enumerate(combos, start=1):
        tmp = build_features(df.copy())
        tmp = label_outcomes(tmp, horizon=horizon, threshold=threshold)
        tmp = tmp.dropna().reset_index(drop=True)
        stats = run_windows(tmp, train_bars, test_bars)
        if stats is None:
            print(f"Progress {render_progress(idx, total)}")
            continue
        stability = stats["mean"] - stats["std"]
        rows.append({
            "horizon": horizon,
            "threshold": threshold,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "windows": stats["windows"],
            "stability": stability,
        })

        # Throttle to reduce CPU usage
        import time
        time.sleep(args.sleep)
        print(f"Progress {render_progress(idx, total)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("stability", ascending=False)
    df_out.to_csv(out, index=False)

    if df_out.empty:
        Path(args.summary).write_text("No results", encoding="ascii")
        print("No results")
        return

    best = df_out.iloc[0]
    worst = df_out.iloc[-1]

    summary = (
        "Best (stability)\n"
        f"horizon={int(best['horizon'])} threshold={float(best['threshold'])} "
        f"train={int(best['train_bars'])} test={int(best['test_bars'])}\n"
        f"mean={float(best['mean']):.4f} std={float(best['std']):.4f} stability={float(best['stability']):.4f}\n\n"
        "Worst (stability)\n"
        f"horizon={int(worst['horizon'])} threshold={float(worst['threshold'])} "
        f"train={int(worst['train_bars'])} test={int(worst['test_bars'])}\n"
        f"mean={float(worst['mean']):.4f} std={float(worst['std']):.4f} stability={float(worst['stability']):.4f}\n"
    )

    Path(args.summary).write_text(summary, encoding="ascii")
    print(summary)


if __name__ == "__main__":
    main()
