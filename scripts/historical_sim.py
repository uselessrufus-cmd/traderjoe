import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from ml_features import build_features, label_outcomes


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def train_model(train_df):
    feature_cols = [
        "ret_1","ret_3","ret_12","vol_12","vol_48",
        "macd_line","macd_signal","macd_hist","mfi_14","atr_14",
        "trend_up","ob_bull_recent","ob_bear_recent",
        "close_vs_sma50","close_vs_sma200",
    ]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]

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
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
    )
    return model, feature_cols


def simulate_trades(
    df,
    preds,
    min_conf=0.2,
    sl_atr=2.0,
    tp_atr=4.0,
    stagger_tp=False,
    tp1_atr=2.0,
    tp2_atr=4.0,
    tp1_pct=0.5,
    dca=False,
    dca_atr=1.0,
    martingale=1.5,
    max_dca=2,
):
    # Single-position simulation
    position = None
    trades = []
    loss_streak = 0

    for i in range(len(df)):
        row = df.iloc[i]
        probs = preds[i]
        label = int(np.argmax(probs) - 1)
        conf = float(np.max(probs))
        if conf < min_conf:
            continue

        signal = "hold"
        if label == 1:
            signal = "buy"
        elif label == -1:
            signal = "sell"

        price = float(row["close"])
        ts = int(row["timestamp"])

        if position is None:
            if signal == "buy" and not np.isnan(row["atr_14"]):
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                position = {
                    "side": "long",
                    "entry": price,
                    "ts": ts,
                    "atr": row["atr_14"],
                    "size": base_size,
                    "base_size": base_size,
                    "dca_steps": 0,
                    "tp1_hit": False,
                }
            elif signal == "sell" and not np.isnan(row["atr_14"]):
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                position = {
                    "side": "short",
                    "entry": price,
                    "ts": ts,
                    "atr": row["atr_14"],
                    "size": base_size,
                    "base_size": base_size,
                    "dca_steps": 0,
                    "tp1_hit": False,
                }
            continue

        # Stop-loss / take-profit check (ATR-based)
        if position["side"] == "long":
            stop = position["entry"] - sl_atr * position["atr"]
            take = position["entry"] + (tp2_atr if stagger_tp else tp_atr) * position["atr"]
            pnl = (price - position["entry"]) / position["entry"]
        else:
            stop = position["entry"] + sl_atr * position["atr"]
            take = position["entry"] - (tp2_atr if stagger_tp else tp_atr) * position["atr"]
            pnl = (position["entry"] - price) / position["entry"]

        # Staggered take profit (partial exit)
        if stagger_tp and not position["tp1_hit"]:
            if position["side"] == "long":
                tp1 = position["entry"] + tp1_atr * position["atr"]
                if price >= tp1:
                    trades.append(((tp1 - position["entry"]) / position["entry"]) * position["size"] * tp1_pct)
                    position["size"] *= (1.0 - tp1_pct)
                    position["tp1_hit"] = True
            else:
                tp1 = position["entry"] - tp1_atr * position["atr"]
                if price <= tp1:
                    trades.append(((position["entry"] - tp1) / position["entry"]) * position["size"] * tp1_pct)
                    position["size"] *= (1.0 - tp1_pct)
                    position["tp1_hit"] = True

        if (position["side"] == "long" and (price <= stop or price >= take)) or (
            position["side"] == "short" and (price >= stop or price <= take)
        ):
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            position = None
            continue

        # DCA / martingale
        if dca and position["dca_steps"] < max_dca:
            if position["side"] == "long":
                trigger = position["entry"] - (position["dca_steps"] + 1) * dca_atr * position["atr"]
                if price <= trigger:
                    add = position["base_size"] * (martingale ** (position["dca_steps"] + 1))
                    new_size = position["size"] + add
                    position["entry"] = (position["entry"] * position["size"] + price * add) / new_size
                    position["size"] = new_size
                    position["dca_steps"] += 1
            else:
                trigger = position["entry"] + (position["dca_steps"] + 1) * dca_atr * position["atr"]
                if price >= trigger:
                    add = position["base_size"] * (martingale ** (position["dca_steps"] + 1))
                    new_size = position["size"] + add
                    position["entry"] = (position["entry"] * position["size"] + price * add) / new_size
                    position["size"] = new_size
                    position["dca_steps"] += 1

        if position["side"] == "long" and signal == "sell":
            pnl = (price - position["entry"]) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            position = {
                "side": "short",
                "entry": price,
                "ts": ts,
                "atr": row["atr_14"],
                "size": base_size,
                "base_size": base_size,
                "dca_steps": 0,
                "tp1_hit": False,
            }
        elif position["side"] == "short" and signal == "buy":
            pnl = (position["entry"] - price) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            position = {
                "side": "long",
                "entry": price,
                "ts": ts,
                "atr": row["atr_14"],
                "size": base_size,
                "base_size": base_size,
                "dca_steps": 0,
                "tp1_hit": False,
            }

    return trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--out", default="data/models/historical_sim_report.txt")
    p.add_argument("--sl-atr", type=float, default=2.0)
    p.add_argument("--tp-atr", type=float, default=4.0)
    p.add_argument("--stagger-tp", action="store_true")
    p.add_argument("--tp1-atr", type=float, default=2.0)
    p.add_argument("--tp2-atr", type=float, default=4.0)
    p.add_argument("--tp1-pct", type=float, default=0.5)
    p.add_argument("--dca", action="store_true")
    p.add_argument("--martingale", type=float, default=1.0)
    p.add_argument("--dca-atr", type=float, default=1.0)
    p.add_argument("--max-dca", type=int, default=2)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_features(df)
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    df = label_outcomes(df, horizon=args.horizon, threshold=args.threshold)
    df = df.dropna().reset_index(drop=True)

    train = df.iloc[: args.train_bars]
    test = df.iloc[args.train_bars : args.train_bars + args.test_bars]

    model, feature_cols = train_model(train)
    preds = model.predict(test[feature_cols])

    trades = simulate_trades(
        test,
        preds,
        min_conf=0.2,
        sl_atr=args.sl_atr,
        tp_atr=args.tp_atr,
        stagger_tp=args.stagger_tp,
        tp1_atr=args.tp1_atr,
        tp2_atr=args.tp2_atr,
        tp1_pct=args.tp1_pct,
        dca=args.dca,
        dca_atr=args.dca_atr,
        martingale=args.martingale,
        max_dca=args.max_dca,
    )

    if trades:
        arr = np.array(trades)
        total = len(arr)
        win_rate = float((arr > 0).mean())
        avg = float(arr.mean())
        equity = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
    else:
        total = 0
        win_rate = avg = max_dd = 0.0

    # Feature importances
    imp = model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:6]

    summary = (
        f"Trades: {total}\n"
        f"Win rate: {win_rate:.2%}\n"
        f"Avg trade: {avg:.2%}\n"
        f"Max drawdown: {max_dd:.2%}\n\n"
        f"SL ATR: {args.sl_atr:.2f}  TP ATR: {args.tp_atr:.2f}  "
        f"Stagger TP: {args.stagger_tp}  TP1 ATR: {args.tp1_atr:.2f}  TP2 ATR: {args.tp2_atr:.2f}  TP1%: {args.tp1_pct:.2f}\n"
        f"DCA: {args.dca}  Martingale: {args.martingale:.2f}  DCA ATR: {args.dca_atr:.2f}  Max DCA: {args.max_dca}\n\n"
        "Top factors (gain):\n" + "\n".join([f"- {k}: {v:.2f}" for k, v in top])
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(summary, encoding="ascii")
    print(summary)


if __name__ == "__main__":
    main()
