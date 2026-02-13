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


def build_signal_rationale(row, signal: str, conf: float, min_conf: float) -> str:
    parts = [f"conf={conf:.3f}"]
    if conf < min_conf:
        parts.append("below_min_conf")
    macd_hist = float(row.get("macd_hist", 0.0)) if pd.notna(row.get("macd_hist", np.nan)) else 0.0
    mfi = float(row.get("mfi_14", 50.0)) if pd.notna(row.get("mfi_14", np.nan)) else 50.0
    trend_up = int(row.get("trend_up", 0)) if pd.notna(row.get("trend_up", np.nan)) else 0
    ob_bull = int(row.get("ob_bull_recent", 0)) if pd.notna(row.get("ob_bull_recent", np.nan)) else 0
    ob_bear = int(row.get("ob_bear_recent", 0)) if pd.notna(row.get("ob_bear_recent", np.nan)) else 0
    parts.append(f"macd_hist={macd_hist:.4f}")
    parts.append(f"mfi_14={mfi:.2f}")
    parts.append(f"trend_up={trend_up}")
    parts.append(f"ob_bull={ob_bull}")
    parts.append(f"ob_bear={ob_bear}")
    parts.append(f"signal={signal}")
    return " | ".join(parts)


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
    max_leverage=1.0,
    progress_step=0,
):
    # Single-position simulation
    position = None
    trades = []
    trade_rows = []
    bar_rows = []
    loss_streak = 0

    total = len(df)
    progress_mod = int(progress_step) if progress_step and progress_step > 0 else max(1, total // 40)
    for i in range(total):
        row = df.iloc[i]
        probs = preds[i]
        label = int(np.argmax(probs) - 1)
        conf = float(np.max(probs))

        signal = "hold"
        if conf >= min_conf and label == 1:
            signal = "buy"
        elif conf >= min_conf and label == -1:
            signal = "sell"

        price = float(row["close"])
        ts = int(row["timestamp"])
        dt = row.get("_dt")
        action = "none"
        reward = 0.0
        rationale = build_signal_rationale(row, signal, conf, min_conf)

        if position is None:
            if signal == "buy" and not np.isnan(row["atr_14"]):
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                base_size = min(base_size, max_leverage)
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
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "dt": dt,
                        "side": "long",
                        "action": "open",
                        "entry_ts": ts,
                        "entry_price": price,
                        "exit_price": price,
                        "size": base_size,
                        "pnl": 0.0,
                        "rationale": rationale,
                    }
                )
                action = "open_long"
            elif signal == "sell" and not np.isnan(row["atr_14"]):
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                base_size = min(base_size, max_leverage)
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
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "dt": dt,
                        "side": "short",
                        "action": "open",
                        "entry_ts": ts,
                        "entry_price": price,
                        "exit_price": price,
                        "size": base_size,
                        "pnl": 0.0,
                        "rationale": rationale,
                    }
                )
                action = "open_short"
            unrealized = 0.0
            if position is not None:
                if position["side"] == "long":
                    unrealized = ((price - position["entry"]) / position["entry"]) * position["size"]
                else:
                    unrealized = ((position["entry"] - price) / position["entry"]) * position["size"]
            bar_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "close": price,
                    "signal": signal,
                    "confidence": conf,
                    "action": action,
                    "position_side": position["side"] if position is not None else "none",
                    "position_size": position["size"] if position is not None else 0.0,
                    "position_entry": position["entry"] if position is not None else np.nan,
                    "position_unrealized": unrealized,
                    "reward": reward,
                    "rationale": rationale,
                }
            )
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
                    partial_pnl = ((tp1 - position["entry"]) / position["entry"]) * position["size"] * tp1_pct
                    trades.append(partial_pnl)
                    reward += partial_pnl
                    trade_rows.append(
                        {
                            "timestamp": ts,
                            "dt": dt,
                            "side": "long",
                            "action": "tp1",
                            "entry_ts": position["ts"],
                            "entry_price": position["entry"],
                            "exit_price": tp1,
                            "size": position["size"] * tp1_pct,
                            "pnl": partial_pnl,
                            "rationale": rationale,
                        }
                    )
                    position["size"] *= (1.0 - tp1_pct)
                    position["tp1_hit"] = True
                    action = "tp1_long"
            else:
                tp1 = position["entry"] - tp1_atr * position["atr"]
                if price <= tp1:
                    partial_pnl = ((position["entry"] - tp1) / position["entry"]) * position["size"] * tp1_pct
                    trades.append(partial_pnl)
                    reward += partial_pnl
                    trade_rows.append(
                        {
                            "timestamp": ts,
                            "dt": dt,
                            "side": "short",
                            "action": "tp1",
                            "entry_ts": position["ts"],
                            "entry_price": position["entry"],
                            "exit_price": tp1,
                            "size": position["size"] * tp1_pct,
                            "pnl": partial_pnl,
                            "rationale": rationale,
                        }
                    )
                    position["size"] *= (1.0 - tp1_pct)
                    position["tp1_hit"] = True
                    action = "tp1_short"

        if (position["side"] == "long" and (price <= stop or price >= take)) or (
            position["side"] == "short" and (price >= stop or price <= take)
        ):
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            reward += trade_pnl
            trade_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "side": position["side"],
                    "action": "stop_or_tp",
                    "entry_ts": position["ts"],
                    "entry_price": position["entry"],
                    "exit_price": price,
                    "size": position["size"],
                    "pnl": trade_pnl,
                    "rationale": rationale,
                }
            )
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            action = "close_stop_or_tp"
            position = None
            bar_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "close": price,
                    "signal": signal,
                    "confidence": conf,
                    "action": action,
                    "position_side": "none",
                    "position_size": 0.0,
                    "position_entry": np.nan,
                    "position_unrealized": 0.0,
                    "reward": reward,
                    "rationale": rationale,
                }
            )
            continue

        # DCA / martingale
        if dca and position["dca_steps"] < max_dca:
            if position["side"] == "long":
                trigger = position["entry"] - (position["dca_steps"] + 1) * dca_atr * position["atr"]
                if price <= trigger:
                    add = position["base_size"] * (martingale ** (position["dca_steps"] + 1))
                    old_size = position["size"]
                    new_size = min(max_leverage, old_size + add)
                    add_eff = new_size - old_size
                    if add_eff > 0:
                        position["entry"] = (position["entry"] * old_size + price * add_eff) / new_size
                        position["size"] = new_size
                        position["dca_steps"] += 1
                        action = "dca_long"
                        trade_rows.append(
                            {
                                "timestamp": ts,
                                "dt": dt,
                                "side": position["side"],
                                "action": "dca_add",
                                "entry_ts": position["ts"],
                                "entry_price": position["entry"],
                                "exit_price": price,
                                "size": add_eff,
                                "pnl": 0.0,
                                "rationale": rationale,
                            }
                        )
            else:
                trigger = position["entry"] + (position["dca_steps"] + 1) * dca_atr * position["atr"]
                if price >= trigger:
                    add = position["base_size"] * (martingale ** (position["dca_steps"] + 1))
                    old_size = position["size"]
                    new_size = min(max_leverage, old_size + add)
                    add_eff = new_size - old_size
                    if add_eff > 0:
                        position["entry"] = (position["entry"] * old_size + price * add_eff) / new_size
                        position["size"] = new_size
                        position["dca_steps"] += 1
                        action = "dca_short"
                        trade_rows.append(
                            {
                                "timestamp": ts,
                                "dt": dt,
                                "side": position["side"],
                                "action": "dca_add",
                                "entry_ts": position["ts"],
                                "entry_price": position["entry"],
                                "exit_price": price,
                                "size": add_eff,
                                "pnl": 0.0,
                                "rationale": rationale,
                            }
                        )

        if position["side"] == "long" and signal == "sell":
            pnl = (price - position["entry"]) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            reward += trade_pnl
            trade_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "side": "long",
                    "action": "reverse_to_short",
                    "entry_ts": position["ts"],
                    "entry_price": position["entry"],
                    "exit_price": price,
                    "size": position["size"],
                    "pnl": trade_pnl,
                    "rationale": rationale,
                }
            )
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            base_size = min(base_size, max_leverage)
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
            trade_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "side": "short",
                    "action": "open",
                    "entry_ts": ts,
                    "entry_price": price,
                    "exit_price": price,
                    "size": base_size,
                    "pnl": 0.0,
                    "rationale": rationale,
                }
            )
            action = "reverse_to_short"
        elif position["side"] == "short" and signal == "buy":
            pnl = (position["entry"] - price) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            reward += trade_pnl
            trade_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "side": "short",
                    "action": "reverse_to_long",
                    "entry_ts": position["ts"],
                    "entry_price": position["entry"],
                    "exit_price": price,
                    "size": position["size"],
                    "pnl": trade_pnl,
                    "rationale": rationale,
                }
            )
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            base_size = min(base_size, max_leverage)
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
            trade_rows.append(
                {
                    "timestamp": ts,
                    "dt": dt,
                    "side": "long",
                    "action": "open",
                    "entry_ts": ts,
                    "entry_price": price,
                    "exit_price": price,
                    "size": base_size,
                    "pnl": 0.0,
                    "rationale": rationale,
                }
            )
            action = "reverse_to_long"

        unrealized = 0.0
        if position is not None:
            if position["side"] == "long":
                unrealized = ((price - position["entry"]) / position["entry"]) * position["size"]
            else:
                unrealized = ((position["entry"] - price) / position["entry"]) * position["size"]
        bar_rows.append(
            {
                "timestamp": ts,
                "dt": dt,
                "close": price,
                "signal": signal,
                "confidence": conf,
                "action": action,
                "position_side": position["side"] if position is not None else "none",
                "position_size": position["size"] if position is not None else 0.0,
                "position_entry": position["entry"] if position is not None else np.nan,
                "position_unrealized": unrealized,
                "reward": reward,
                "rationale": rationale,
            }
        )
        if (i + 1) % progress_mod == 0 or (i + 1) == total:
            print(f"Sim {render_progress(i + 1, total)}")

    return trades, trade_rows, bar_rows


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
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--trades-out", default="data/models/historical_sim_trades.csv")
    p.add_argument("--pred-out", default="data/models/historical_sim_predictions.csv")
    p.add_argument("--split-out", default="data/models/historical_sim_split.csv")
    p.add_argument("--bars-out", default="data/models/historical_sim_bars.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_features(df)
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    df = label_outcomes(df, horizon=args.horizon, threshold=args.threshold)
    df = df.dropna().reset_index(drop=True)
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    train = df.iloc[: args.train_bars]
    test = df.iloc[args.train_bars : args.train_bars + args.test_bars]

    model, feature_cols = train_model(train)
    preds = model.predict(test[feature_cols])

    max_lev = min(1.0, max(0.1, float(args.max_leverage)))
    trades, trade_rows, bar_rows = simulate_trades(
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
        max_leverage=max_lev,
        progress_step=max(1, args.test_bars // 40),
    )

    pred_labels = np.argmax(preds, axis=1) - 1
    pred_conf = np.max(preds, axis=1)
    keep_cols = [
        "timestamp",
        "_dt",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "label",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "mfi_14",
        "atr_14",
        "close_vs_sma50",
        "close_vs_sma200",
        "trend_up",
        "ob_bull_recent",
        "ob_bear_recent",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    pred_df = test[keep_cols].copy()
    pred_df["pred_label"] = pred_labels
    pred_df["confidence"] = pred_conf
    pred_df["phase"] = "test"

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
        f"DCA: {args.dca}  Martingale: {args.martingale:.2f}  DCA ATR: {args.dca_atr:.2f}  Max DCA: {args.max_dca}  Max Lev: {max_lev:.2f}\n\n"
        "Top factors (gain):\n" + "\n".join([f"- {k}: {v:.2f}" for k, v in top])
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(summary, encoding="ascii")
    pd.DataFrame(trade_rows).to_csv(args.trades_out, index=False)
    pd.DataFrame(bar_rows).to_csv(args.bars_out, index=False)
    pred_df.to_csv(args.pred_out, index=False)
    split_row = pd.DataFrame(
        [
            {
                "train_start_ts": int(train["timestamp"].iloc[0]),
                "train_end_ts": int(train["timestamp"].iloc[-1]),
                "test_start_ts": int(test["timestamp"].iloc[0]),
                "test_end_ts": int(test["timestamp"].iloc[-1]),
                "train_bars": int(len(train)),
                "test_bars": int(len(test)),
            }
        ]
    )
    split_row.to_csv(args.split_out, index=False)
    print(summary)


if __name__ == "__main__":
    main()
