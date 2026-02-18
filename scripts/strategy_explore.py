import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

from signal_engine import build_signals
from progress_bar import render_progress


def fmt_strategy_line(r: pd.Series) -> str:
    return (
        f"- mfi={int(r.mfi_lower)}/{int(r.mfi_upper)} ob={int(r.ob_lookback)} trend={bool(r.trend_filter)} "
        f"dca={bool(r.dca)} mg={r.martingale} dca_atr={r.dca_atr} st_tp={bool(r.stagger_tp)} "
        f"sl_atr={r.sl_atr} tp_atr={r.tp_atr} tp1={r.tp1_atr} tp2={r.tp2_atr} "
        f"win={r.win_rate:.2%} avg={r.avg_trade:.2%} dd={r.max_dd:.2%} ret={r.total_ret:.2%} score={r.score:.4f}"
    )


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


def simulate_trades(
    df,
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
):
    position = None
    trades = []
    loss_streak = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        atr_val = float(row["atr_14"]) if not np.isnan(row["atr_14"]) else None

        # stop loss / take profit
        if position is not None:
            if atr_val is not None:
                if position["side"] == "long":
                    stop = position["entry"] - sl_atr * position["atr"]
                    take = position["entry"] + (tp2_atr if stagger_tp else tp_atr) * position["atr"]
                    pnl = (price - position["entry"]) / position["entry"]
                else:
                    stop = position["entry"] + sl_atr * position["atr"]
                    take = position["entry"] - (tp2_atr if stagger_tp else tp_atr) * position["atr"]
                    pnl = (position["entry"] - price) / position["entry"]

                # Staggered TP (partial)
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
                            old_size = position["size"]
                            new_size = min(max_leverage, old_size + add)
                            add_eff = new_size - old_size
                            if add_eff > 0:
                                position["entry"] = (position["entry"] * old_size + price * add_eff) / new_size
                                position["size"] = new_size
                                position["dca_steps"] += 1
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

        # signal-based transitions
        if position is None:
            if bool(row.get("buy")) and atr_val is not None:
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                base_size = min(base_size, max_leverage)
                position = {"side": "long", "entry": price, "atr": atr_val, "size": base_size, "base_size": base_size, "dca_steps": 0, "tp1_hit": False}
            elif bool(row.get("sell")) and atr_val is not None:
                base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
                base_size = min(base_size, max_leverage)
                position = {"side": "short", "entry": price, "atr": atr_val, "size": base_size, "base_size": base_size, "dca_steps": 0, "tp1_hit": False}
            continue

        if position["side"] == "long" and bool(row.get("sell")):
            pnl = (price - position["entry"]) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            base_size = min(base_size, max_leverage)
            position = {"side": "short", "entry": price, "atr": atr_val, "size": base_size, "base_size": base_size, "dca_steps": 0, "tp1_hit": False}
        elif position["side"] == "short" and bool(row.get("buy")):
            pnl = (position["entry"] - price) / position["entry"]
            trade_pnl = pnl * position["size"]
            trades.append(trade_pnl)
            if trade_pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            base_size = 1.0 * (martingale ** loss_streak) if martingale > 1.0 else 1.0
            base_size = min(base_size, max_leverage)
            position = {"side": "long", "entry": price, "atr": atr_val, "size": base_size, "base_size": base_size, "dca_steps": 0, "tp1_hit": False}

    return trades


def metrics(trades):
    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(trades, dtype=float)
    arr = np.clip(arr, -0.999, 10.0)
    total = len(arr)
    win_rate = float((arr > 0).mean())
    avg = float(arr.mean())
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    total_ret = float(equity[-1] - 1)
    return total, win_rate, avg, max_dd, total_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--out", default="data/models/strategy_explore.csv")
    p.add_argument("--summary", default="data/models/strategy_explore_summary.txt")
    p.add_argument("--max-combos", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--min-total-ret", type=float, default=-0.99)
    p.add_argument("--min-max-dd", type=float, default=-0.99)
    p.add_argument("--min-trades", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    mfi_lower = [30, 35, 40, 45]
    mfi_upper = [65, 70, 80]
    ob_lookback = [6, 12, 18]
    trend_filter = [True, False]
    sl_atr = [1.5, 2.0, 2.5]
    tp_atr = [2.5, 3.0, 4.0]
    tp1_atr = [1.5, 2.0]
    tp2_atr = [3.0, 4.0]
    stagger_tp = [False, True]
    dca_opts = [False, True]
    martingale_opts = [1.0, 1.2, 1.4, 1.6, 2.0]
    dca_atr = [0.8, 1.0]

    rows = []
    combos = []
    for mfi_l, mfi_u, ob_l, trend, sl, dca_flag, mg, dca_a, st in itertools.product(
        mfi_lower, mfi_upper, ob_lookback, trend_filter, sl_atr, dca_opts, martingale_opts, dca_atr, stagger_tp
    ):
        if st:
            for t1, t2 in itertools.product(tp1_atr, tp2_atr):
                combos.append((mfi_l, mfi_u, ob_l, trend, sl, None, dca_flag, mg, dca_a, st, t1, t2))
        else:
            for tp in tp_atr:
                combos.append((mfi_l, mfi_u, ob_l, trend, sl, tp, dca_flag, mg, dca_a, st, None, None))
    if args.max_combos and args.max_combos < len(combos):
        combos = combos[: args.max_combos]
    combo_total = len(combos)
    for i, (mfi_l, mfi_u, ob_l, trend, sl, tp, dca_flag, mg, dca_a, st, t1, t2) in enumerate(combos, start=1):
        sig = build_signals(
            df,
            mfi_lower=mfi_l,
            mfi_upper=mfi_u,
            ob_lookback=ob_l,
            trend_filter=trend,
        )
        trades = simulate_trades(
            sig,
            sl_atr=sl,
            tp_atr=tp if tp is not None else 4.0,
            stagger_tp=st,
            tp1_atr=t1 if t1 is not None else 2.0,
            tp2_atr=t2 if t2 is not None else 4.0,
            dca=dca_flag,
            dca_atr=dca_a,
            martingale=mg,
            max_dca=2,
            max_leverage=min(1.0, max(0.1, float(args.max_leverage))),
        )
        trades_count, win_rate, avg, max_dd, total_ret = metrics(trades)
        # score balances return vs drawdown (no hard cap)
        score = total_ret - 0.5 * abs(max_dd)
        rows.append({
            "mfi_lower": mfi_l,
            "mfi_upper": mfi_u,
            "ob_lookback": ob_l,
            "trend_filter": trend,
            "sl_atr": sl,
            "tp_atr": tp if tp is not None else None,
            "tp1_atr": t1,
            "tp2_atr": t2,
            "stagger_tp": st,
            "dca": dca_flag,
            "martingale": mg,
            "dca_atr": dca_a,
            "trades": trades_count,
            "win_rate": win_rate,
            "avg_trade": avg,
            "max_dd": max_dd,
            "total_ret": total_ret,
            "score": score,
        })
        # simple progress
        if args.sleep:
            import time
            time.sleep(args.sleep)
        if i % 5 == 0 or i == combo_total:
            print(f"Progress {render_progress(i, combo_total)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("score", ascending=False)
    df_out.to_csv(out, index=False)

    viable_mask = (
        np.isfinite(df_out["score"])
        & np.isfinite(df_out["total_ret"])
        & np.isfinite(df_out["max_dd"])
        & (df_out["trades"] >= args.min_trades)
        & (df_out["total_ret"] > args.min_total_ret)
        & (df_out["max_dd"] > args.min_max_dd)
    )
    df_viable = df_out[viable_mask].sort_values("score", ascending=False)

    if df_viable.empty:
        raw_top = df_out.head(5)
        raw_lines = []
        for _, r in raw_top.iterrows():
            raw_lines.append(fmt_strategy_line(r))
        summary = (
            "No viable strategies found\n"
            f"Criteria: trades>={args.min_trades}, total_ret>{args.min_total_ret:.2%}, "
            f"max_dd>{args.min_max_dd:.2%}, finite score/max_dd/total_ret\n\n"
            "Top 5 raw strategies (debug)\n" + "\n".join(raw_lines) + "\n"
        )
        Path(args.summary).write_text(summary, encoding="ascii")
        print(summary)
        print("Skipped writing data/models/best_strategy.txt because no viable strategy passed filters.")
        return

    best = df_viable.iloc[0]
    worst = df_viable.iloc[-1]

    top5 = df_viable.head(5)
    top_lines = []
    for _, r in top5.iterrows():
        top_lines.append(fmt_strategy_line(r))

    summary = (
        f"Viable strategies: {len(df_viable)}/{len(df_out)}\n\n"
        "Best strategy\n"
        f"mfi_lower={best.mfi_lower} mfi_upper={best.mfi_upper} ob_lookback={best.ob_lookback} trend={best.trend_filter} dca={best.dca} mg={best.martingale}\n"
        f"sl_atr={best.sl_atr} tp_atr={best.tp_atr} tp1={best.tp1_atr} tp2={best.tp2_atr} st_tp={best.stagger_tp} trades={best.trades}\n"
        f"win={best.win_rate:.2%} avg={best.avg_trade:.2%} dd={best.max_dd:.2%} ret={best.total_ret:.2%} score={best.score:.4f}\n\n"
        "Top 5 strategies\n" + "\n".join(top_lines) + "\n\n"
        "Worst strategy\n"
        f"mfi_lower={worst.mfi_lower} mfi_upper={worst.mfi_upper} ob_lookback={worst.ob_lookback} trend={worst.trend_filter} dca={worst.dca} mg={worst.martingale}\n"
        f"sl_atr={worst.sl_atr} tp_atr={worst.tp_atr} tp1={worst.tp1_atr} tp2={worst.tp2_atr} st_tp={worst.stagger_tp} trades={worst.trades}\n"
        f"win={worst.win_rate:.2%} avg={worst.avg_trade:.2%} dd={worst.max_dd:.2%} ret={worst.total_ret:.2%} score={worst.score:.4f}\n"
    )

    Path(args.summary).write_text(summary, encoding="ascii")
    # Persist best strategy for reuse
    best_path = Path("data/models/best_strategy.txt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(
        f"mfi_lower={int(best.mfi_lower)}\n"
        f"mfi_upper={int(best.mfi_upper)}\n"
        f"ob_lookback={int(best.ob_lookback)}\n"
        f"trend_filter={bool(best.trend_filter)}\n"
        f"dca={bool(best.dca)}\n"
        f"martingale={float(best.martingale)}\n"
        f"max_dca=2\n"
        f"dca_atr={float(best.dca_atr)}\n"
        f"sl_atr={float(best.sl_atr)}\n"
        f"tp_atr={float(best.tp_atr) if best.tp_atr==best.tp_atr else 0}\n"
        f"stagger_tp={bool(best.stagger_tp)}\n"
        f"tp1_atr={float(best.tp1_atr) if best.tp1_atr==best.tp1_atr else 0}\n"
        f"tp2_atr={float(best.tp2_atr) if best.tp2_atr==best.tp2_atr else 0}\n"
        f"tp1_pct=0.5\n",
        encoding="ascii",
    )
    print(summary)


if __name__ == "__main__":
    main()
