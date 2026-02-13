import argparse
from collections import defaultdict
from pathlib import Path
import sys
from pathlib import Path as _Path
import time

import numpy as np
import pandas as pd

sys.path.append(str((_Path(__file__).resolve().parent)))
from ml_features import build_features
from progress_bar import render_progress


ACTIONS = ("buy", "sell", "hold")


def bucket_mfi(v: float) -> str:
    if v < 30:
        return "low"
    if v > 70:
        return "high"
    return "mid"


def bucket_macd(v: float) -> str:
    return "up" if v >= 0 else "down"


def state_from_row(row: pd.Series, pos_side: str) -> str:
    trend = "bull" if int(row.get("trend_up", 0)) == 1 else "bear"
    mfi = bucket_mfi(float(row.get("mfi_14", 50.0)))
    macd = bucket_macd(float(row.get("macd_hist", 0.0)))
    ob_bull = int(row.get("ob_bull_recent", 0))
    ob_bear = int(row.get("ob_bear_recent", 0))
    ob = "bull" if ob_bull and not ob_bear else "bear" if ob_bear and not ob_bull else "mix"
    return f"{trend}|mfi:{mfi}|macd:{macd}|ob:{ob}|pos:{pos_side}"


def choose_action(q, state: str, epsilon: float) -> str:
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    vals = [q[(state, a)] for a in ACTIONS]
    return ACTIONS[int(np.argmax(vals))]


def step_env(action: str, pos_side: str, entry: float | None, price: float, next_price: float, fee_rate: float):
    reward = 0.0

    # Spot-only transitions: no shorting, no borrowing.
    if action == "buy":
        if pos_side == "none":
            pos_side = "long"
            entry = price
            reward -= fee_rate
    elif action == "sell":
        if pos_side == "long" and entry is not None:
            reward += (price - entry) / entry
            reward -= fee_rate
            pos_side = "none"
            entry = None

    # Unrealized next-bar mark-to-market reward (no leverage)
    if pos_side == "long":
        reward += (next_price - price) / price

    return reward, pos_side, entry


def train_q_learning(
    df: pd.DataFrame,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    reward_clip: float,
    fee_rate: float,
):
    q = defaultdict(float)

    n = len(df)
    for ep in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))
        if (ep + 1) % max(1, episodes // 10) == 0 or ep == 0 or ep == episodes - 1:
            print(f"  episode {render_progress(ep + 1, episodes)}")
        pos_side = "none"
        entry = None

        for i in range(n - 1):
            row = df.iloc[i]
            nxt = df.iloc[i + 1]
            s = state_from_row(row, pos_side)
            a = choose_action(q, s, epsilon)
            reward, next_pos_side, next_entry = step_env(a, pos_side, entry, float(row["close"]), float(nxt["close"]), fee_rate)
            reward = float(np.clip(reward, -reward_clip, reward_clip))
            s2 = state_from_row(nxt, next_pos_side)
            best_next = max(q[(s2, aa)] for aa in ACTIONS)
            q[(s, a)] = q[(s, a)] + alpha * (reward + gamma * best_next - q[(s, a)])
            pos_side, entry = next_pos_side, next_entry

    return q


def evaluate_policy(df: pd.DataFrame, q, fee_rate: float):
    pos_side = "none"
    entry = None
    rewards = []
    actions = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        s = state_from_row(row, pos_side)
        vals = [q[(s, a)] for a in ACTIONS]
        a = ACTIONS[int(np.argmax(vals))]
        reward, pos_side, entry = step_env(a, pos_side, entry, float(row["close"]), float(nxt["close"]), fee_rate)
        reward = float(np.clip(reward, -0.20, 0.20))
        rewards.append(reward)
        actions.append(a)

    if rewards:
        arr = np.array(rewards, dtype=float)
        equity = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return {
            "bars": len(rewards),
            "mean_reward": float(arr.mean()),
            "win_rate": float((arr > 0).mean()),
            "total_return": float(equity[-1] - 1.0),
            "max_drawdown": float(dd.min()),
            "equity_curve": equity,
            "actions": actions,
        }
    return {"bars": 0, "mean_reward": 0.0, "win_rate": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "equity_curve": np.array([]), "actions": []}


def q_to_df(q):
    rows = [{"state": s, "action": a, "q": float(v)} for (s, a), v in q.items()]
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["state", "q"], ascending=[True, False]).reset_index(drop=True)


def build_trial_grid(args):
    trial_grid = []
    episode_choices = [max(1, int(args.episodes))]
    for t in range(max(1, args.optimize_trials)):
        if t == 0:
            trial_grid.append((args.alpha, args.gamma, args.eps_start, args.eps_end, args.episodes))
        else:
            trial_grid.append(
                (
                    float(np.random.choice([0.05, 0.1, 0.15, 0.2, 0.25])),
                    float(np.random.choice([0.9, 0.93, 0.95, 0.97, 0.99])),
                    float(np.random.choice([0.1, 0.2, 0.3, 0.4])),
                    float(np.random.choice([0.01, 0.02, 0.05])),
                    int(np.random.choice(episode_choices)),
                )
            )
    return trial_grid


def optimize_split(train: pd.DataFrame, test: pd.DataFrame, args, label: str, deadline: float | None = None):
    trial_grid = build_trial_grid(args)
    best = None
    best_q = None
    best_train = None
    best_test = None

    total_trials = len(trial_grid)
    for idx, (alpha, gamma, eps_s, eps_e, episodes) in enumerate(trial_grid, start=1):
        if deadline is not None and time.time() > deadline:
            print(f"time budget reached during {label}, stopping early")
            break
        print(
            f"{label} | trial {idx}/{total_trials} ({idx/total_trials:.0%}) | "
            f"alpha={alpha} gamma={gamma} eps={eps_s}->{eps_e} episodes={episodes}"
        )
        q = train_q_learning(
            train,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=eps_s,
            epsilon_end=eps_e,
            reward_clip=args.reward_clip,
            fee_rate=args.fee_rate,
        )
        train_eval = evaluate_policy(train, q, args.fee_rate)
        test_eval = evaluate_policy(test, q, args.fee_rate)
        objective = test_eval["total_return"] - 0.5 * abs(test_eval["max_drawdown"])
        print(
            f"  result | train_ret={train_eval['total_return']:.2%} test_ret={test_eval['total_return']:.2%} "
            f"test_dd={test_eval['max_drawdown']:.2%} objective={objective:.4f}"
        )
        print(f"  trials {render_progress(idx, total_trials)}")
        cand = (objective, alpha, gamma, eps_s, eps_e, episodes)
        if best is None or cand[0] > best[0]:
            best = cand
            best_q = q
            best_train = train_eval
            best_test = test_eval

    if best is None:
        return None
    return best, best_q, best_train, best_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=0.25)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--q-out", default="data/models/rl_qtable.csv")
    p.add_argument("--summary-out", default="data/models/rl_train_summary.txt")
    p.add_argument("--policy-out", default="data/models/rl_policy_latest.txt")
    p.add_argument("--equity-out", default="data/models/rl_test_equity.csv")
    p.add_argument("--optimize-trials", type=int, default=12)
    p.add_argument("--reward-clip", type=float, default=0.20)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--step-bars", type=int, default=24 * 30)
    p.add_argument("--wf-out", default="data/models/rl_walkforward.csv")
    p.add_argument("--wf-summary-out", default="data/models/rl_walkforward_summary.txt")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_features(df).dropna().reset_index(drop=True)

    deadline = (time.time() + args.max_minutes * 60) if args.max_minutes > 0 else None
    best = None
    best_q = None
    best_train = None
    best_test = None
    walk_rows = []

    if args.walkforward:
        start = 0
        widx = 0
        total_windows = 0
        if len(df) >= args.train_bars + args.test_bars:
            total_windows = ((len(df) - args.train_bars - args.test_bars) // args.step_bars) + 1
        while start + args.train_bars + args.test_bars <= len(df):
            if deadline is not None and time.time() > deadline:
                print(f"time budget reached ({args.max_minutes} min), stopping walk-forward")
                break
            tr = df.iloc[start : start + args.train_bars].reset_index(drop=True)
            te = df.iloc[start + args.train_bars : start + args.train_bars + args.test_bars].reset_index(drop=True)
            label = f"window {widx + 1}"
            res = optimize_split(tr, te, args, label=label, deadline=deadline)
            if res is None:
                break
            (objective, alpha_b, gamma_b, eps_s_b, eps_e_b, episodes_b), q, tr_eval, te_eval = res
            walk_rows.append(
                {
                    "window": widx + 1,
                    "train_start": int(start),
                    "train_end": int(start + args.train_bars),
                    "test_end": int(start + args.train_bars + args.test_bars),
                    "objective": objective,
                    "alpha": alpha_b,
                    "gamma": gamma_b,
                    "eps_start": eps_s_b,
                    "eps_end": eps_e_b,
                    "episodes": episodes_b,
                    "train_return": tr_eval["total_return"],
                    "test_return": te_eval["total_return"],
                    "test_dd": te_eval["max_drawdown"],
                    "test_win_rate": te_eval["win_rate"],
                }
            )
            cand = (objective, alpha_b, gamma_b, eps_s_b, eps_e_b, episodes_b)
            if best is None or cand[0] > best[0]:
                best = cand
                best_q = q
                best_train = tr_eval
                best_test = te_eval
            start += args.step_bars
            widx += 1
            if total_windows > 0:
                print(f"walk-forward {render_progress(widx, total_windows)}")
        if len(walk_rows) == 0:
            raise RuntimeError("No walk-forward windows were evaluated.")
    else:
        split = int(len(df) * args.train_ratio)
        split = max(500, min(split, len(df) - 200))
        train = df.iloc[:split].reset_index(drop=True)
        test = df.iloc[split:].reset_index(drop=True)
        res = optimize_split(train, test, args, label="single-split", deadline=deadline)
        if res is None:
            raise RuntimeError("No trials completed.")
        best, best_q, best_train, best_test = res

    objective, alpha_b, gamma_b, eps_s_b, eps_e_b, episodes_b = best

    q_df = q_to_df(best_q)
    Path(args.q_out).parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(args.q_out, index=False)

    split_line = (
        f"train_bars={args.train_bars} test_bars={args.test_bars} step_bars={args.step_bars}\n\n"
        if args.walkforward
        else f"train_bars={len(train)} test_bars={len(test)}\n\n"
    )
    summary = (
        "RL Historical Training (No Borrowing)\n"
        f"mode={'walkforward' if args.walkforward else 'single_split'}\n"
        f"trials={max(1, args.optimize_trials)}\n"
        f"best episodes={episodes_b} alpha={alpha_b} gamma={gamma_b} eps_start={eps_s_b} eps_end={eps_e_b} fee_rate={args.fee_rate}\n"
        f"{split_line}"
        "Train metrics\n"
        f"mean_reward={best_train['mean_reward']:.6f} win_rate={best_train['win_rate']:.2%} total_return={best_train['total_return']:.2%} max_dd={best_train['max_drawdown']:.2%}\n\n"
        "Test metrics\n"
        f"mean_reward={best_test['mean_reward']:.6f} win_rate={best_test['win_rate']:.2%} total_return={best_test['total_return']:.2%} max_dd={best_test['max_drawdown']:.2%}\n"
        f"objective={objective:.4f}\n"
    )
    Path(args.summary_out).write_text(summary, encoding="ascii")

    if args.walkforward:
        wf_df = pd.DataFrame(walk_rows)
        wf_df.to_csv(args.wf_out, index=False)
        cons_mean = float(wf_df["objective"].mean())
        cons_std = float(wf_df["objective"].std(ddof=0)) if len(wf_df) > 1 else 0.0
        pos_rate = float((wf_df["test_return"] > 0).mean())
        wf_summary = (
            "RL Walk-Forward Summary\n"
            f"windows={len(wf_df)}\n"
            f"mean_objective={cons_mean:.4f}\n"
            f"std_objective={cons_std:.4f}\n"
            f"consistency_score={cons_mean - cons_std:.4f}\n"
            f"positive_test_return_rate={pos_rate:.2%}\n"
            f"mean_test_return={float(wf_df['test_return'].mean()):.2%}\n"
            f"mean_test_dd={float(wf_df['test_dd'].mean()):.2%}\n"
        )
        Path(args.wf_summary_out).write_text(wf_summary, encoding="ascii")

    # Save latest policy action map by state (argmax Q)
    if len(q_df) > 0:
        policy = q_df.sort_values(["state", "q"], ascending=[True, False]).drop_duplicates("state")
        policy = policy[["state", "action", "q"]]
        policy.to_csv(args.policy_out, index=False)
    else:
        Path(args.policy_out).write_text("state,action,q\n", encoding="ascii")

    if len(best_test["equity_curve"]) > 0:
        eq_df = pd.DataFrame({"step": np.arange(len(best_test["equity_curve"])), "equity": best_test["equity_curve"]})
        eq_df.to_csv(args.equity_out, index=False)
    else:
        Path(args.equity_out).write_text("step,equity\n", encoding="ascii")

    print(summary)
    print(f"Wrote Q-table to {args.q_out}")


if __name__ == "__main__":
    main()
