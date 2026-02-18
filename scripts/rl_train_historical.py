import argparse
from datetime import datetime, timezone
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
from experiment_tracker import record_experiment


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


def rationale_key_from_state(state: str) -> str:
    if "|pos:" in state:
        return state.split("|pos:", 1)[0]
    return state


def choose_action(q, state: str, epsilon: float) -> str:
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    vals = [q[(state, a)] for a in ACTIONS]
    return ACTIONS[int(np.argmax(vals))]


def objective_score(test_eval: dict, return_weight: float, dd_penalty: float) -> float:
    # By default this is return-only. Drawdown penalty is optional via CLI.
    test_ret = float(test_eval.get("total_return", 0.0))
    test_dd = float(test_eval.get("max_drawdown", 0.0))
    return (return_weight * test_ret) - (dd_penalty * abs(test_dd))


def load_rationale_memory(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        mem = pd.read_csv(path)
    except Exception:
        return {}
    out = {}
    if "rationale_key" not in mem.columns or "action" not in mem.columns or "mean_reward" not in mem.columns:
        return out
    for _, r in mem.iterrows():
        rk = str(r["rationale_key"])
        a = str(r["action"])
        out[(rk, a)] = float(r.get("mean_reward", 0.0))
    return out


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
    rationale_prior: dict,
    rationale_prior_weight: float,
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
            if rationale_prior_weight > 0:
                reward += rationale_prior_weight * float(rationale_prior.get((rationale_key_from_state(s), a), 0.0))
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
    decision_log = []

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
        decision_log.append(
            {
                "state": s,
                "rationale_key": rationale_key_from_state(s),
                "action": a,
                "reward": reward,
            }
        )

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
            "decision_log": decision_log,
        }
    return {
        "bars": 0,
        "mean_reward": 0.0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "equity_curve": np.array([]),
        "actions": [],
        "decision_log": [],
    }


def build_rationale_assessment(decision_log: list[dict]) -> pd.DataFrame:
    if len(decision_log) == 0:
        return pd.DataFrame(
            columns=[
                "rationale_key",
                "action",
                "count",
                "win_rate",
                "mean_reward",
                "mean_abs_reward",
                "accuracy",
            ]
        )
    df = pd.DataFrame(decision_log)
    df["hit"] = (df["reward"] > 0).astype(int)
    df["abs_reward"] = df["reward"].abs()
    out = (
        df.groupby(["rationale_key", "action"], as_index=False)
        .agg(
            count=("reward", "count"),
            win_rate=("hit", "mean"),
            mean_reward=("reward", "mean"),
            mean_abs_reward=("abs_reward", "mean"),
        )
        .sort_values(["mean_reward", "win_rate", "count"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    out["accuracy"] = out["win_rate"]
    return out


def update_rationale_memory(mem_path: Path, assessed: pd.DataFrame):
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    if mem_path.exists():
        try:
            mem = pd.read_csv(mem_path)
        except Exception:
            mem = pd.DataFrame(columns=["rationale_key", "action", "count", "mean_reward", "win_rate", "updated_at"])
    else:
        mem = pd.DataFrame(columns=["rationale_key", "action", "count", "mean_reward", "win_rate", "updated_at"])
    if len(assessed) == 0:
        mem.to_csv(mem_path, index=False)
        return mem

    now = datetime.now(timezone.utc).isoformat()
    key_cols = ["rationale_key", "action"]
    for _, row in assessed.iterrows():
        rk = str(row["rationale_key"])
        ac = str(row["action"])
        cnt = int(row["count"])
        mr = float(row["mean_reward"])
        wr = float(row["win_rate"])
        mask = (mem["rationale_key"] == rk) & (mem["action"] == ac) if len(mem) > 0 else pd.Series([], dtype=bool)
        if len(mem) > 0 and mask.any():
            idx = mem.index[mask][0]
            old_cnt = int(mem.at[idx, "count"])
            new_cnt = old_cnt + cnt
            mem.at[idx, "mean_reward"] = (float(mem.at[idx, "mean_reward"]) * old_cnt + mr * cnt) / max(1, new_cnt)
            mem.at[idx, "win_rate"] = (float(mem.at[idx, "win_rate"]) * old_cnt + wr * cnt) / max(1, new_cnt)
            mem.at[idx, "count"] = new_cnt
            mem.at[idx, "updated_at"] = now
        else:
            row_df = pd.DataFrame(
                [
                    {
                        "rationale_key": rk,
                        "action": ac,
                        "count": cnt,
                        "mean_reward": mr,
                        "win_rate": wr,
                        "updated_at": now,
                    }
                ]
            )
            if len(mem) == 0:
                mem = row_df
            else:
                mem = pd.concat([mem, row_df], ignore_index=True)
    mem = mem.sort_values(["mean_reward", "win_rate", "count"], ascending=[False, False, False]).reset_index(drop=True)
    mem.to_csv(mem_path, index=False)
    return mem


def q_to_df(q):
    rows = [{"state": s, "action": a, "q": float(v)} for (s, a), v in q.items()]
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["state", "q"], ascending=[True, False]).reset_index(drop=True)


def classify_regime(test_df: pd.DataFrame, vol_baseline: float) -> tuple[str, str, float]:
    trend_mean = float(test_df["trend_up"].mean()) if "trend_up" in test_df.columns else 0.5
    if trend_mean >= 0.60:
        regime = "bull"
    elif trend_mean <= 0.40:
        regime = "bear"
    else:
        regime = "range"

    vol_med = float(test_df["vol_48"].median()) if "vol_48" in test_df.columns else 0.0
    vol_regime = "high" if vol_med > vol_baseline else "low"
    return regime, vol_regime, trend_mean


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
            rationale_prior=args._rationale_prior,
            rationale_prior_weight=args.rationale_prior_weight,
        )
        train_eval = evaluate_policy(train, q, args.fee_rate)
        test_eval = evaluate_policy(test, q, args.fee_rate)
        objective = objective_score(test_eval, args.return_weight, args.dd_penalty)
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
    p.add_argument("--return-weight", type=float, default=1.0)
    p.add_argument("--dd-penalty", type=float, default=0.0)
    p.add_argument("--rationale-prior-weight", type=float, default=0.05)
    p.add_argument("--rationale-memory", default="data/models/rl_rationale_memory.csv")
    p.add_argument("--rationale-out", default="data/models/rl_rationale_assessment.csv")
    p.add_argument("--rationale-summary-out", default="data/models/rl_rationale_summary.txt")
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--step-bars", type=int, default=24 * 30)
    p.add_argument("--purge-bars", type=int, default=12)
    p.add_argument("--wf-out", default="data/models/rl_walkforward.csv")
    p.add_argument("--wf-summary-out", default="data/models/rl_walkforward_summary.txt")
    p.add_argument("--exp-root", default="data/models/experiments")
    p.add_argument("--exp-source", default="rl_train_historical")
    p.add_argument("--exp-parent-run", default="")
    p.add_argument("--exp-tag", default="")
    p.add_argument("--exp-note", default="")
    p.add_argument("--no-exp", action="store_true")
    args = p.parse_args()
    args._rationale_prior = load_rationale_memory(Path(args.rationale_memory))
    started_at_utc = datetime.now(timezone.utc).isoformat()

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
        vol_baseline = float(df["vol_48"].median()) if "vol_48" in df.columns else 0.0
        wf_df = pd.DataFrame()
        total_windows = 0
        if len(df) >= args.train_bars + args.purge_bars + args.test_bars:
            total_windows = ((len(df) - args.train_bars - args.purge_bars - args.test_bars) // args.step_bars) + 1
        while start + args.train_bars + args.purge_bars + args.test_bars <= len(df):
            if deadline is not None and time.time() > deadline:
                print(f"time budget reached ({args.max_minutes} min), stopping walk-forward")
                break
            tr = df.iloc[start : start + args.train_bars].reset_index(drop=True)
            te_start = start + args.train_bars + args.purge_bars
            te_end = te_start + args.test_bars
            te = df.iloc[te_start:te_end].reset_index(drop=True)
            label = f"window {widx + 1}"
            res = optimize_split(tr, te, args, label=label, deadline=deadline)
            if res is None:
                break
            (objective, alpha_b, gamma_b, eps_s_b, eps_e_b, episodes_b), q, tr_eval, te_eval = res
            regime, vol_regime, trend_mean = classify_regime(te, vol_baseline)
            walk_rows.append(
                {
                    "window": widx + 1,
                    "train_start": int(start),
                    "train_end": int(start + args.train_bars),
                    "test_start": int(te_start),
                    "test_end": int(te_end),
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
                    "regime": regime,
                    "vol_regime": vol_regime,
                    "trend_mean": trend_mean,
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
        train_end = max(500, split - args.purge_bars)
        test_start = min(len(df) - 200, split + args.purge_bars)
        train = df.iloc[:train_end].reset_index(drop=True)
        test = df.iloc[test_start:].reset_index(drop=True)
        res = optimize_split(train, test, args, label="single-split", deadline=deadline)
        if res is None:
            raise RuntimeError("No trials completed.")
        best, best_q, best_train, best_test = res

    objective, alpha_b, gamma_b, eps_s_b, eps_e_b, episodes_b = best

    q_df = q_to_df(best_q)
    Path(args.q_out).parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(args.q_out, index=False)

    split_line = (
        f"train_bars={args.train_bars} purge_bars={args.purge_bars} test_bars={args.test_bars} step_bars={args.step_bars}\n\n"
        if args.walkforward
        else f"train_bars={len(train)} purge_bars={args.purge_bars} test_bars={len(test)}\n\n"
    )
    summary = (
        "RL Historical Training (No Borrowing)\n"
        f"mode={'walkforward' if args.walkforward else 'single_split'}\n"
        f"trials={max(1, args.optimize_trials)}\n"
        f"best episodes={episodes_b} alpha={alpha_b} gamma={gamma_b} eps_start={eps_s_b} eps_end={eps_e_b} fee_rate={args.fee_rate}\n"
        f"objective=return_weight*test_return - dd_penalty*abs(test_dd) where return_weight={args.return_weight} dd_penalty={args.dd_penalty}\n"
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
        regime_lines = []
        if "regime" in wf_df.columns and len(wf_df) > 0:
            grouped = wf_df.groupby("regime", as_index=False).agg(
                windows=("window", "count"),
                mean_objective=("objective", "mean"),
                mean_test_return=("test_return", "mean"),
                mean_test_dd=("test_dd", "mean"),
                positive_test_return_rate=("test_return", lambda s: float((s > 0).mean())),
            )
            grouped = grouped.sort_values("mean_objective", ascending=False)
            for _, r in grouped.iterrows():
                regime_lines.append(
                    f"- {r['regime']}: windows={int(r['windows'])} mean_obj={float(r['mean_objective']):.4f} "
                    f"ret={float(r['mean_test_return']):.2%} dd={float(r['mean_test_dd']):.2%} "
                    f"pos={float(r['positive_test_return_rate']):.2%}"
                )
        regime_block = ""
        if len(regime_lines) > 0:
            regime_block = "regimes\n" + "\n".join(regime_lines) + "\n"
        wf_summary = (
            "RL Walk-Forward Summary\n"
            f"windows={len(wf_df)}\n"
            f"mean_objective={cons_mean:.4f}\n"
            f"std_objective={cons_std:.4f}\n"
            f"consistency_score={cons_mean - cons_std:.4f}\n"
            f"positive_test_return_rate={pos_rate:.2%}\n"
            f"mean_test_return={float(wf_df['test_return'].mean()):.2%}\n"
            f"mean_test_dd={float(wf_df['test_dd'].mean()):.2%}\n"
            f"{regime_block}"
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

    assessed = build_rationale_assessment(best_test.get("decision_log", []))
    Path(args.rationale_out).parent.mkdir(parents=True, exist_ok=True)
    assessed.to_csv(args.rationale_out, index=False)
    mem_df = update_rationale_memory(Path(args.rationale_memory), assessed)
    overall_acc = float((pd.Series([d["reward"] for d in best_test.get("decision_log", [])]) > 0).mean()) if len(best_test.get("decision_log", [])) > 0 else 0.0
    top_lines = []
    if len(assessed) > 0:
        top = assessed.head(8)
        for _, r in top.iterrows():
            top_lines.append(
                f"- {r['rationale_key']} | {r['action']} | n={int(r['count'])} "
                f"acc={float(r['accuracy']):.2%} mean_reward={float(r['mean_reward']):.4f}"
            )
    rationale_summary = (
        "RL Rationale Assessment\n"
        f"decisions={len(best_test.get('decision_log', []))}\n"
        f"overall_accuracy={overall_acc:.2%}\n"
        f"patterns={len(assessed)}\n"
        f"memory_rows={len(mem_df)}\n"
        f"prior_weight={args.rationale_prior_weight}\n\n"
        "Top rationale patterns\n"
        + ("\n".join(top_lines) if len(top_lines) > 0 else "none")
        + "\n"
    )
    Path(args.rationale_summary_out).write_text(rationale_summary, encoding="ascii")

    print(summary)
    print(rationale_summary)
    if not args.no_exp:
        params_payload = {
            "input": args.input,
            "mode": "walkforward" if args.walkforward else "single_split",
            "episodes": int(episodes_b),
            "alpha": float(alpha_b),
            "gamma": float(gamma_b),
            "eps_start": float(eps_s_b),
            "eps_end": float(eps_e_b),
            "optimize_trials": int(args.optimize_trials),
            "reward_clip": float(args.reward_clip),
            "fee_rate": float(args.fee_rate),
            "return_weight": float(args.return_weight),
            "dd_penalty": float(args.dd_penalty),
            "rationale_prior_weight": float(args.rationale_prior_weight),
            "train_ratio": float(args.train_ratio),
            "train_bars": int(args.train_bars),
            "test_bars": int(args.test_bars),
            "step_bars": int(args.step_bars),
            "purge_bars": int(args.purge_bars),
            "max_minutes": float(args.max_minutes),
        }
        wf_windows = int(len(wf_df)) if args.walkforward else 0
        wf_consistency = float(wf_df["objective"].mean() - wf_df["objective"].std(ddof=0)) if args.walkforward and len(wf_df) > 0 else 0.0
        metrics_payload = {
            "objective": float(objective),
            "train_return": float(best_train["total_return"]),
            "train_dd": float(best_train["max_drawdown"]),
            "train_win_rate": float(best_train["win_rate"]),
            "test_return": float(best_test["total_return"]),
            "test_dd": float(best_test["max_drawdown"]),
            "test_win_rate": float(best_test["win_rate"]),
            "test_mean_reward": float(best_test["mean_reward"]),
            "walkforward_windows": wf_windows,
            "walkforward_consistency_score": wf_consistency,
        }
        artifacts = {
            "q_table": args.q_out,
            "train_summary": args.summary_out,
            "policy": args.policy_out,
            "test_equity": args.equity_out,
            "rationale_assessment": args.rationale_out,
            "rationale_summary": args.rationale_summary_out,
            "rationale_memory": args.rationale_memory,
            "walkforward_windows": args.wf_out if args.walkforward else None,
            "walkforward_summary": args.wf_summary_out if args.walkforward else None,
        }
        exp = record_experiment(
            exp_root=Path(args.exp_root),
            source=args.exp_source,
            mode="walkforward" if args.walkforward else "single_split",
            params=params_payload,
            metrics=metrics_payload,
            artifacts=artifacts,
            parent_run_id=args.exp_parent_run,
            tag=args.exp_tag,
            note=args.exp_note,
            run_prefix="rl",
            started_at_utc=started_at_utc,
        )
        last_path = Path("data/models/rl_last_run.txt")
        last_path.parent.mkdir(parents=True, exist_ok=True)
        last_path.write_text(
            f"run_id={exp['run_id']}\nrun_dir={exp['run_dir']}\ncreated_at_utc={exp['created_at_utc']}\n",
            encoding="ascii",
        )
        print(f"Recorded experiment: {exp['run_id']}")
    print(f"Wrote Q-table to {args.q_out}")


if __name__ == "__main__":
    main()
