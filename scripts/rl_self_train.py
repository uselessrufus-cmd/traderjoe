import argparse
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
import pandas as pd


def parse_kv_text(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_pct(v: str, default: float = 0.0) -> float:
    try:
        return float(v.strip().replace("%", "")) / 100.0
    except Exception:
        return default


def parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v.strip())
    except Exception:
        return default


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def progress_bar(done: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {done}/{total} ({done/total:.0%})"


def parse_best_params(summary_path: Path, fallback: dict) -> dict:
    if not summary_path.exists():
        return dict(fallback)
    txt = summary_path.read_text(encoding="ascii", errors="ignore")
    m = re.search(
        r"best episodes=(?P<episodes>\d+)\s+alpha=(?P<alpha>[-0-9.]+)\s+gamma=(?P<gamma>[-0-9.]+)\s+eps_start=(?P<eps_start>[-0-9.]+)\s+eps_end=(?P<eps_end>[-0-9.]+)",
        txt,
    )
    if not m:
        return dict(fallback)
    out = dict(fallback)
    out["episodes"] = int(m.group("episodes"))
    out["alpha"] = float(m.group("alpha"))
    out["gamma"] = float(m.group("gamma"))
    out["eps_start"] = float(m.group("eps_start"))
    out["eps_end"] = float(m.group("eps_end"))
    return out


def mutate_params(base: dict) -> dict:
    p = dict(base)
    p["alpha"] = min(0.30, max(0.03, p["alpha"] * random.uniform(0.85, 1.15)))
    p["gamma"] = min(0.995, max(0.85, p["gamma"] + random.uniform(-0.02, 0.02)))
    p["eps_start"] = min(0.50, max(0.05, p["eps_start"] + random.uniform(-0.08, 0.08)))
    p["eps_end"] = min(0.10, max(0.005, p["eps_end"] + random.uniform(-0.02, 0.02)))
    if p["eps_end"] > p["eps_start"]:
        p["eps_end"] = max(0.005, p["eps_start"] * 0.5)
    eps = max(1, int(base["episodes"]))
    p["episodes"] = max(1, int(random.choice([eps, max(1, eps - 1), eps + 1])))
    return p


def run_round(py: Path, base: Path, args, params: dict) -> tuple[bool, str]:
    return run_round_profile(
        py,
        base,
        args,
        params,
        {
            "train_bars": args.train_bars,
            "test_bars": args.test_bars,
            "step_bars": args.step_bars,
        },
    )


def run_round_profile(py: Path, base: Path, args, params: dict, profile: dict) -> tuple[bool, str]:
    cmd = [
        str(py),
        str(base / "rl_train_historical.py"),
        "--walkforward",
        "--episodes",
        str(params["episodes"]),
        "--alpha",
        str(params["alpha"]),
        "--gamma",
        str(params["gamma"]),
        "--eps-start",
        str(params["eps_start"]),
        "--eps-end",
        str(params["eps_end"]),
        "--optimize-trials",
        str(args.optimize_trials),
        "--train-bars",
        str(profile["train_bars"]),
        "--test-bars",
        str(profile["test_bars"]),
        "--step-bars",
        str(profile["step_bars"]),
        "--max-minutes",
        str(profile.get("max_minutes", args.per_round_minutes)),
        "--reward-clip",
        str(args.reward_clip),
        "--fee-rate",
        str(args.fee_rate),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, proc.stdout[-2000:]


def evaluate_params(py: Path, base: Path, args, params: dict, profiles: list[dict]) -> tuple[bool, dict, str]:
    score_vals = []
    ret_vals = []
    dd_vals = []
    pos_vals = []
    tail = ""
    for pf in profiles:
        ok, tail = run_round_profile(py, base, args, params, pf)
        if not ok:
            return False, {}, tail
        wf_summary = parse_kv_text(Path("data/models/rl_walkforward_summary.txt"))
        score_vals.append(parse_float(wf_summary.get("consistency_score", "-1e9"), -1e9))
        ret_vals.append(parse_pct(wf_summary.get("mean_test_return", "0%"), 0.0))
        dd_vals.append(parse_pct(wf_summary.get("mean_test_dd", "0%"), -1.0))
        pos_vals.append(parse_pct(wf_summary.get("positive_test_return_rate", "0%"), 0.0))
    if len(score_vals) == 0:
        return False, {}, tail
    out = {
        "score": float(sum(score_vals) / len(score_vals)),
        "score_std": float(pd.Series(score_vals).std(ddof=0)),
        "mean_ret": float(sum(ret_vals) / len(ret_vals)),
        "mean_dd": float(sum(dd_vals) / len(dd_vals)),
        "pos_rate": float(sum(pos_vals) / len(pos_vals)),
    }
    return True, out, tail


def load_pool(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(
        columns=[
            "strategy_id",
            "episodes",
            "alpha",
            "gamma",
            "eps_start",
            "eps_end",
            "score",
            "mean_test_return",
            "mean_test_dd",
            "positive_test_return_rate",
            "eval_count",
            "accepted_count",
            "last_round",
        ]
    )


def strategy_id(params: dict) -> str:
    return (
        f"ep={int(params['episodes'])}|a={float(params['alpha']):.4f}|g={float(params['gamma']):.4f}|"
        f"es={float(params['eps_start']):.4f}|ee={float(params['eps_end']):.4f}"
    )


def upsert_pool_row(pool: pd.DataFrame, params: dict, score: float, mean_ret: float, mean_dd: float, pos_rate: float, rnd: int, accepted: bool) -> pd.DataFrame:
    sid = strategy_id(params)
    if len(pool) > 0 and sid in set(pool["strategy_id"].astype(str)):
        idx = pool.index[pool["strategy_id"] == sid][0]
        prev_n = int(pool.at[idx, "eval_count"])
        n = prev_n + 1
        pool.at[idx, "score"] = (float(pool.at[idx, "score"]) * prev_n + score) / n
        pool.at[idx, "mean_test_return"] = (float(pool.at[idx, "mean_test_return"]) * prev_n + mean_ret) / n
        pool.at[idx, "mean_test_dd"] = (float(pool.at[idx, "mean_test_dd"]) * prev_n + mean_dd) / n
        pool.at[idx, "positive_test_return_rate"] = (float(pool.at[idx, "positive_test_return_rate"]) * prev_n + pos_rate) / n
        pool.at[idx, "eval_count"] = n
        pool.at[idx, "accepted_count"] = int(pool.at[idx, "accepted_count"]) + (1 if accepted else 0)
        pool.at[idx, "last_round"] = rnd
        return pool
    row = {
        "strategy_id": sid,
        "episodes": int(params["episodes"]),
        "alpha": float(params["alpha"]),
        "gamma": float(params["gamma"]),
        "eps_start": float(params["eps_start"]),
        "eps_end": float(params["eps_end"]),
        "score": float(score),
        "mean_test_return": float(mean_ret),
        "mean_test_dd": float(mean_dd),
        "positive_test_return_rate": float(pos_rate),
        "eval_count": 1,
        "accepted_count": 1 if accepted else 0,
        "last_round": rnd,
    }
    if len(pool) == 0:
        return pd.DataFrame([row], columns=pool.columns if len(pool.columns) > 0 else None)
    return pd.concat([pool, pd.DataFrame([row])], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--max-minutes", type=float, default=30.0)
    p.add_argument("--per-round-minutes", type=float, default=5.0)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--optimize-trials", type=int, default=6)
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--step-bars", type=int, default=24 * 30)
    p.add_argument("--reward-clip", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--status-out", default="data/models/rl_self_train_status.txt")
    p.add_argument("--pool-out", default="data/models/rl_strategy_pool.csv")
    p.add_argument("--recheck-out", default="data/models/rl_strategy_recheck.csv")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--profiles", type=int, default=2)
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    py = (Path(".python") / "python.exe").resolve()
    models = Path("data/models")
    models.mkdir(parents=True, exist_ok=True)

    best_q = models / "rl_best_qtable.csv"
    best_policy = models / "rl_best_policy.csv"
    best_train = models / "rl_best_train_summary.txt"
    best_wf = models / "rl_best_walkforward.csv"
    best_wf_summary = models / "rl_best_walkforward_summary.txt"
    pool_path = Path(args.pool_out)

    start = time.time()
    hard_deadline = start + max(0.0, float(args.max_minutes)) * 60.0
    best_score = -10**9
    accepted = 0
    best_params = {
        "episodes": int(args.episodes),
        "alpha": 0.15,
        "gamma": 0.95,
        "eps_start": 0.25,
        "eps_end": 0.02,
    }

    # bootstrap from existing best summary if present
    if best_wf_summary.exists():
        prev = parse_kv_text(best_wf_summary)
        best_score = parse_float(prev.get("consistency_score", "-1e9"), -1e9)
    best_params = parse_best_params(best_train, best_params)
    pool = load_pool(pool_path)
    total_steps_done = 0
    total_steps_planned = 0

    for rnd in range(1, args.rounds + 1):
        elapsed_min = (time.time() - start) / 60.0
        if elapsed_min >= args.max_minutes or time.time() >= hard_deadline:
            break

        # Candidate set: current best + top pool strategies + local mutations.
        candidates = [dict(best_params)]
        if len(pool) > 0:
            top_pool = pool.sort_values("score", ascending=False).head(max(1, args.top_k))
            for _, r in top_pool.iterrows():
                candidates.append(
                    {
                        "episodes": int(r["episodes"]),
                        "alpha": float(r["alpha"]),
                        "gamma": float(r["gamma"]),
                        "eps_start": float(r["eps_start"]),
                        "eps_end": float(r["eps_end"]),
                    }
                )
        candidates.append(mutate_params(best_params))
        candidates.append(mutate_params(best_params))
        # De-duplicate candidate list by strategy id.
        dedup = {}
        for c in candidates:
            dedup[strategy_id(c)] = c
        candidates = list(dedup.values())

        profiles = [
            {"train_bars": args.train_bars, "test_bars": args.test_bars, "step_bars": args.step_bars, "max_minutes": args.per_round_minutes / max(1, args.profiles)},
        ]
        if args.profiles >= 2:
            profiles.append(
                {
                    "train_bars": max(1000, args.train_bars // 2),
                    "test_bars": max(300, args.test_bars // 2),
                    "step_bars": max(120, args.step_bars // 2),
                    "max_minutes": args.per_round_minutes / max(1, args.profiles),
                }
            )

        round_steps = len(candidates) * len(profiles)
        total_steps_planned += round_steps
        print(
            f"round {rnd}/{args.rounds} | candidates={len(candidates)} profiles={len(profiles)} "
            f"time_budget={args.per_round_minutes:.2f}m"
        )

        best_candidate = None
        best_candidate_tail = ""
        per_candidate = []
        hit_budget = False
        for cidx, cp in enumerate(candidates, start=1):
            if time.time() >= hard_deadline:
                hit_budget = True
                break
            sid = strategy_id(cp)
            print(f"  candidate {cidx}/{len(candidates)} | {sid}")
            score_vals = []
            ret_vals = []
            dd_vals = []
            pos_vals = []
            tail = ""
            ok = True
            for pidx, pf in enumerate(profiles, start=1):
                if time.time() >= hard_deadline:
                    hit_budget = True
                    ok = False
                    break
                ok_prof, tail = run_round_profile(py, base, args, cp, pf)
                total_steps_done += 1
                print(
                    "    "
                    + progress_bar(total_steps_done, max(total_steps_done, total_steps_planned))
                    + f" | profile {pidx}/{len(profiles)}"
                )
                if not ok_prof:
                    ok = False
                    break
                wf_summary = parse_kv_text(models / "rl_walkforward_summary.txt")
                score_vals.append(parse_float(wf_summary.get("consistency_score", "-1e9"), -1e9))
                ret_vals.append(parse_pct(wf_summary.get("mean_test_return", "0%"), 0.0))
                dd_vals.append(parse_pct(wf_summary.get("mean_test_dd", "0%"), -1.0))
                pos_vals.append(parse_pct(wf_summary.get("positive_test_return_rate", "0%"), 0.0))
            if not ok or len(score_vals) == 0:
                print("    result=failed")
                continue
            metrics = {
                "score": float(sum(score_vals) / len(score_vals)),
                "score_std": float(pd.Series(score_vals).std(ddof=0)),
                "mean_ret": float(sum(ret_vals) / len(ret_vals)),
                "mean_dd": float(sum(dd_vals) / len(dd_vals)),
                "pos_rate": float(sum(pos_vals) / len(pos_vals)),
            }
            score = metrics["score"]
            mean_ret = metrics["mean_ret"]
            mean_dd = metrics["mean_dd"]
            pos_rate = metrics["pos_rate"]
            improved = score > best_score
            sane = (pos_rate >= 0.50) and (mean_dd > -0.35)
            accept = improved and sane
            print(
                "    result="
                f"score={score:.4f} ret={mean_ret:.2%} dd={mean_dd:.2%} pos={pos_rate:.2%} "
                f"improved={improved} sane={sane} accept={accept}"
            )
            pool = upsert_pool_row(pool, cp, score, mean_ret, mean_dd, pos_rate, rnd, accept)
            per_candidate.append(
                {
                    "round": rnd,
                    "strategy_id": sid,
                    "score": score,
                    "score_std": metrics["score_std"],
                    "mean_test_return": mean_ret,
                    "mean_test_dd": mean_dd,
                    "positive_test_return_rate": pos_rate,
                    "improved": improved,
                    "sane": sane,
                    "accept": accept,
                }
            )
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "params": cp,
                    "score": score,
                    "mean_ret": mean_ret,
                    "mean_dd": mean_dd,
                    "pos_rate": pos_rate,
                    "improved": improved,
                    "sane": sane,
                    "accept": accept,
                    "sid": sid,
                }
                best_candidate_tail = tail

        if hit_budget:
            print("time budget reached during round, stopping candidate evaluation")

        if best_candidate is None:
            Path(args.status_out).write_text(
                f"round={rnd}\nstatus=error\nmessage=no candidate evaluated\n",
                encoding="ascii",
            )
            break

        score = best_candidate["score"]
        mean_ret = best_candidate["mean_ret"]
        mean_dd = best_candidate["mean_dd"]
        pos_rate = best_candidate["pos_rate"]
        improved = best_candidate["improved"]
        sane = best_candidate["sane"]
        accept = best_candidate["accept"]
        candidate_params = best_candidate["params"]
        tail = best_candidate_tail

        if accept:
            accepted += 1
            best_score = score
            copy_if_exists(models / "rl_qtable.csv", best_q)
            copy_if_exists(models / "rl_policy_latest.txt", best_policy)
            copy_if_exists(models / "rl_train_summary.txt", best_train)
            copy_if_exists(models / "rl_walkforward.csv", best_wf)
            copy_if_exists(models / "rl_walkforward_summary.txt", best_wf_summary)
            best_params = dict(candidate_params)
        else:
            # Roll back active policy to last best snapshot.
            copy_if_exists(best_q, models / "rl_qtable.csv")
            copy_if_exists(best_policy, models / "rl_policy_latest.txt")
            copy_if_exists(best_train, models / "rl_train_summary.txt")
            copy_if_exists(best_wf, models / "rl_walkforward.csv")
            copy_if_exists(best_wf_summary, models / "rl_walkforward_summary.txt")

        if len(pool) > 0:
            pool = pool.sort_values(["score", "accepted_count"], ascending=[False, False]).reset_index(drop=True)
            pool.to_csv(pool_path, index=False)

        # Explicitly re-check top strategies for consistency on current profile set.
        # This provides a direct "best strategies and consistency" table for the UI.
        recheck_df = pd.DataFrame(per_candidate)
        if len(recheck_df) > 0:
            recheck_df = recheck_df.sort_values(["score", "positive_test_return_rate"], ascending=[False, False]).reset_index(drop=True)
            top_n = recheck_df.head(max(1, args.top_k)).copy()
            top_n.to_csv(args.recheck_out, index=False)

        status = (
            f"round={rnd}\n"
            f"elapsed_minutes={(time.time() - start)/60.0:.2f}\n"
            f"accepted_rounds={accepted}\n"
            f"accept={accept}\n"
            f"improved={improved}\n"
            f"sane={sane}\n"
            f"score={score:.6f}\n"
            f"best_score={best_score:.6f}\n"
            f"positive_test_return_rate={pos_rate:.2%}\n"
            f"mean_test_return={mean_ret:.2%}\n"
            f"mean_test_dd={mean_dd:.2%}\n"
            f"candidate_episodes={candidate_params['episodes']}\n"
            f"candidate_alpha={candidate_params['alpha']:.4f}\n"
            f"candidate_gamma={candidate_params['gamma']:.4f}\n"
            f"candidate_eps_start={candidate_params['eps_start']:.4f}\n"
            f"candidate_eps_end={candidate_params['eps_end']:.4f}\n"
            f"best_episodes={best_params['episodes']}\n"
            f"best_alpha={best_params['alpha']:.4f}\n"
            f"best_gamma={best_params['gamma']:.4f}\n"
            f"best_eps_start={best_params['eps_start']:.4f}\n"
            f"best_eps_end={best_params['eps_end']:.4f}\n"
            f"candidate_strategy_id={best_candidate['sid']}\n"
            f"pool_size={len(pool)}\n"
            f"recheck_rows={len(per_candidate)}\n"
            f"run_tail={tail.replace(chr(10), ' | ')}\n"
        )
        Path(args.status_out).write_text(status, encoding="ascii")
        print(status)
        print("  " + progress_bar(rnd, args.rounds))

    print("self-train loop complete")


if __name__ == "__main__":
    main()
