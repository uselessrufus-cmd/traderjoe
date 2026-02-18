import argparse
import os
import hashlib
import math
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd


CHAMPION_REGIMES = ("global", "bull", "bear", "range")


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


def sid_to_key(sid: str) -> str:
    return hashlib.sha1(sid.encode("ascii", errors="ignore")).hexdigest()[:16]


def is_finite_number(v: float) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _as_float(v, default: float) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def has_strategy_id(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in {"", "nan", "none", "<na>"}


def parse_fee_schedule(text: str) -> list[float]:
    out = []
    raw = str(text or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            fee = float(part)
            out.append(max(0.0, fee))
        except Exception:
            continue
    return out


def resolve_worker_count(workers: int, cpu_target_pct: int) -> int:
    if int(workers) > 0:
        return max(1, int(workers))
    cores = int(os.cpu_count() or 1)
    pct = max(5, min(100, int(cpu_target_pct)))
    return max(1, int(math.ceil(cores * (pct / 100.0))))


def normalize_params(params: dict, fallback: dict | None = None) -> dict:
    fb = dict(
        fallback
        if fallback is not None
        else {
            "episodes": 8,
            "alpha": 0.15,
            "gamma": 0.95,
            "eps_start": 0.25,
            "eps_end": 0.02,
        }
    )
    out = {}
    out["episodes"] = max(1, int(_as_float(params.get("episodes", fb["episodes"]), float(fb["episodes"]))))
    out["alpha"] = min(0.30, max(0.03, _as_float(params.get("alpha", fb["alpha"]), float(fb["alpha"]))))
    out["gamma"] = min(0.995, max(0.85, _as_float(params.get("gamma", fb["gamma"]), float(fb["gamma"]))))
    out["eps_start"] = min(0.50, max(0.05, _as_float(params.get("eps_start", fb["eps_start"]), float(fb["eps_start"]))))
    out["eps_end"] = min(0.10, max(0.005, _as_float(params.get("eps_end", fb["eps_end"]), float(fb["eps_end"]))))
    if out["eps_end"] > out["eps_start"]:
        out["eps_end"] = max(0.005, out["eps_start"] * 0.5)
    return out


def is_valid_param_row(row: pd.Series) -> bool:
    p = normalize_params(
        {
            "episodes": row.get("episodes", 0),
            "alpha": row.get("alpha", 0),
            "gamma": row.get("gamma", 0),
            "eps_start": row.get("eps_start", 0),
            "eps_end": row.get("eps_end", 0),
        }
    )
    sid = f"ep={int(row.get('episodes', 0))}|a={_as_float(row.get('alpha', 0), 0.0):.4f}|g={_as_float(row.get('gamma', 0), 0.0):.4f}|es={_as_float(row.get('eps_start', 0), 0.0):.4f}|ee={_as_float(row.get('eps_end', 0), 0.0):.4f}"
    if sid == "ep=0|a=0.0000|g=0.0000|es=0.0000|ee=0.0000":
        return False
    return (
        int(p["episodes"]) >= 1
        and float(p["alpha"]) > 0
        and float(p["gamma"]) > 0
        and float(p["eps_start"]) > 0
        and float(p["eps_end"]) > 0
        and float(p["eps_end"]) <= float(p["eps_start"])
    )


def copy_active_bundle(models: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "rl_qtable.csv": "qtable.csv",
        "rl_policy_latest.txt": "policy.csv",
        "rl_train_summary.txt": "train_summary.txt",
        "rl_test_equity.csv": "test_equity.csv",
        "rl_walkforward.csv": "walkforward.csv",
        "rl_walkforward_summary.txt": "walkforward_summary.txt",
    }
    for src_name, dst_name in files.items():
        copy_if_exists(models / src_name, dst_dir / dst_name)


def copy_bundle_to_active(models: Path, src_dir: Path):
    files = {
        "qtable.csv": "rl_qtable.csv",
        "policy.csv": "rl_policy_latest.txt",
        "train_summary.txt": "rl_train_summary.txt",
        "test_equity.csv": "rl_test_equity.csv",
        "walkforward.csv": "rl_walkforward.csv",
        "walkforward_summary.txt": "rl_walkforward_summary.txt",
    }
    for src_name, dst_name in files.items():
        copy_if_exists(src_dir / src_name, models / dst_name)


def build_profile_outputs(profile_dir: Path) -> dict[str, Path]:
    return {
        "q_out": profile_dir / "qtable.csv",
        "summary_out": profile_dir / "train_summary.txt",
        "policy_out": profile_dir / "policy.csv",
        "equity_out": profile_dir / "test_equity.csv",
        "wf_out": profile_dir / "walkforward.csv",
        "wf_summary_out": profile_dir / "walkforward_summary.txt",
        "rationale_memory": profile_dir / "rationale_memory.csv",
        "rationale_out": profile_dir / "rationale_assessment.csv",
        "rationale_summary_out": profile_dir / "rationale_summary.txt",
    }


def copy_profile_bundle(profile_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(profile_dir / "qtable.csv", dst_dir / "qtable.csv")
    copy_if_exists(profile_dir / "policy.csv", dst_dir / "policy.csv")
    copy_if_exists(profile_dir / "train_summary.txt", dst_dir / "train_summary.txt")
    copy_if_exists(profile_dir / "test_equity.csv", dst_dir / "test_equity.csv")
    copy_if_exists(profile_dir / "walkforward.csv", dst_dir / "walkforward.csv")
    copy_if_exists(profile_dir / "walkforward_summary.txt", dst_dir / "walkforward_summary.txt")


def split_main_holdout_windows(wf_df: pd.DataFrame, holdout_bars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(wf_df) == 0 or int(holdout_bars) <= 0 or "test_end" not in wf_df.columns:
        return wf_df.copy(), pd.DataFrame(columns=wf_df.columns)
    max_end = int(pd.to_numeric(wf_df["test_end"], errors="coerce").max())
    cutoff = max_end - int(holdout_bars)
    holdout = wf_df[pd.to_numeric(wf_df["test_end"], errors="coerce") > cutoff].copy()
    main = wf_df[pd.to_numeric(wf_df["test_end"], errors="coerce") <= cutoff].copy()
    if len(main) == 0:
        return wf_df.copy(), pd.DataFrame(columns=wf_df.columns)
    return main, holdout


def summarize_windows(wf_df: pd.DataFrame) -> dict:
    if len(wf_df) == 0:
        return {"score": -1e9, "mean_ret": 0.0, "mean_dd": 0.0, "pos_rate": 0.0, "windows": 0, "recent_objectives": []}
    obj = pd.to_numeric(wf_df["objective"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(wf_df["test_return"], errors="coerce").fillna(0.0)
    dd = pd.to_numeric(wf_df["test_dd"], errors="coerce").fillna(0.0)
    score = float(obj.mean()) - float(obj.std(ddof=0))
    return {
        "score": score,
        "mean_ret": float(ret.mean()),
        "mean_dd": float(dd.mean()),
        "pos_rate": float((ret > 0).mean()),
        "windows": int(len(wf_df)),
        "recent_objectives": [float(x) for x in obj.tolist()],
    }


def load_champion_window_metrics(wf_path: Path, holdout_bars: int) -> dict:
    if not wf_path.exists():
        return {
            "main_score": -1e9,
            "main_windows": 0,
            "main_recent_objectives": [],
            "holdout_score": -1e9,
            "holdout_windows": 0,
        }
    try:
        wf = pd.read_csv(wf_path)
    except Exception:
        return {
            "main_score": -1e9,
            "main_windows": 0,
            "main_recent_objectives": [],
            "holdout_score": -1e9,
            "holdout_windows": 0,
        }
    main_df, holdout_df = split_main_holdout_windows(wf, holdout_bars)
    main_stats = summarize_windows(main_df)
    holdout_stats = summarize_windows(holdout_df)
    return {
        "main_score": float(main_stats["score"]),
        "main_windows": int(main_stats["windows"]),
        "main_recent_objectives": list(main_stats["recent_objectives"]),
        "holdout_score": float(holdout_stats["score"]),
        "holdout_windows": int(holdout_stats["windows"]),
    }


def run_holdout_probe(
    py: Path,
    base: Path,
    args,
    params: dict,
    fee_rate: float | None,
    candidate_eval_dir: Path,
    exp_parent_run: str,
    exp_tag_prefix: str,
    exp_note: str,
) -> tuple[bool, dict, str]:
    try:
        src = Path(args.input)
        if not src.exists():
            return False, {}, "holdout probe: input file not found"
        raw = pd.read_csv(src)
        if len(raw) < 200:
            return False, {}, "holdout probe: input too small"

        holdout_bars = max(300, int(args.holdout_bars))
        holdout_train = max(180, int(holdout_bars * 0.50))
        holdout_test = max(90, int(holdout_bars * 0.25))
        holdout_step = max(60, int(holdout_bars * 0.10))
        min_rows = holdout_train + int(args.purge_bars) + holdout_test + holdout_step * max(1, int(args.holdout_min_windows) - 1)
        use_rows = max(holdout_bars, min_rows)
        tail = raw.tail(use_rows).copy()
        if len(tail) < (holdout_train + int(args.purge_bars) + holdout_test):
            return False, {}, "holdout probe: insufficient rows after slicing"

        hold_dir = candidate_eval_dir / "holdout_probe"
        hold_dir.mkdir(parents=True, exist_ok=True)
        hold_input = hold_dir / "holdout_input.csv"
        tail.to_csv(hold_input, index=False)

        ok, tail_out = run_round_profile(
            py,
            base,
            args,
            params,
            {
                "train_bars": holdout_train,
                "test_bars": holdout_test,
                "step_bars": holdout_step,
                "max_minutes": max(0.5, min(2.0, float(args.per_round_minutes) * 0.6)),
            },
            optimize_trials_override=1,
            time_scale=1.0,
            exp_source="rl_self_train_holdout",
            exp_parent_run=exp_parent_run,
            exp_tag=(f"{exp_tag_prefix}|holdout_probe" if exp_tag_prefix else "holdout_probe"),
            exp_note=exp_note,
            fee_rate=fee_rate,
            output_dir=hold_dir,
            input_path=hold_input,
        )
        if not ok:
            return False, {}, tail_out

        wf_path = hold_dir / "walkforward.csv"
        if not wf_path.exists():
            return False, {}, "holdout probe: walkforward output missing"
        wf = pd.read_csv(wf_path)
        stats = summarize_windows(wf)
        return True, stats, tail_out
    except Exception as exc:
        return False, {}, f"holdout probe exception: {exc}"


def parse_best_params(summary_path: Path, fallback: dict) -> dict:
    if not summary_path.exists():
        return normalize_params(dict(fallback), fallback=fallback)
    txt = summary_path.read_text(encoding="ascii", errors="ignore")
    m = re.search(
        r"best episodes=(?P<episodes>\d+)\s+alpha=(?P<alpha>[-0-9.]+)\s+gamma=(?P<gamma>[-0-9.]+)\s+eps_start=(?P<eps_start>[-0-9.]+)\s+eps_end=(?P<eps_end>[-0-9.]+)",
        txt,
    )
    if not m:
        return normalize_params(dict(fallback), fallback=fallback)
    out = dict(fallback)
    out["episodes"] = int(m.group("episodes"))
    out["alpha"] = float(m.group("alpha"))
    out["gamma"] = float(m.group("gamma"))
    out["eps_start"] = float(m.group("eps_start"))
    out["eps_end"] = float(m.group("eps_end"))
    return normalize_params(out, fallback=fallback)


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


def mutate_refine(base: dict) -> dict:
    p = dict(base)
    p["alpha"] = min(0.30, max(0.03, p["alpha"] * random.uniform(0.95, 1.05)))
    p["gamma"] = min(0.995, max(0.85, p["gamma"] + random.uniform(-0.01, 0.01)))
    p["eps_start"] = min(0.50, max(0.05, p["eps_start"] + random.uniform(-0.03, 0.03)))
    p["eps_end"] = min(0.10, max(0.005, p["eps_end"] + random.uniform(-0.01, 0.01)))
    if p["eps_end"] > p["eps_start"]:
        p["eps_end"] = max(0.005, p["eps_start"] * 0.5)
    eps = max(1, int(base["episodes"]))
    p["episodes"] = max(1, int(random.choice([eps, max(1, eps - 1), eps + 1])))
    return p


def mutate_explore(base: dict) -> dict:
    p = dict(base)
    p["alpha"] = float(random.choice([0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
    p["gamma"] = float(random.choice([0.85, 0.9, 0.93, 0.95, 0.97, 0.99]))
    p["eps_start"] = float(random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    p["eps_end"] = float(random.choice([0.005, 0.01, 0.02, 0.05, 0.1]))
    if p["eps_end"] > p["eps_start"]:
        p["eps_end"] = max(0.005, p["eps_start"] * 0.5)
    eps = max(1, int(base["episodes"]))
    p["episodes"] = max(1, int(random.choice([1, 2, 4, 6, 8, eps, eps + 2])))
    return p


def run_round(py: Path, base: Path, args, params: dict, fee_rate: float | None = None) -> tuple[bool, str]:
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
        fee_rate=fee_rate,
    )


def run_round_profile(
    py: Path,
    base: Path,
    args,
    params: dict,
    profile: dict,
    optimize_trials_override: int | None = None,
    time_scale: float = 1.0,
    exp_source: str = "rl_self_train",
    exp_parent_run: str = "",
    exp_tag: str = "",
    exp_note: str = "",
    fee_rate: float | None = None,
    output_dir: Path | None = None,
    input_path: Path | None = None,
) -> tuple[bool, str]:
    effective_fee_rate = float(args.fee_rate) if fee_rate is None else float(fee_rate)
    run_dir = output_dir if output_dir is not None else Path("data/models")
    run_dir.mkdir(parents=True, exist_ok=True)
    outs = build_profile_outputs(run_dir)
    # Seed local rationale memory from global memory so parallel runs start from the same prior.
    global_rationale = Path("data/models/rl_rationale_memory.csv")
    if output_dir is not None and not outs["rationale_memory"].exists() and global_rationale.exists():
        copy_if_exists(global_rationale, outs["rationale_memory"])
    cmd = [
        str(py),
        str(base / "rl_train_historical.py"),
        "--walkforward",
        "--input",
        str(input_path if input_path is not None else args.input),
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
        str(optimize_trials_override if optimize_trials_override is not None else args.optimize_trials),
        "--train-bars",
        str(profile["train_bars"]),
        "--test-bars",
        str(profile["test_bars"]),
        "--step-bars",
        str(profile["step_bars"]),
        "--purge-bars",
        str(args.purge_bars),
        "--max-minutes",
        str(profile.get("max_minutes", args.per_round_minutes) * max(0.2, float(time_scale))),
        "--reward-clip",
        str(args.reward_clip),
        "--fee-rate",
        str(effective_fee_rate),
        "--q-out",
        str(outs["q_out"]),
        "--summary-out",
        str(outs["summary_out"]),
        "--policy-out",
        str(outs["policy_out"]),
        "--equity-out",
        str(outs["equity_out"]),
        "--wf-out",
        str(outs["wf_out"]),
        "--wf-summary-out",
        str(outs["wf_summary_out"]),
        "--rationale-memory",
        str(outs["rationale_memory"]),
        "--rationale-out",
        str(outs["rationale_out"]),
        "--rationale-summary-out",
        str(outs["rationale_summary_out"]),
        "--exp-source",
        str(exp_source),
    ]
    if exp_parent_run:
        cmd.extend(["--exp-parent-run", str(exp_parent_run)])
    if exp_tag:
        cmd.extend(["--exp-tag", str(exp_tag)])
    if exp_note:
        cmd.extend(["--exp-note", str(exp_note)])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, proc.stdout[-2000:]


def evaluate_params(
    py: Path,
    base: Path,
    args,
    params: dict,
    profiles: list[dict],
    mode: str = "refine",
    deadline: float | None = None,
    exp_parent_run: str = "",
    exp_tag_prefix: str = "",
    exp_note: str = "",
    fee_rate: float | None = None,
    candidate_eval_dir: Path | None = None,
) -> tuple[bool, dict, str, Path | None]:
    score_vals = []
    ret_vals = []
    dd_vals = []
    pos_vals = []
    holdout_score_vals = []
    holdout_ret_vals = []
    holdout_dd_vals = []
    holdout_pos_vals = []
    holdout_windows_vals = []
    reg_acc: dict[str, dict] = {}
    tail = ""
    selected_profile_dir: Path | None = None
    selected_profile_score = -1e18
    selected_recent_objectives: list[float] = []
    selected_holdout_score = -1e9
    selected_holdout_windows = 0
    optimize_trials_override = None
    time_scale = 1.0
    if mode == "explore":
        optimize_trials_override = max(1, int(args.optimize_trials) // 2)
        time_scale = 0.7
    elif mode == "champion_refine":
        optimize_trials_override = max(1, int(args.optimize_trials))
        time_scale = 1.1
    sid = strategy_id(params)
    for pidx, pf in enumerate(profiles, start=1):
        if deadline is not None and time.time() >= deadline:
            break
        pf_eff = dict(pf)
        if deadline is not None:
            remain_minutes = max(0.0, (deadline - time.time()) / 60.0)
            pf_eff["max_minutes"] = min(float(pf_eff.get("max_minutes", args.per_round_minutes)), remain_minutes)
            if pf_eff["max_minutes"] <= 0:
                break
        profile_dir = None
        if candidate_eval_dir is not None:
            profile_dir = candidate_eval_dir / f"profile_{pidx}"
        ok, tail = run_round_profile(
            py,
            base,
            args,
            params,
            pf_eff,
            optimize_trials_override=optimize_trials_override,
            time_scale=time_scale,
            exp_source="rl_self_train",
            exp_parent_run=exp_parent_run,
            exp_tag=(
                f"{exp_tag_prefix}|{mode}|{sid}|profile={pidx}"
                if exp_tag_prefix
                else f"{mode}|{sid}|profile={pidx}"
            ),
            exp_note=exp_note,
            fee_rate=fee_rate,
            output_dir=profile_dir,
        )
        if not ok:
            return False, {}, tail, None
        wf_path = (profile_dir / "walkforward.csv") if profile_dir is not None else Path("data/models/rl_walkforward.csv")
        profile_score = -1e9
        profile_recent: list[float] = []
        profile_holdout_score = -1e9
        profile_holdout_windows = 0
        if wf_path.exists():
            try:
                wf_df = pd.read_csv(wf_path)
                if len(wf_df) > 0:
                    main_df, holdout_df = split_main_holdout_windows(wf_df, int(args.holdout_bars))
                    main_stats = summarize_windows(main_df)
                    holdout_stats = summarize_windows(holdout_df)
                    profile_score = float(main_stats["score"])
                    profile_recent = list(main_stats["recent_objectives"])
                    profile_holdout_score = float(holdout_stats["score"])
                    profile_holdout_windows = int(holdout_stats["windows"])

                    score_vals.append(profile_score)
                    ret_vals.append(float(main_stats["mean_ret"]))
                    dd_vals.append(float(main_stats["mean_dd"]))
                    pos_vals.append(float(main_stats["pos_rate"]))
                    holdout_score_vals.append(float(holdout_stats["score"]))
                    holdout_ret_vals.append(float(holdout_stats["mean_ret"]))
                    holdout_dd_vals.append(float(holdout_stats["mean_dd"]))
                    holdout_pos_vals.append(float(holdout_stats["pos_rate"]))
                    holdout_windows_vals.append(int(holdout_stats["windows"]))

                if "regime" in wf_df.columns and len(wf_df) > 0:
                    regime_df = main_df if len(main_df) > 0 else wf_df
                    for reg, rdf in regime_df.groupby("regime"):
                        windows = int(len(rdf))
                        if windows <= 0:
                            continue
                        r_score = float(rdf["objective"].mean()) - float(rdf["objective"].std(ddof=0) if windows > 1 else 0.0)
                        r_ret = float(rdf["test_return"].mean())
                        r_dd = float(rdf["test_dd"].mean())
                        r_pos = float((rdf["test_return"] > 0).mean())
                        bucket = reg_acc.setdefault(
                            str(reg),
                            {"score_sum": 0.0, "ret_sum": 0.0, "dd_sum": 0.0, "pos_sum": 0.0, "windows": 0},
                        )
                        bucket["score_sum"] += r_score * windows
                        bucket["ret_sum"] += r_ret * windows
                        bucket["dd_sum"] += r_dd * windows
                        bucket["pos_sum"] += r_pos * windows
                        bucket["windows"] += windows
            except Exception:
                pass
        if profile_score <= -1e8:
            wf_summary_path = (profile_dir / "walkforward_summary.txt") if profile_dir is not None else Path("data/models/rl_walkforward_summary.txt")
            wf_summary = parse_kv_text(wf_summary_path)
            profile_score = parse_float(wf_summary.get("consistency_score", "-1e9"), -1e9)
            score_vals.append(profile_score)
            ret_vals.append(parse_pct(wf_summary.get("mean_test_return", "0%"), 0.0))
            dd_vals.append(parse_pct(wf_summary.get("mean_test_dd", "0%"), -1.0))
            pos_vals.append(parse_pct(wf_summary.get("positive_test_return_rate", "0%"), 0.0))
            holdout_score_vals.append(-1e9)
            holdout_ret_vals.append(0.0)
            holdout_dd_vals.append(0.0)
            holdout_pos_vals.append(0.0)
            holdout_windows_vals.append(0)

        if profile_dir is not None and profile_score >= selected_profile_score:
            selected_profile_score = profile_score
            selected_profile_dir = profile_dir
            selected_recent_objectives = profile_recent
            selected_holdout_score = profile_holdout_score
            selected_holdout_windows = profile_holdout_windows

    if (
        int(args.holdout_bars) > 0
        and int(args.holdout_min_windows) > 0
        and selected_holdout_windows < int(args.holdout_min_windows)
        and candidate_eval_dir is not None
    ):
        probe_ok, probe_stats, probe_tail = run_holdout_probe(
            py,
            base,
            args,
            params,
            fee_rate,
            candidate_eval_dir,
            exp_parent_run,
            exp_tag_prefix,
            exp_note,
        )
        if probe_ok:
            selected_holdout_score = float(probe_stats.get("score", -1e9))
            selected_holdout_windows = int(probe_stats.get("windows", 0))
        elif len(tail) == 0:
            tail = probe_tail
    if len(score_vals) == 0:
        return False, {}, tail, None
    regime_metrics = {}
    for reg, acc in reg_acc.items():
        w = max(1, int(acc["windows"]))
        regime_metrics[reg] = {
            "score": float(acc["score_sum"] / w),
            "mean_ret": float(acc["ret_sum"] / w),
            "mean_dd": float(acc["dd_sum"] / w),
            "pos_rate": float(acc["pos_sum"] / w),
            "windows": int(acc["windows"]),
        }
    out = {
        "score": float(sum(score_vals) / len(score_vals)),
        "score_std": float(pd.Series(score_vals).std(ddof=0)),
        "mean_ret": float(sum(ret_vals) / len(ret_vals)),
        "mean_dd": float(sum(dd_vals) / len(dd_vals)),
        "pos_rate": float(sum(pos_vals) / len(pos_vals)),
        "holdout_score": float(sum(holdout_score_vals) / len(holdout_score_vals)) if len(holdout_score_vals) > 0 else -1e9,
        "holdout_mean_ret": float(sum(holdout_ret_vals) / len(holdout_ret_vals)) if len(holdout_ret_vals) > 0 else 0.0,
        "holdout_mean_dd": float(sum(holdout_dd_vals) / len(holdout_dd_vals)) if len(holdout_dd_vals) > 0 else 0.0,
        "holdout_pos_rate": float(sum(holdout_pos_vals) / len(holdout_pos_vals)) if len(holdout_pos_vals) > 0 else 0.0,
        "holdout_windows": int(max(holdout_windows_vals)) if len(holdout_windows_vals) > 0 else 0,
        "recent_objectives": selected_recent_objectives,
        "selected_holdout_score": float(selected_holdout_score),
        "selected_holdout_windows": int(selected_holdout_windows),
        "regime_metrics": regime_metrics,
    }
    bundle_dir = None
    if candidate_eval_dir is not None and selected_profile_dir is not None:
        bundle_dir = candidate_eval_dir / "bundle"
        copy_profile_bundle(selected_profile_dir, bundle_dir)
    return True, out, tail, bundle_dir


def load_pool(path: Path) -> pd.DataFrame:
    cols = [
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
    if path.exists():
        try:
            df = pd.read_csv(path)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0
            if len(df) > 0:
                df = df[df.apply(is_valid_param_row, axis=1)].copy()
            return df[cols].reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(
        columns=cols
    )


def load_champions(path: Path) -> pd.DataFrame:
    cols = [
        "regime",
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
        "windows",
        "eval_count",
        "accepted_count",
        "last_round",
    ]
    if path.exists():
        try:
            df = pd.read_csv(path)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0
            return df[cols]
        except Exception:
            pass
    rows = []
    for reg in CHAMPION_REGIMES:
        rows.append(
            {
                "regime": reg,
                "strategy_id": "",
                "episodes": 1,
                "alpha": 0.15,
                "gamma": 0.95,
                "eps_start": 0.25,
                "eps_end": 0.02,
                "score": -1e9,
                "mean_test_return": 0.0,
                "mean_test_dd": -1.0,
                "positive_test_return_rate": 0.0,
                "windows": 0,
                "eval_count": 0,
                "accepted_count": 0,
                "last_round": 0,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def champion_row(champions: pd.DataFrame, regime: str) -> pd.Series:
    rows = champions[champions["regime"] == regime]
    if len(rows) == 0:
        return pd.Series(
            {
                "regime": regime,
                "strategy_id": "",
                "episodes": 1,
                "alpha": 0.15,
                "gamma": 0.95,
                "eps_start": 0.25,
                "eps_end": 0.02,
                "score": -1e9,
                "mean_test_return": 0.0,
                "mean_test_dd": -1.0,
                "positive_test_return_rate": 0.0,
                "windows": 0,
                "eval_count": 0,
                "accepted_count": 0,
                "last_round": 0,
            }
        )
    return rows.iloc[0]


def upsert_champion(
    champions: pd.DataFrame,
    regime: str,
    params: dict,
    sid: str,
    score: float,
    mean_ret: float,
    mean_dd: float,
    pos_rate: float,
    windows: int,
    rnd: int,
):
    mask = champions["regime"] == regime
    if mask.any():
        idx = champions.index[mask][0]
        champions.at[idx, "strategy_id"] = sid
        champions.at[idx, "episodes"] = int(params["episodes"])
        champions.at[idx, "alpha"] = float(params["alpha"])
        champions.at[idx, "gamma"] = float(params["gamma"])
        champions.at[idx, "eps_start"] = float(params["eps_start"])
        champions.at[idx, "eps_end"] = float(params["eps_end"])
        champions.at[idx, "score"] = float(score)
        champions.at[idx, "mean_test_return"] = float(mean_ret)
        champions.at[idx, "mean_test_dd"] = float(mean_dd)
        champions.at[idx, "positive_test_return_rate"] = float(pos_rate)
        champions.at[idx, "windows"] = int(windows)
        champions.at[idx, "eval_count"] = int(champions.at[idx, "eval_count"]) + 1
        champions.at[idx, "accepted_count"] = int(champions.at[idx, "accepted_count"]) + 1
        champions.at[idx, "last_round"] = int(rnd)
        return champions
    row = {
        "regime": regime,
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
        "windows": int(windows),
        "eval_count": 1,
        "accepted_count": 1,
        "last_round": int(rnd),
    }
    return pd.concat([champions, pd.DataFrame([row])], ignore_index=True)


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
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--optimize-trials", type=int, default=6)
    p.add_argument("--train-bars", type=int, default=24 * 365)
    p.add_argument("--test-bars", type=int, default=24 * 90)
    p.add_argument("--step-bars", type=int, default=24 * 30)
    p.add_argument("--purge-bars", type=int, default=12)
    p.add_argument("--reward-clip", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--fee-schedule", default="")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu-target-pct", type=int, default=50)
    p.add_argument("--status-out", default="data/models/rl_self_train_status.txt")
    p.add_argument("--pool-out", default="data/models/rl_strategy_pool.csv")
    p.add_argument("--recheck-out", default="data/models/rl_strategy_recheck.csv")
    p.add_argument("--champions-out", default="data/models/rl_champions.csv")
    p.add_argument("--champions-summary-out", default="data/models/rl_champions_summary.txt")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--profiles", type=int, default=2)
    p.add_argument("--refine-candidates", type=int, default=3)
    p.add_argument("--explore-candidates", type=int, default=3)
    p.add_argument("--global-min-improve", type=float, default=0.02)
    p.add_argument("--global-min-windows", type=int, default=1)
    p.add_argument("--regime-min-windows", "--min-regime-windows", dest="regime_min_windows", type=int, default=1)
    p.add_argument("--regime-min-improve", type=float, default=0.01)
    p.add_argument("--holdout-bars", type=int, default=24 * 90)
    p.add_argument("--holdout-min-windows", type=int, default=2)
    p.add_argument("--holdout-min-improve", type=float, default=0.0)
    p.add_argument("--promote-recent-windows", type=int, default=8)
    p.add_argument("--promote-recent-min-wins", type=int, default=5)
    p.add_argument("--promote-recent-min-improve", type=float, default=0.0)
    p.add_argument("--promote-min-fee", type=float, default=-1.0)
    p.add_argument("--exp-tag", default="")
    p.add_argument("--exp-note", default="")
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    py = (Path(".python") / "python.exe").resolve()
    models = Path("data/models")
    models.mkdir(parents=True, exist_ok=True)

    pool_path = Path(args.pool_out)
    champions_path = Path(args.champions_out)
    champions_summary_path = Path(args.champions_summary_out)
    status_path = Path(args.status_out)
    recheck_path = Path(args.recheck_out)

    artifact_dir = champions_path.parent if str(champions_path.parent) != "." else models
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    champions_path.parent.mkdir(parents=True, exist_ok=True)
    champions_summary_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    recheck_path.parent.mkdir(parents=True, exist_ok=True)

    best_q = artifact_dir / "rl_best_qtable.csv"
    best_policy = artifact_dir / "rl_best_policy.csv"
    best_train = artifact_dir / "rl_best_train_summary.txt"
    best_wf = artifact_dir / "rl_best_walkforward.csv"
    best_wf_summary = artifact_dir / "rl_best_walkforward_summary.txt"
    champions_dir = artifact_dir / "champions"
    temp_snapshots_dir = artifact_dir / "_candidate_snapshots"

    start = time.time()
    self_train_run_id = f"selftrain_{int(start)}"
    max_minutes = float(args.max_minutes)
    hard_deadline = None if max_minutes <= 0 else (start + max_minutes * 60.0)
    fee_schedule = parse_fee_schedule(args.fee_schedule)
    if len(fee_schedule) == 0:
        fee_schedule = [float(args.fee_rate)]
    worker_count = resolve_worker_count(args.workers, args.cpu_target_pct)
    print(
        f"self-train setup | workers={worker_count} "
        f"(cores={int(os.cpu_count() or 1)} target={int(args.cpu_target_pct)}%)"
    )
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
    champions = load_champions(champions_path)
    g_row = champion_row(champions, "global")
    best_score = parse_float(str(g_row.get("score", -1e9)), -1e9)
    if best_score <= -1e8 and best_wf_summary.exists():
        prev = parse_kv_text(best_wf_summary)
        best_score = parse_float(prev.get("consistency_score", "-1e9"), -1e9)
    if has_strategy_id(g_row.get("strategy_id", "")):
        best_params = normalize_params(
            {
            "episodes": int(g_row["episodes"]),
            "alpha": float(g_row["alpha"]),
            "gamma": float(g_row["gamma"]),
            "eps_start": float(g_row["eps_start"]),
            "eps_end": float(g_row["eps_end"]),
            },
            fallback=best_params,
        )
    else:
        best_params = parse_best_params(best_train, best_params)
    best_params = normalize_params(best_params, fallback=best_params)
    pool = load_pool(pool_path)
    total_steps_done = 0
    total_steps_planned = 0

    for rnd in range(1, args.rounds + 1):
        elapsed_min = (time.time() - start) / 60.0
        if hard_deadline is not None and (elapsed_min >= max_minutes or time.time() >= hard_deadline):
            break
        round_fee_rate = float(fee_schedule[min(rnd - 1, len(fee_schedule) - 1)])
        promote_min_fee = float(args.promote_min_fee)
        promote_fee_pass = (promote_min_fee < 0.0) or (round_fee_rate >= promote_min_fee)

        # Two-program workflow:
        # - explore: generate new ideas quickly
        # - refine/champion_refine: improve proven ideas for promotion
        candidates = []
        candidates.append({"params": normalize_params(dict(best_params), fallback=best_params), "mode": "champion_refine", "source": "best"})

        for reg in CHAMPION_REGIMES:
            cr = champion_row(champions, reg)
            if has_strategy_id(cr.get("strategy_id", "")):
                cp = normalize_params(
                    {
                    "episodes": int(cr["episodes"]),
                    "alpha": float(cr["alpha"]),
                    "gamma": float(cr["gamma"]),
                    "eps_start": float(cr["eps_start"]),
                    "eps_end": float(cr["eps_end"]),
                    },
                    fallback=best_params,
                )
                candidates.append({"params": cp, "mode": "champion_refine", "source": f"champion:{reg}"})

        if len(pool) > 0:
            top_pool = pool.sort_values("score", ascending=False).head(max(1, args.top_k))
            for _, r in top_pool.iterrows():
                cp = normalize_params(
                    {
                    "episodes": int(r["episodes"]),
                    "alpha": float(r["alpha"]),
                    "gamma": float(r["gamma"]),
                    "eps_start": float(r["eps_start"]),
                    "eps_end": float(r["eps_end"]),
                    },
                    fallback=best_params,
                )
                candidates.append({"params": cp, "mode": "refine", "source": "pool"})

        for _ in range(max(0, int(args.refine_candidates))):
            candidates.append({"params": mutate_refine(best_params), "mode": "refine", "source": "mutate_refine"})
        for _ in range(max(0, int(args.explore_candidates))):
            candidates.append({"params": mutate_explore(best_params), "mode": "explore", "source": "mutate_explore"})

        # De-duplicate candidate list by strategy id, keeping the highest-priority mode.
        mode_priority = {"champion_refine": 3, "refine": 2, "pool": 2, "explore": 1}
        dedup = {}
        for c in candidates:
            sid = strategy_id(c["params"])
            prev = dedup.get(sid)
            if prev is None or mode_priority.get(c["mode"], 0) > mode_priority.get(prev["mode"], 0):
                dedup[sid] = c
        candidates = list(dedup.values())

        global_champ_metrics = load_champion_window_metrics(champions_dir / "global" / "walkforward.csv", int(args.holdout_bars))
        if global_champ_metrics["main_windows"] == 0 and best_wf.exists():
            global_champ_metrics = load_champion_window_metrics(best_wf, int(args.holdout_bars))

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

        scheduled_candidates = []
        hit_budget = False
        for cidx, cobj in enumerate(candidates, start=1):
            if hard_deadline is not None and time.time() >= hard_deadline:
                hit_budget = True
                break
            scheduled_candidates.append((cidx, cobj))

        round_steps = len(scheduled_candidates) * len(profiles)
        total_steps_planned += round_steps
        active_workers = max(1, min(int(worker_count), max(1, len(scheduled_candidates))))
        print(
            f"round {rnd}/{args.rounds} | candidates={len(scheduled_candidates)} profiles={len(profiles)} "
            f"time_budget={args.per_round_minutes:.2f}m fee_rate={round_fee_rate:.6f} workers={active_workers}"
        )
        if len(scheduled_candidates) == 0:
            print("time budget reached before scheduling candidates for this round")
            break

        best_candidate = None
        best_candidate_tail = ""
        per_candidate = []
        snapshot_cache: dict[str, Path] = {}
        if temp_snapshots_dir.exists():
            shutil.rmtree(temp_snapshots_dir, ignore_errors=True)
        temp_snapshots_dir.mkdir(parents=True, exist_ok=True)
        futures = {}
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            for cidx, cobj in scheduled_candidates:
                cp = cobj["params"]
                mode = cobj.get("mode", "refine")
                source = cobj.get("source", "unknown")
                sid = strategy_id(cp)
                print(f"  candidate {cidx}/{len(scheduled_candidates)} | {sid} | mode={mode} source={source}")
                cand_eval_dir = temp_snapshots_dir / f"round_{rnd}" / f"cand_{cidx}_{sid_to_key(sid)}"
                fut = executor.submit(
                    evaluate_params,
                    py,
                    base,
                    args,
                    cp,
                    profiles,
                    mode,
                    hard_deadline,
                    self_train_run_id,
                    args.exp_tag,
                    args.exp_note,
                    round_fee_rate,
                    cand_eval_dir,
                )
                futures[fut] = (cidx, cobj, sid)

            for fut in as_completed(futures):
                cidx, cobj, sid = futures[fut]
                cp = cobj["params"]
                mode = cobj.get("mode", "refine")
                source = cobj.get("source", "unknown")
                try:
                    ok, metrics, tail, bundle_dir = fut.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    ok, metrics, tail, bundle_dir = False, {}, f"worker exception: {exc}", None

                total_steps_done += len(profiles)
                print(
                    "    "
                    + progress_bar(total_steps_done, max(total_steps_done, total_steps_planned))
                    + f" | candidate {cidx}/{len(scheduled_candidates)}"
                )
                if not ok:
                    print(f"    result=failed ({tail[:200]})")
                    continue
                score = metrics["score"]
                mean_ret = metrics["mean_ret"]
                mean_dd = metrics["mean_dd"]
                pos_rate = metrics["pos_rate"]
                holdout_score = float(metrics.get("selected_holdout_score", metrics.get("holdout_score", -1e9)))
                holdout_windows = int(metrics.get("selected_holdout_windows", metrics.get("holdout_windows", 0)))
                cand_recent = list(metrics.get("recent_objectives", []))
                global_windows = int(sum(int(v.get("windows", 0)) for v in metrics.get("regime_metrics", {}).values()))
                improved = score > (best_score + float(args.global_min_improve))
                # Keep promotion gates data-driven: no fixed drawdown/hit-rate thresholds.
                sane = (
                    is_finite_number(score)
                    and is_finite_number(mean_ret)
                    and is_finite_number(mean_dd)
                    and is_finite_number(pos_rate)
                    and (global_windows >= int(args.global_min_windows))
                )
                holdout_pass = True
                if int(args.holdout_bars) > 0:
                    holdout_pass = holdout_windows >= int(args.holdout_min_windows)
                    champ_hold_w = int(global_champ_metrics.get("holdout_windows", 0))
                    champ_hold_score = float(global_champ_metrics.get("holdout_score", -1e9))
                    if holdout_pass and champ_hold_w >= int(args.holdout_min_windows):
                        holdout_pass = holdout_score > (champ_hold_score + float(args.holdout_min_improve))

                recent_pass = True
                recent_beats = 0
                recent_compared = 0
                if int(args.promote_recent_windows) > 0:
                    champ_recent = list(global_champ_metrics.get("main_recent_objectives", []))
                    recent_compared = min(int(args.promote_recent_windows), len(cand_recent), len(champ_recent))
                    if recent_compared > 0:
                        cand_tail = cand_recent[-recent_compared:]
                        champ_tail = champ_recent[-recent_compared:]
                        recent_beats = sum(
                            1
                            for c_obj, g_obj in zip(cand_tail, champ_tail)
                            if float(c_obj) > (float(g_obj) + float(args.promote_recent_min_improve))
                        )
                        needed = min(recent_compared, int(args.promote_recent_min_wins))
                        recent_pass = recent_beats >= max(1, needed)
                    else:
                        # If no champion baseline exists yet, do not block promotion.
                        recent_pass = len(champ_recent) == 0
                allow_promote_mode = mode in {"champion_refine", "refine", "pool"}
                accept = improved and sane and allow_promote_mode and holdout_pass and recent_pass and promote_fee_pass
                print(
                    "    result="
                    f"score={score:.4f} ret={mean_ret:.2%} dd={mean_dd:.2%} pos={pos_rate:.2%} "
                    f"windows={global_windows} holdout={holdout_score:.4f}/{holdout_windows} "
                    f"recent_beats={recent_beats}/{recent_compared} "
                    f"improved={improved} sane={sane} holdout_pass={holdout_pass} "
                    f"recent_pass={recent_pass} fee_pass={promote_fee_pass} accept={accept}"
                )
                pool = upsert_pool_row(pool, cp, score, mean_ret, mean_dd, pos_rate, rnd, accept)
                per_candidate.append(
                    {
                        "round": rnd,
                        "strategy_id": sid,
                        "mode": mode,
                        "source": source,
                        "score": score,
                        "score_std": metrics["score_std"],
                        "mean_test_return": mean_ret,
                        "mean_test_dd": mean_dd,
                        "positive_test_return_rate": pos_rate,
                        "global_windows": global_windows,
                        "holdout_score": holdout_score,
                        "holdout_windows": holdout_windows,
                        "recent_beats": recent_beats,
                        "recent_compared": recent_compared,
                        "holdout_pass": holdout_pass,
                        "recent_pass": recent_pass,
                        "regime_metrics": metrics.get("regime_metrics", {}),
                        "params": dict(cp),
                        "improved": improved,
                        "sane": sane,
                        "accept": accept,
                    }
                )
                if bundle_dir is not None and bundle_dir.exists():
                    snapshot_cache[sid] = bundle_dir
                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {
                        "params": cp,
                        "mode": mode,
                        "source": source,
                        "score": score,
                        "mean_ret": mean_ret,
                        "mean_dd": mean_dd,
                        "pos_rate": pos_rate,
                        "global_windows": global_windows,
                        "holdout_score": holdout_score,
                        "holdout_windows": holdout_windows,
                        "recent_beats": recent_beats,
                        "recent_compared": recent_compared,
                        "holdout_pass": holdout_pass,
                        "recent_pass": recent_pass,
                        "improved": improved,
                        "sane": sane,
                        "accept": accept,
                        "sid": sid,
                        "regime_metrics": metrics.get("regime_metrics", {}),
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
        holdout_score = float(best_candidate.get("holdout_score", -1e9))
        holdout_windows = int(best_candidate.get("holdout_windows", 0))
        recent_beats = int(best_candidate.get("recent_beats", 0))
        recent_compared = int(best_candidate.get("recent_compared", 0))
        holdout_pass = bool(best_candidate.get("holdout_pass", True))
        recent_pass = bool(best_candidate.get("recent_pass", True))
        improved = best_candidate["improved"]
        sane = best_candidate["sane"]
        accept = best_candidate["accept"]
        candidate_params = best_candidate["params"]
        candidate_mode = best_candidate.get("mode", "refine")
        candidate_source = best_candidate.get("source", "unknown")
        candidate_windows = int(best_candidate.get("global_windows", 0))
        tail = best_candidate_tail
        global_existing = has_strategy_id(champion_row(champions, "global").get("strategy_id", ""))
        if (not global_existing) and sane and candidate_mode != "explore" and holdout_pass and promote_fee_pass:
            # Bootstrap a global champion if none exists yet.
            accept = True
            improved = True
        promoted_regimes = []

        if accept:
            accepted += 1
            best_score = score
            sid = best_candidate["sid"]
            snap = snapshot_cache.get(sid)
            if snap is not None and snap.exists():
                copy_if_exists(snap / "qtable.csv", best_q)
                copy_if_exists(snap / "policy.csv", best_policy)
                copy_if_exists(snap / "train_summary.txt", best_train)
                copy_if_exists(snap / "walkforward.csv", best_wf)
                copy_if_exists(snap / "walkforward_summary.txt", best_wf_summary)
                copy_bundle_to_active(models, snap)
                champ_dst = champions_dir / "global"
                copy_active_bundle(models, champ_dst)
            else:
                copy_if_exists(models / "rl_qtable.csv", best_q)
                copy_if_exists(models / "rl_policy_latest.txt", best_policy)
                copy_if_exists(models / "rl_train_summary.txt", best_train)
                copy_if_exists(models / "rl_walkforward.csv", best_wf)
                copy_if_exists(models / "rl_walkforward_summary.txt", best_wf_summary)
                copy_active_bundle(models, champions_dir / "global")
            best_params = dict(candidate_params)
            champions = upsert_champion(
                champions,
                "global",
                candidate_params,
                best_candidate["sid"],
                score,
                mean_ret,
                mean_dd,
                pos_rate,
                int(max(1, len(pd.read_csv(models / "rl_walkforward.csv")))) if (models / "rl_walkforward.csv").exists() else 0,
                rnd,
            )
            promoted_regimes.append("global")
        else:
            # Roll back active policy to last best snapshot.
            copy_if_exists(best_q, models / "rl_qtable.csv")
            copy_if_exists(best_policy, models / "rl_policy_latest.txt")
            copy_if_exists(best_train, models / "rl_train_summary.txt")
            copy_if_exists(best_wf, models / "rl_walkforward.csv")
            copy_if_exists(best_wf_summary, models / "rl_walkforward_summary.txt")

        # Regime champions: promote best candidate per regime independently.
        for regime in ("bull", "bear", "range"):
            current = champion_row(champions, regime)
            current_score = parse_float(str(current.get("score", -1e9)), -1e9)
            candidates_with_regime = []
            for row in per_candidate:
                reg = row.get("regime_metrics", {}).get(regime)
                if not reg:
                    continue
                if int(reg.get("windows", 0)) < int(args.regime_min_windows):
                    continue
                candidates_with_regime.append((row, reg))
            if len(candidates_with_regime) == 0:
                continue
            candidates_with_regime.sort(key=lambda x: float(x[1]["score"]), reverse=True)
            cand_row, reg_metrics = candidates_with_regime[0]
            reg_score = float(reg_metrics["score"])
            reg_ret = float(reg_metrics["mean_ret"])
            reg_dd = float(reg_metrics["mean_dd"])
            reg_pos = float(reg_metrics["pos_rate"])
            reg_windows = int(reg_metrics["windows"])
            reg_improved = reg_score > (current_score + float(args.regime_min_improve))
            reg_sane = (
                is_finite_number(reg_score)
                and is_finite_number(reg_ret)
                and is_finite_number(reg_dd)
                and is_finite_number(reg_pos)
                and (reg_windows >= int(args.regime_min_windows))
            )
            reg_mode_ok = str(cand_row.get("mode", "refine")) in {"champion_refine", "refine", "pool"}
            reg_holdout_pass = bool(cand_row.get("holdout_pass", True))
            reg_recent_pass = bool(cand_row.get("recent_pass", True))
            if not (reg_improved and reg_sane and reg_mode_ok and reg_holdout_pass and reg_recent_pass and promote_fee_pass):
                continue
            sid = str(cand_row["strategy_id"])
            if not has_strategy_id(sid):
                continue
            snap = snapshot_cache.get(sid)
            if snap is None or not snap.exists():
                continue
            dst = champions_dir / regime
            dst.mkdir(parents=True, exist_ok=True)
            copy_if_exists(snap / "qtable.csv", dst / "qtable.csv")
            copy_if_exists(snap / "policy.csv", dst / "policy.csv")
            copy_if_exists(snap / "train_summary.txt", dst / "train_summary.txt")
            copy_if_exists(snap / "walkforward.csv", dst / "walkforward.csv")
            copy_if_exists(snap / "walkforward_summary.txt", dst / "walkforward_summary.txt")
            champions = upsert_champion(
                champions,
                regime,
                cand_row["params"],
                sid,
                reg_score,
                reg_ret,
                reg_dd,
                reg_pos,
                reg_windows,
                rnd,
            )
            promoted_regimes.append(regime)

        if len(pool) > 0:
            pool = pool.sort_values(["score", "accepted_count"], ascending=[False, False]).reset_index(drop=True)
            pool.to_csv(pool_path, index=False)
        champions = champions.sort_values(["regime"]).reset_index(drop=True)
        champions.to_csv(champions_path, index=False)
        champ_lines = []
        for reg in CHAMPION_REGIMES:
            cr = champion_row(champions, reg)
            champ_lines.append(
                f"{reg}: score={parse_float(str(cr['score']), -1e9):.4f} "
                f"ret={parse_float(str(cr['mean_test_return']), 0.0):.2%} "
                f"dd={parse_float(str(cr['mean_test_dd']), -1.0):.2%} "
                f"pos={parse_float(str(cr['positive_test_return_rate']), 0.0):.2%} "
                f"sid={str(cr['strategy_id'])}"
            )
        champions_summary_path.write_text(
            "RL Champions\n" + "\n".join(champ_lines) + "\n",
            encoding="ascii",
        )

        # Explicitly re-check top strategies for consistency on current profile set.
        # This provides a direct "best strategies and consistency" table for the UI.
        recheck_df = pd.DataFrame(per_candidate)
        if len(recheck_df) > 0:
            recheck_df = recheck_df.sort_values(["score", "positive_test_return_rate"], ascending=[False, False]).reset_index(drop=True)
            top_n = recheck_df.head(max(1, args.top_k)).copy()
            top_n.to_csv(args.recheck_out, index=False)

        status = (
            f"self_train_run_id={self_train_run_id}\n"
            f"round={rnd}\n"
            f"elapsed_minutes={(time.time() - start)/60.0:.2f}\n"
            f"round_fee_rate={round_fee_rate:.6f}\n"
            f"fee_schedule={','.join(f'{x:.6f}' for x in fee_schedule)}\n"
            f"promote_min_fee={promote_min_fee:.6f}\n"
            f"promote_fee_pass={promote_fee_pass}\n"
            f"accepted_rounds={accepted}\n"
            f"accept={accept}\n"
            f"improved={improved}\n"
            f"sane={sane}\n"
            f"score={score:.6f}\n"
            f"best_score={best_score:.6f}\n"
            f"champion_main_score={float(global_champ_metrics.get('main_score', -1e9)):.6f}\n"
            f"champion_holdout_score={float(global_champ_metrics.get('holdout_score', -1e9)):.6f}\n"
            f"positive_test_return_rate={pos_rate:.2%}\n"
            f"mean_test_return={mean_ret:.2%}\n"
            f"mean_test_dd={mean_dd:.2%}\n"
            f"holdout_score={holdout_score:.6f}\n"
            f"holdout_windows={holdout_windows}\n"
            f"holdout_pass={holdout_pass}\n"
            f"recent_beats={recent_beats}\n"
            f"recent_compared={recent_compared}\n"
            f"recent_pass={recent_pass}\n"
            f"candidate_episodes={candidate_params['episodes']}\n"
            f"candidate_alpha={candidate_params['alpha']:.4f}\n"
            f"candidate_gamma={candidate_params['gamma']:.4f}\n"
            f"candidate_eps_start={candidate_params['eps_start']:.4f}\n"
            f"candidate_eps_end={candidate_params['eps_end']:.4f}\n"
            f"candidate_mode={candidate_mode}\n"
            f"candidate_source={candidate_source}\n"
            f"candidate_windows={candidate_windows}\n"
            f"best_episodes={best_params['episodes']}\n"
            f"best_alpha={best_params['alpha']:.4f}\n"
            f"best_gamma={best_params['gamma']:.4f}\n"
            f"best_eps_start={best_params['eps_start']:.4f}\n"
            f"best_eps_end={best_params['eps_end']:.4f}\n"
            f"candidate_strategy_id={best_candidate['sid']}\n"
            f"pool_size={len(pool)}\n"
            f"recheck_rows={len(per_candidate)}\n"
            f"champion_promotions={','.join(promoted_regimes) if len(promoted_regimes)>0 else 'none'}\n"
            f"run_tail={tail.replace(chr(10), ' | ')}\n"
        )
        Path(args.status_out).write_text(status, encoding="ascii")
        print(status)
        print("  " + progress_bar(rnd, args.rounds))

    print("self-train loop complete")


if __name__ == "__main__":
    main()
