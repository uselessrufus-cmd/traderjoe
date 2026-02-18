import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
import sys
from pathlib import Path as _Path
from datetime import datetime, timezone
import textwrap
import time

try:
    from ai_analysis import analyze_with_openai, list_available_models
except Exception:
    analyze_with_openai = None
    list_available_models = None

sys.path.append(str((_Path(__file__).resolve().parent / "scripts")))
from signal_engine import build_signals


def _load_tf(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def _compute_trend(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 200:
        return 0.0
    close = df["close"]
    sma_fast = close.rolling(50).mean()
    sma_slow = close.rolling(200).mean()
    last = len(df) - 1
    if pd.isna(sma_fast.iloc[last]) or pd.isna(sma_slow.iloc[last]):
        return 0.0
    return 1.0 if sma_fast.iloc[last] > sma_slow.iloc[last] else -1.0


def _compute_momentum(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 30:
        return 0.0
    close = df["close"]
    ret = close.pct_change()
    mom = ret.tail(12).mean()
    vol = ret.tail(48).std()
    if pd.isna(mom) or pd.isna(vol) or vol == 0:
        return 0.0
    return float(mom / vol)


def _position_size(score: float) -> int:
    # Map score [-2, 2] to position size [0, 100]
    score = max(-2.0, min(2.0, score))
    size = int((score + 2.0) / 4.0 * 100)
    return size


def _load_best_strategy() -> dict:
    path = _Path("data") / "models" / "best_strategy.txt"
    cfg = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in {"mfi_lower", "mfi_upper", "ob_lookback"}:
            cfg[k] = int(float(v))
        elif k in {"trend_filter", "allow_short"}:
            cfg[k] = v.lower() == "true"
        elif k in {"stop_loss", "take_profit"}:
            cfg[k] = float(v)
        else:
            cfg[k] = v
    return cfg

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a 'timestamp' column")

    ts = df["timestamp"].astype("int64")
    if ts.max() > 10_000_000_000:
        dt = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        dt = pd.to_datetime(ts, unit="s", utc=True)
    df["_dt"] = dt
    return df.sort_values("_dt").reset_index(drop=True)


def add_levels(fig, levels_df, ypad):
    if levels_df is None or levels_df.empty:
        return
    for _, row in levels_df.iterrows():
        y = float(row["price"])
        color = "#2b7a0b" if row["type"] == "support" else "#a61b1b"
        fig.add_hline(y=y, line_color=color, line_width=1, opacity=0.6)
        fig.add_annotation(
            xref="paper",
            x=1.002,
            y=y,
            text=row["type"],
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.6)",
            yshift=0,
        )


def add_indicator(fig, df, col, name, row=2, color=None):
    if col not in df.columns:
        return
    fig.add_trace(
        go.Scatter(x=df["_dt"], y=df[col], mode="lines", name=name, line=dict(color=color)),
        row=row,
        col=1,
    )


def _ts_to_dt(series: pd.Series) -> pd.Series:
    ts = pd.to_numeric(series, errors="coerce")
    if ts.dropna().empty:
        return pd.to_datetime(series, errors="coerce", utc=True)
    if ts.max() > 10_000_000_000:
        return pd.to_datetime(ts, unit="ms", errors="coerce", utc=True)
    return pd.to_datetime(ts, unit="s", errors="coerce", utc=True)


def _render_equity_chart(eq_path: Path, title: str):
    if not eq_path.exists():
        st.info(f"Missing file: {eq_path}")
        return
    try:
        eq_df = pd.read_csv(eq_path)
    except Exception as e:
        st.warning(f"Could not read equity file: {e}")
        return
    if "step" not in eq_df.columns or "equity" not in eq_df.columns or len(eq_df) == 0:
        st.info("Equity file is empty or invalid.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_df["step"], y=eq_df["equity"], mode="lines", name=title))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--levels", default="")
    args, _ = p.parse_known_args()

    st.set_page_config(page_title="TraderJoe Chart", layout="wide")
    st.title("TraderJoe Chart")

    st.caption("Signals are informational only, not financial advice.")

    interval = st.sidebar.radio(
        "Interval",
        options=["1h"],
        index=0,
        horizontal=True,
    )
    default_data = Path(f"data/bitstamp/ohlc/bitstamp_{interval}.csv")
    data_path = Path(st.sidebar.text_input("Data CSV", str(default_data)))
    levels_path = Path(st.sidebar.text_input("S/R Levels CSV", args.levels)) if args.levels else None

    if not data_path.exists():
        st.warning(f"Data file not found: {data_path}")
        st.stop()

    df = load_csv(data_path)

    levels_df = None
    if levels_path and levels_path.exists():
        try:
            levels_df = pd.read_csv(levels_path)
            levels_df.columns = [c.lower() for c in levels_df.columns]
        except Exception:
            levels_df = None

    # UI
    colset = set(df.columns)
    has_ohlc = all(c in colset for c in ["open", "high", "low", "close"]) 

    # Build indicators/signals dynamically using best strategy if available
    best_cfg = _load_best_strategy()
    df = build_signals(
        df,
        mfi_lower=best_cfg.get("mfi_lower", 40),
        mfi_upper=best_cfg.get("mfi_upper", 80),
        ob_lookback=best_cfg.get("ob_lookback", 12),
        trend_filter=best_cfg.get("trend_filter", True),
    )

    indicators = st.sidebar.multiselect(
        "Indicators",
        options=[c for c in df.columns if c.startswith(("rsi_", "mfi_", "bb_", "sma_", "ema_", "macd_"))],
        default=[c for c in df.columns if c.startswith(("sma_fast", "sma_slow", "macd_line"))],
    )

    show_volume = st.sidebar.checkbox("Show volume", value="volume" in colset)
    show_levels = False
    limit = st.sidebar.slider("Bars", min_value=200, max_value=2000, value=800, step=100)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        st.sidebar.caption("Refreshing every 60s")
        components.html(
            "<script>setTimeout(()=>{window.location.reload();}, 60000);</script>",
            height=0,
            width=0,
        )

    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    with st.sidebar.expander("AI Analysis", expanded=False):
        if analyze_with_openai is None:
            st.info("AI disabled: openai package missing. Run .\\.python\\python.exe -m pip install -r requirements.txt")
        else:
            model_options = []
            if list_available_models is not None:
                try:
                    model_options = list_available_models()
                except Exception:
                    model_options = []

            # Price-tiered shortlist (only if available)
            cheap = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini"]
            medium = ["gpt-4.1", "gpt-4o", "gpt-5", "gpt-5.1"]
            premium = ["gpt-5-pro", "o3-pro", "o1"]

            if model_options:
                available = set(model_options)
                shortlist = [m for m in cheap + medium + premium if m in available]
            else:
                shortlist = cheap + medium + premium

            default_index = 0
            if "gpt-4o-mini" in shortlist:
                default_index = shortlist.index("gpt-4o-mini")
            ai_model = st.selectbox("Model", options=shortlist, index=default_index)
            lookback = st.number_input("Lookback bars", min_value=100, max_value=2000, value=300, step=50)
            if st.button("Run AI Analysis"):
                try:
                    result = analyze_with_openai(df, interval, lookback=lookback, model=ai_model)
                    st.session_state["ai_result"] = result
                    history = st.session_state.get("ai_history", [])
                    history.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "signal": result.signal,
                            "trend": result.trend,
                            "confidence": result.confidence,
                            "interval": interval,
                            "rationale": result.rationale,
                            "risk": result.risk,
                        }
                    )
                    st.session_state["ai_history"] = history[-50:]
                except Exception as e:
                    st.session_state["ai_error"] = str(e)

    # ML Signal (LightGBM)
    ml_path = _Path("data") / "models" / "lgbm_latest_signal.txt"
    ml_signal = None
    if ml_path.exists():
        with st.expander("ML Signal (LightGBM)", expanded=False):
            if ml_path.exists():
                st.markdown("**Single timeframe (1h) — Default**")
                ml_text = ml_path.read_text(encoding="ascii", errors="ignore").strip()
                st.code(ml_text, language="text")
                ml_signal = ml_text

    best_path = _Path("data") / "models" / "best_strategy.txt"
    if best_path.exists():
        with st.expander("Best Strategy (Active)", expanded=False):
            st.code(best_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    perf_path = _Path("data") / "models" / "performance_summary.txt"
    if perf_path.exists():
        with st.expander("Model Performance Summary", expanded=False):
            st.code(perf_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    paper_path = _Path("data") / "models" / "paper_summary.txt"
    if paper_path.exists():
        with st.expander("Paper Trading Summary", expanded=False):
            st.code(paper_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    sim_path = _Path("data") / "models" / "sim_report.txt"
    if sim_path.exists():
        with st.expander("Simulation Report", expanded=False):
            st.code(sim_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    sim_sum = _Path("data") / "models" / "sim_summary.txt"
    if sim_sum.exists():
        with st.expander("Simulation Summary", expanded=False):
            st.code(sim_sum.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    sim_explore = _Path("data") / "models" / "sim_explore_summary.txt"
    if sim_explore.exists():
        with st.expander("Simulation Explorer Summary", expanded=False):
            st.code(sim_explore.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    sim_hist = _Path("data") / "models" / "sim_history.csv"
    if sim_hist.exists():
        with st.expander("Simulation History", expanded=False):
            hist_df = pd.read_csv(sim_hist)
            if len(hist_df) >= 2:
                last = hist_df.iloc[-1]
                prev = hist_df.iloc[-2]
                delta = last["stability"] - prev["stability"]
                st.write(f"Stability change vs last run: {delta:+.2f}%")
            st.dataframe(hist_df.tail(10), use_container_width=True)

    strat_sum = _Path("data") / "models" / "strategy_explore_summary.txt"
    if strat_sum.exists():
        with st.expander("Strategy Explorer Summary", expanded=False):
            st.code(strat_sum.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    profit_path = _Path("data") / "models" / "profit_signal.txt"
    if profit_path.exists():
        with st.expander("Profit Model Signal", expanded=False):
            st.code(profit_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    regime_path = _Path("data") / "models" / "regime_report.txt"
    if regime_path.exists():
        with st.expander("Regime Report", expanded=False):
            st.code(regime_path.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    hist_sim = _Path("data") / "models" / "historical_sim_report.txt"
    if hist_sim.exists():
        with st.expander("Historical Simulation Report", expanded=False):
            st.code(hist_sim.read_text(encoding="ascii", errors="ignore").strip(), language="text")

    rl_status = _Path("data") / "models" / "rl_status.txt"
    if rl_status.exists():
        with st.expander("RL Live Status", expanded=False):
            st.code(rl_status.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    rl_train_summary = _Path("data") / "models" / "rl_train_summary.txt"
    if rl_train_summary.exists():
        with st.expander("RL Training Summary", expanded=False):
            st.code(rl_train_summary.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    rl_policy = _Path("data") / "models" / "rl_policy_latest.txt"
    if rl_policy.exists():
        with st.expander("RL Policy (Latest)", expanded=False):
            pol_df = pd.read_csv(rl_policy)
            st.dataframe(pol_df.head(200), use_container_width=True)
    rl_rationale_summary = _Path("data") / "models" / "rl_rationale_summary.txt"
    if rl_rationale_summary.exists():
        with st.expander("RL Rationale Summary", expanded=False):
            st.code(rl_rationale_summary.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    rl_rationale_assessment = _Path("data") / "models" / "rl_rationale_assessment.csv"
    if rl_rationale_assessment.exists():
        with st.expander("RL Rationale Assessment", expanded=False):
            rdf = pd.read_csv(rl_rationale_assessment)
            st.dataframe(rdf.head(100), use_container_width=True)
    rl_rationale_memory = _Path("data") / "models" / "rl_rationale_memory.csv"
    if rl_rationale_memory.exists():
        with st.expander("RL Rationale Memory", expanded=False):
            mdf = pd.read_csv(rl_rationale_memory)
            st.dataframe(mdf.head(100), use_container_width=True)
    rl_eq = _Path("data") / "models" / "rl_test_equity.csv"
    if rl_eq.exists():
        with st.expander("RL Test Equity Curve", expanded=False):
            eq_df = pd.read_csv(rl_eq)
            if "step" in eq_df.columns and "equity" in eq_df.columns and len(eq_df) > 0:
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(x=eq_df["step"], y=eq_df["equity"], mode="lines", name="RL Test Equity"))
                eq_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(eq_fig, use_container_width=True)
            else:
                st.info("RL equity curve file is empty.")
    rl_wf_summary = _Path("data") / "models" / "rl_walkforward_summary.txt"
    if rl_wf_summary.exists():
        with st.expander("RL Walk-Forward Summary", expanded=False):
            st.code(rl_wf_summary.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    rl_wf = _Path("data") / "models" / "rl_walkforward.csv"
    if rl_wf.exists():
        with st.expander("RL Walk-Forward Windows", expanded=False):
            wf_df = pd.read_csv(rl_wf)
            st.dataframe(wf_df.tail(30), use_container_width=True)
    rl_self_status = _Path("data") / "models" / "rl_self_train_status.txt"
    if rl_self_status.exists():
        with st.expander("RL Self-Train Status", expanded=False):
            st.code(rl_self_status.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    rl_pool = _Path("data") / "models" / "rl_strategy_pool.csv"
    if rl_pool.exists():
        with st.expander("RL Strategy Pool", expanded=False):
            pool_df = pd.read_csv(rl_pool)
            st.dataframe(pool_df.head(50), use_container_width=True)
    rl_recheck = _Path("data") / "models" / "rl_strategy_recheck.csv"
    if rl_recheck.exists():
        with st.expander("RL Strategy Recheck (Top)", expanded=False):
            rchk_df = pd.read_csv(rl_recheck)
            st.dataframe(rchk_df.head(50), use_container_width=True)
    rl_champions = _Path("data") / "models" / "rl_champions.csv"
    if rl_champions.exists():
        with st.expander("RL Champions", expanded=False):
            cdf = pd.read_csv(rl_champions)
            st.dataframe(cdf, use_container_width=True)
    rl_champions_summary = _Path("data") / "models" / "rl_champions_summary.txt"
    if rl_champions_summary.exists():
        with st.expander("RL Champions Summary", expanded=False):
            st.code(rl_champions_summary.read_text(encoding="ascii", errors="ignore").strip(), language="text")
    fee_root = _Path("data") / "models" / "fee_ladder"
    if fee_root.exists():
        with st.expander("RL Fee Ladder (Candidates + Champions)", expanded=False):
            fee_status = fee_root / "rl_self_train_status.txt"
            fee_summary = fee_root / "rl_champions_summary.txt"
            fee_champions = fee_root / "rl_champions.csv"
            fee_recheck = fee_root / "rl_strategy_recheck.csv"

            cols = st.columns(2)
            with cols[0]:
                if fee_status.exists():
                    st.caption("Fee Ladder Status")
                    st.code(fee_status.read_text(encoding="ascii", errors="ignore").strip(), language="text")
            with cols[1]:
                if fee_summary.exists():
                    st.caption("Fee Ladder Champions Summary")
                    st.code(fee_summary.read_text(encoding="ascii", errors="ignore").strip(), language="text")

            if fee_champions.exists():
                cdf = pd.read_csv(fee_champions)
                st.caption("Fee Ladder Champions")
                st.dataframe(cdf, use_container_width=True)

            if fee_recheck.exists():
                rdf = pd.read_csv(fee_recheck)
                st.caption("Fee Ladder Recheck (Top)")
                st.dataframe(rdf.head(30), use_container_width=True)

            st.caption("Equity Curves")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Champion Equity Curve**")
                champ_eq_map = {}
                for regime in ("global", "bull", "bear", "range"):
                    p = fee_root / "champions" / regime / "test_equity.csv"
                    if p.exists():
                        champ_eq_map[regime] = p
                if len(champ_eq_map) == 0:
                    st.info("No champion equity curves yet.")
                else:
                    regime = st.selectbox(
                        "Champion regime",
                        options=list(champ_eq_map.keys()),
                        index=0,
                        key="fee_champion_regime",
                    )
                    _render_equity_chart(champ_eq_map[regime], f"Champion {regime}")

            with c2:
                st.markdown("**Candidate Equity Curve**")
                cand_files = sorted(
                    fee_root.glob("_candidate_snapshots/round_*/cand_*/profile_*/test_equity.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if len(cand_files) == 0:
                    st.info("No candidate equity curves yet.")
                else:
                    labels = []
                    for p in cand_files[:100]:
                        try:
                            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        except Exception:
                            mtime = "unknown"
                        labels.append(f"{mtime} | {p.relative_to(fee_root)}")
                    selected = st.selectbox(
                        "Candidate run",
                        options=labels,
                        index=0,
                        key="fee_candidate_curve",
                    )
                    selected_idx = labels.index(selected)
                    _render_equity_chart(cand_files[selected_idx], "Candidate")

    exp_index = _Path("data") / "models" / "experiments" / "index.csv"
    if exp_index.exists():
        with st.expander("Experiment Tracking", expanded=False):
            try:
                exp_df = pd.read_csv(exp_index)
                if len(exp_df) == 0:
                    st.info("No experiment runs logged yet.")
                else:
                    show_cols = [
                        c
                        for c in [
                            "created_at_utc",
                            "run_id",
                            "source",
                            "mode",
                            "objective",
                            "test_return",
                            "test_dd",
                            "win_rate",
                            "tag",
                            "parent_run_id",
                        ]
                        if c in exp_df.columns
                    ]
                    st.caption(f"Total runs: {len(exp_df)}")
                    st.dataframe(exp_df[show_cols].tail(50).iloc[::-1], use_container_width=True)
                    latest = exp_df.iloc[-1]
                    run_dir = _Path(str(latest.get("run_dir", "")))
                    if run_dir.exists():
                        mpath = run_dir / "metrics.json"
                        ppath = run_dir / "params.json"
                        meta = run_dir / "metadata.json"
                        cols = st.columns(3)
                        cols[0].write(f"Latest run: `{latest.get('run_id', '')}`")
                        cols[1].write(f"Mode: `{latest.get('mode', '')}`")
                        cols[2].write(f"Source: `{latest.get('source', '')}`")
                        if mpath.exists():
                            cols2 = st.columns(2)
                            cols2[0].code(mpath.read_text(encoding="ascii", errors="ignore").strip(), language="json")
                        if ppath.exists():
                            cols2 = st.columns(2)
                            cols2[1].code(ppath.read_text(encoding="ascii", errors="ignore").strip(), language="json")
                        if meta.exists():
                            meta_obj = json.loads(meta.read_text(encoding="ascii", errors="ignore"))
                            st.write("Artifacts")
                            st.json(meta_obj.get("artifacts", {}))
            except Exception as e:
                st.warning(f"Could not read experiment index: {e}")

    trades_csv = _Path("data") / "models" / "historical_sim_trades.csv"
    preds_csv = _Path("data") / "models" / "historical_sim_predictions.csv"
    split_csv = _Path("data") / "models" / "historical_sim_split.csv"
    bars_csv = _Path("data") / "models" / "historical_sim_bars.csv"
    if preds_csv.exists():
        with st.expander("Train/Test Debug Chart", expanded=False):
            auto_advance = False
            auto_advance_ms = 0
            pred_df = pd.read_csv(preds_csv)
            if "timestamp" in pred_df.columns:
                pred_df["_dt"] = _ts_to_dt(pred_df["timestamp"])
            elif "_dt" in pred_df.columns:
                pred_df["_dt"] = pd.to_datetime(pred_df["_dt"], utc=True, errors="coerce")
            else:
                pred_df["_dt"] = pd.NaT
            pred_df = pred_df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
            debug_indicator_cols = [
                c
                for c in pred_df.columns
                if c
                in {
                    "macd_line",
                    "macd_signal",
                    "macd_hist",
                    "mfi_14",
                    "atr_14",
                    "close_vs_sma50",
                    "close_vs_sma200",
                }
            ]
            selected_debug_inds = st.multiselect(
                "Debug indicators",
                options=debug_indicator_cols,
                default=[c for c in ["macd_line", "mfi_14"] if c in debug_indicator_cols],
            )

            split = None
            train_start = None
            train_end = None
            test_start = None
            test_end = None
            if split_csv.exists():
                split_df = pd.read_csv(split_csv)
                if len(split_df) > 0:
                    split = split_df.iloc[0]
                    train_start = _ts_to_dt(pd.Series([split["train_start_ts"]])).iloc[0]
                    train_end = _ts_to_dt(pd.Series([split["train_end_ts"]])).iloc[0]
                    test_start = _ts_to_dt(pd.Series([split["test_start_ts"]])).iloc[0]
                    test_end = _ts_to_dt(pd.Series([split["test_end_ts"]])).iloc[0]

            dbg_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
            price_df = load_csv(data_path)[["timestamp", "close", "_dt"]]
            if split is not None:
                price_df = price_df[(price_df["_dt"] >= test_start) & (price_df["_dt"] <= test_end)].copy()

            tdf = None
            if trades_csv.exists():
                tdf = pd.read_csv(trades_csv)
                if "timestamp" in tdf.columns:
                    tdf["_dt"] = _ts_to_dt(tdf["timestamp"])
                elif "dt" in tdf.columns:
                    tdf["_dt"] = pd.to_datetime(tdf["dt"], utc=True, errors="coerce")
                else:
                    tdf["_dt"] = pd.NaT
                tdf = tdf.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

            bar_df = None
            if bars_csv.exists():
                bar_df = pd.read_csv(bars_csv)
                if "timestamp" in bar_df.columns:
                    bar_df["_dt"] = _ts_to_dt(bar_df["timestamp"])
                elif "dt" in bar_df.columns:
                    bar_df["_dt"] = pd.to_datetime(bar_df["dt"], utc=True, errors="coerce")
                else:
                    bar_df["_dt"] = pd.NaT
                bar_df = bar_df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

            step = None
            cutoff_dt = None
            zoom_start = None
            zoom_end = None
            mode = st.radio("Playback mode", options=["test bars", "trade events"], horizontal=True)
            if mode == "test bars" and bar_df is not None and len(bar_df) > 0:
                max_step = len(bar_df)
                if "debug_playing" not in st.session_state:
                    st.session_state["debug_playing"] = False
                if "debug_play_speed" not in st.session_state:
                    st.session_state["debug_play_speed"] = 1
                if "debug_play_ms" not in st.session_state:
                    st.session_state["debug_play_ms"] = 800
                if "debug_test_bar_step" not in st.session_state:
                    st.session_state["debug_test_bar_step"] = max_step
                if "debug_next_step" in st.session_state:
                    st.session_state["debug_test_bar_step"] = min(max_step, int(st.session_state["debug_next_step"]))
                    del st.session_state["debug_next_step"]
                st.session_state["debug_test_bar_step"] = min(st.session_state["debug_test_bar_step"], max_step)

                c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 2, 2])
                if c1.button("Play/Pause", key="debug_play_pause_btn"):
                    if not st.session_state["debug_playing"] and st.session_state["debug_test_bar_step"] >= max_step:
                        st.session_state["debug_test_bar_step"] = 0
                    st.session_state["debug_playing"] = not st.session_state["debug_playing"]
                if c2.button("Reset", key="debug_play_reset_btn"):
                    st.session_state["debug_test_bar_step"] = 0
                    st.session_state["debug_playing"] = False
                c3.slider("Bars/tick", min_value=1, max_value=20, key="debug_play_speed")
                c4.slider("Tick ms", min_value=200, max_value=3000, step=100, key="debug_play_ms")
                start_balance = float(c5.number_input("Start balance", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=100.0, key="debug_start_balance"))
                auto_zoom_trade = st.checkbox("Auto zoom current trade", value=True, key="debug_auto_zoom_trade")
                zoom_pad = st.slider("Zoom padding bars", min_value=2, max_value=48, value=8, key="debug_zoom_pad")

                st.slider("Test bar", min_value=0, max_value=max_step, key="debug_test_bar_step")
                step = st.session_state["debug_test_bar_step"]
                if step > 0:
                    cutoff_dt = bar_df.iloc[step - 1]["_dt"]
                    cur = bar_df.iloc[step - 1]
                    st.write(
                        f"Bar {step}/{len(bar_df)} | signal={cur['signal']} | action={cur['action']} | "
                        f"pos={cur['position_side']} size={float(cur['position_size']):.3f} | "
                        f"unrealized={float(cur.get('position_unrealized', 0.0)):.4f} | reward={float(cur['reward']):.4f}"
                    )
                    if "rationale" in cur and str(cur["rationale"]).strip():
                        st.code(textwrap.fill(f"Rationale: {cur['rationale']}", width=110), language="text")
                    hist = bar_df.iloc[:step]
                    realized_rewards = hist["reward"].fillna(0.0).astype(float)
                    if len(realized_rewards) > 0:
                        bal_curve = (1.0 + realized_rewards).cumprod()
                        sim_balance = start_balance * float(bal_curve.iloc[-1])
                    else:
                        sim_balance = start_balance
                    unrealized = float(cur.get("position_unrealized", 0.0))
                    in_pos = str(cur.get("position_side", "none")) != "none"
                    pos_size = min(1.0, float(cur.get("position_size", 0.0)))
                    capital_in_play = sim_balance * pos_size if in_pos else 0.0
                    unrealized_value = capital_in_play * unrealized if in_pos else 0.0
                    equity = sim_balance + unrealized_value
                    last_trade_pnl_usd = 0.0
                    if tdf is not None and len(tdf) > 0 and cutoff_dt is not None:
                        trade_hist = tdf[tdf["_dt"] <= cutoff_dt].copy()
                        if not trade_hist.empty and "action" in trade_hist.columns:
                            realized = trade_hist[~trade_hist["action"].isin(["open", "dca_add"])].copy()
                            if not realized.empty:
                                last_idx = realized.index[-1]
                                pnl_all = trade_hist["pnl"].fillna(0.0).astype(float)
                                pnl_last = float(trade_hist.loc[last_idx, "pnl"])
                                pre = trade_hist.loc[:last_idx].iloc[:-1]
                                if len(pre) > 0:
                                    bal_before_last = start_balance * float((1.0 + pre["pnl"].fillna(0.0).astype(float)).cumprod().iloc[-1])
                                else:
                                    bal_before_last = start_balance
                                last_trade_pnl_usd = bal_before_last * pnl_last
                    st.code(
                        (
                            "State:\n"
                            f"- current_pnl_pct={unrealized * 100:.2f}%\n"
                            f"- current_pnl_value={unrealized_value:,.2f}\n"
                            f"- last_trade_pnl_usd={last_trade_pnl_usd:,.2f}\n"
                            f"- sim_balance={sim_balance:,.2f}\n"
                            f"- equity={equity:,.2f}\n"
                            f"- capital_in_play={capital_in_play:,.2f}\n"
                            f"- position_side={cur.get('position_side', 'none')}\n"
                            f"- position_size={pos_size:.3f}"
                        ),
                        language="text",
                    )
                    if auto_zoom_trade:
                        open_actions = {"open_long", "open_short", "reverse_to_long", "reverse_to_short"}
                        trade_starts = hist[hist["action"].isin(open_actions)]
                        if not trade_starts.empty:
                            trade_start_dt = trade_starts.iloc[-1]["_dt"]
                        else:
                            trade_start_dt = hist.iloc[max(0, step - 1)]["_dt"]
                        zoom_start = trade_start_dt - pd.Timedelta(hours=int(zoom_pad))
                        zoom_end = cutoff_dt + pd.Timedelta(hours=int(zoom_pad))
                if st.session_state["debug_playing"]:
                    if step < max_step:
                        st.session_state["debug_next_step"] = min(max_step, step + st.session_state["debug_play_speed"])
                        auto_advance = True
                        auto_advance_ms = int(st.session_state["debug_play_ms"])
                    else:
                        st.session_state["debug_playing"] = False
            elif mode == "trade events" and tdf is not None and len(tdf) > 0:
                start_balance_evt = float(
                    st.number_input(
                        "Start balance (events)",
                        min_value=100.0,
                        max_value=10_000_000.0,
                        value=10_000.0,
                        step=100.0,
                        key="debug_start_balance_events",
                    )
                )
                step = st.slider("Trade step", min_value=0, max_value=len(tdf), value=len(tdf), step=1)
                if step > 0:
                    cutoff_dt = tdf.iloc[step - 1]["_dt"]
                    cur = tdf.iloc[step - 1]
                    st.write(
                        f"Step {step}/{len(tdf)} | action={cur['action']} | side={cur['side']} | "
                        f"price={float(cur['exit_price']):.2f} | pnl={float(cur['pnl']):.4f}"
                    )
                    if "rationale" in cur and str(cur["rationale"]).strip():
                        st.code(textwrap.fill(f"Rationale: {cur['rationale']}", width=110), language="text")
                    evt_hist = tdf.iloc[:step]
                    pnl_series = evt_hist["pnl"].fillna(0.0).astype(float)
                    if len(pnl_series) > 0:
                        sim_balance_evt = start_balance_evt * float((1.0 + pnl_series).cumprod().iloc[-1])
                    else:
                        sim_balance_evt = start_balance_evt
                    in_pos_evt = cur["action"] in {"open", "dca_add", "reverse_to_long", "reverse_to_short"}
                    pos_size_evt = min(1.0, float(cur.get("size", 0.0))) if in_pos_evt else 0.0
                    capital_in_play_evt = sim_balance_evt * pos_size_evt if in_pos_evt else 0.0
                    equity_evt = sim_balance_evt
                    last_trade_pnl_usd_evt = 0.0
                    if len(evt_hist) > 0 and "action" in evt_hist.columns:
                        realized_evt = evt_hist[~evt_hist["action"].isin(["open", "dca_add"])].copy()
                        if not realized_evt.empty:
                            last_idx_evt = realized_evt.index[-1]
                            pnl_last_evt = float(evt_hist.loc[last_idx_evt, "pnl"])
                            pre_evt = evt_hist.loc[:last_idx_evt].iloc[:-1]
                            if len(pre_evt) > 0:
                                bal_before_last_evt = start_balance_evt * float((1.0 + pre_evt["pnl"].fillna(0.0).astype(float)).cumprod().iloc[-1])
                            else:
                                bal_before_last_evt = start_balance_evt
                            last_trade_pnl_usd_evt = bal_before_last_evt * pnl_last_evt
                    st.code(
                        (
                            "State (events):\n"
                            f"- last_trade_pnl_usd={last_trade_pnl_usd_evt:,.2f}\n"
                            f"- sim_balance={sim_balance_evt:,.2f}\n"
                            f"- equity={equity_evt:,.2f}\n"
                            f"- capital_in_play={capital_in_play_evt:,.2f}\n"
                            f"- event_side={cur.get('side', 'none')}\n"
                            f"- event_size={pos_size_evt:.3f}"
                        ),
                        language="text",
                    )

            price_view = price_df if cutoff_dt is None else price_df[price_df["_dt"] <= cutoff_dt]
            pred_view = pred_df if cutoff_dt is None else pred_df[pred_df["_dt"] <= cutoff_dt]
            dbg_fig.add_trace(go.Scatter(x=price_view["_dt"], y=price_view["close"], mode="lines", name="Close (Test)"), row=1, col=1)
            if "confidence" in pred_view.columns and len(pred_view) > 0:
                dbg_fig.add_trace(go.Scatter(x=pred_view["_dt"], y=pred_view["confidence"], mode="lines", name="Pred Confidence"), row=2, col=1)

            if cutoff_dt is not None:
                dbg_fig.add_vline(x=cutoff_dt, line_dash="dot", line_color="#e5e7eb")

            if bar_df is not None and len(bar_df) > 0:
                bar_view = bar_df if cutoff_dt is None else bar_df[bar_df["_dt"] <= cutoff_dt]
                sig_buy = bar_view[bar_view["signal"] == "buy"]
                sig_sell = bar_view[bar_view["signal"] == "sell"]
                if not sig_buy.empty:
                    dbg_fig.add_trace(
                        go.Scatter(
                            x=sig_buy["_dt"],
                            y=sig_buy["close"],
                            mode="markers",
                            name="Signal BUY",
                            marker=dict(symbol="triangle-up", size=9, color="#00c076"),
                        ),
                        row=1,
                        col=1,
                    )
                if not sig_sell.empty:
                    dbg_fig.add_trace(
                        go.Scatter(
                            x=sig_sell["_dt"],
                            y=sig_sell["close"],
                            mode="markers",
                            name="Signal SELL",
                            marker=dict(symbol="triangle-down", size=9, color="#f6465d"),
                        ),
                        row=1,
                        col=1,
                    )
                action_rows = bar_view[bar_view["action"] != "none"]
                if not action_rows.empty:
                    dbg_fig.add_trace(
                        go.Scatter(
                            x=action_rows["_dt"],
                            y=action_rows["close"],
                            mode="markers",
                            name="Actions",
                            marker=dict(symbol="x", size=7, color="#f59e0b"),
                            text=action_rows["action"],
                            hovertemplate="%{text}<br>%{x}<br>price=%{y}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

            if tdf is not None and len(tdf) > 0:
                tdf_view = tdf if cutoff_dt is None else tdf.iloc[:step]
                if "action" in tdf.columns and "exit_price" in tdf.columns:
                    exits = tdf_view[tdf_view["action"].isin(["stop_or_tp", "reverse_to_short", "reverse_to_long", "tp1"])]
                    if not exits.empty:
                        colors = ["#00c076" if p > 0 else "#f6465d" for p in exits["pnl"].fillna(0)]
                        dbg_fig.add_trace(
                            go.Scatter(
                                x=exits["_dt"],
                                y=exits["exit_price"],
                                mode="markers",
                                name="Trade Exits",
                                marker=dict(size=9, color=colors, symbol="circle"),
                                text=exits["action"],
                            ),
                            row=1,
                            col=1,
                        )
                # Equity line intentionally removed for cleaner step-by-step debugging.

            dbg_fig.update_layout(
                template="plotly_dark",
                height=550,
                xaxis_rangeslider_visible=False,
                uirevision="debug_playback",
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            if zoom_start is not None and zoom_end is not None:
                dbg_fig.update_xaxes(range=[zoom_start, zoom_end], row=1, col=1)
                dbg_fig.update_xaxes(range=[zoom_start, zoom_end], row=2, col=1)
            st.plotly_chart(dbg_fig, use_container_width=True)

            if selected_debug_inds and len(pred_view) > 0:
                ind_fig = go.Figure()
                for c in selected_debug_inds:
                    ind_fig.add_trace(go.Scatter(x=pred_view["_dt"], y=pred_view[c], mode="lines", name=c))
                if cutoff_dt is not None:
                    ind_fig.add_vline(x=cutoff_dt, line_dash="dot", line_color="#e5e7eb")
                ind_fig.update_layout(
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(ind_fig, use_container_width=True)

            if bar_df is not None and len(bar_df) > 0:
                view_rows = bar_df if cutoff_dt is None else bar_df[bar_df["_dt"] <= cutoff_dt]
                bar_cols = ["_dt", "close", "signal", "action", "position_side", "position_size", "position_unrealized", "reward", "rationale"]
                bar_cols = [c for c in bar_cols if c in view_rows.columns]
                st.dataframe(
                    view_rows.tail(20)[bar_cols],
                    use_container_width=True,
                )
            if tdf is not None and len(tdf) > 0:
                trade_r = tdf if cutoff_dt is None else tdf[tdf["_dt"] <= cutoff_dt]
                if "rationale" in trade_r.columns:
                    st.write("**Trade Rationales (latest):**")
                    st.dataframe(
                        trade_r.tail(20)[["_dt", "action", "side", "pnl", "rationale"]],
                        use_container_width=True,
                    )
            if auto_advance:
                time.sleep(max(0.2, auto_advance_ms / 1000.0))
                st.rerun()

            if split is not None:
                st.write(
                    f"Train bars: {int(split['train_bars'])} | Test bars: {int(split['test_bars'])} | "
                    f"Train start: {train_start} | Test start: {test_start}"
                )

    # Clear buy/sell signal summary (rule-based)
    if "buy" in df.columns and "sell" in df.columns and len(df) > 0:
        last = df.iloc[-1]
        final_signal = "HOLD"
        if bool(last.get("buy")):
            final_signal = "BUY"
        elif bool(last.get("sell")):
            final_signal = "SELL"
        st.subheader(f"Signal: {final_signal}")

    # Master signal (AI + ML + Rules)
    master_votes = []
    if "ai_result" in st.session_state:
        master_votes.append(st.session_state["ai_result"].signal)
    if ml_signal:
        for line in ml_signal.splitlines():
            if line.startswith("signal="):
                master_votes.append(line.split("=", 1)[1].strip())
    if "buy" in df.columns and "sell" in df.columns and len(df) > 0:
        if bool(last.get("buy")):
            master_votes.append("buy")
        elif bool(last.get("sell")):
            master_votes.append("sell")
        else:
            master_votes.append("hold")

    if master_votes:
        buy_count = master_votes.count("buy")
        sell_count = master_votes.count("sell")
        hold_count = master_votes.count("hold")
        if buy_count > max(sell_count, hold_count):
            master = "BUY"
        elif sell_count > max(buy_count, hold_count):
            master = "SELL"
        else:
            master = "HOLD"
        st.subheader(f"Master Signal: {master}")

    # Simple mode (1h only)
    h1_trend = _compute_trend(df)
    h1_mom = _compute_momentum(df)
    composite = 0.7 * h1_trend + 0.3 * h1_mom
    pos_size = _position_size(composite)

    direction = "UPTREND" if composite > 0.25 else "DOWNTREND" if composite < -0.25 else "RANGE"
    st.subheader(f"Market Direction: {direction}")
    st.write(f"**Score:** {composite:.2f} | **Position size:** {pos_size}% (Simple 1h mode)")

    # Build chart
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.02)

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df["_dt"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color="#00c076",
                decreasing_line_color="#f6465d",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=df["_dt"], y=df["close"], mode="lines", name="Close"), row=1, col=1)

    # Price overlays: SMA/EMA/BB/MACD (overlay lines only)
    for c in indicators:
        if c.startswith(("sma_", "ema_", "bb_mid_", "bb_up_", "bb_dn_")):
            fig.add_trace(go.Scatter(x=df["_dt"], y=df[c], mode="lines", name=c), row=1, col=1)

    # Order block markers if present
    if "ob_bull" in df.columns:
        ob = df[df["ob_bull"] == True]
        if not ob.empty:
            fig.add_trace(
                go.Scatter(
                    x=ob["_dt"],
                    y=ob["low"],
                    mode="markers",
                    name="OB Bull",
                    marker=dict(symbol="triangle-up", size=10, color="#00c076"),
                ),
                row=1,
                col=1,
            )
    if "ob_bear" in df.columns:
        ob = df[df["ob_bear"] == True]
        if not ob.empty:
            fig.add_trace(
                go.Scatter(
                    x=ob["_dt"],
                    y=ob["high"],
                    mode="markers",
                    name="OB Bear",
                    marker=dict(symbol="triangle-down", size=10, color="#f6465d"),
                ),
                row=1,
                col=1,
            )

    # Buy/Sell markers
    if "buy" in df.columns:
        buys = df[df["buy"] == True]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["_dt"],
                    y=buys["low"],
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="circle", size=8, color="#00c076"),
                ),
                row=1,
                col=1,
            )
    if "sell" in df.columns:
        sells = df[df["sell"] == True]
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["_dt"],
                    y=sells["high"],
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="circle", size=8, color="#f6465d"),
                ),
                row=1,
                col=1,
            )

    # AI signal marker on latest candle
    if "ai_result" in st.session_state and len(df) > 0:
        r = st.session_state["ai_result"]
        last = df.iloc[-1]
        if r.signal in {"buy", "sell"}:
            color = "#00c076" if r.signal == "buy" else "#f6465d"
            symbol = "triangle-up" if r.signal == "buy" else "triangle-down"
            fig.add_trace(
                go.Scatter(
                    x=[last["_dt"]],
                    y=[last["low"] if r.signal == "buy" else last["high"]],
                    mode="markers",
                    name=f"AI {r.signal.upper()}",
                    marker=dict(symbol=symbol, size=14, color=color),
                ),
                row=1,
                col=1,
            )

    if show_volume and "volume" in df.columns:
        colors = ["#00c076" if c >= o else "#f6465d" for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(x=df["_dt"], y=df["volume"], name="Volume", marker_color=colors, opacity=0.6),
            row=2,
            col=1,
        )

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Price", "Oscillators", "AI"])

    with tab1:
        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            uirevision="main_price",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="#0e0f14",
            paper_bgcolor="#0e0f14",
            font=dict(color="#e5e7eb"),
        )
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True, gridcolor="#1f2937")
        fig.update_yaxes(showspikes=True, spikemode="across", showgrid=True, gridcolor="#1f2937")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        osc = [c for c in indicators if c.startswith(("rsi_", "mfi_", "macd_"))]
        if not osc:
            st.info("Select RSI/MFI in the sidebar to show oscillators.")
        else:
            osc_fig = go.Figure()
            for c in osc:
                osc_fig.add_trace(go.Scatter(x=df["_dt"], y=df[c], mode="lines", name=c))
            osc_fig.update_layout(height=400)
            st.plotly_chart(osc_fig, use_container_width=True)

    with tab3:
        if "ai_error" in st.session_state:
            st.error(st.session_state["ai_error"])

        if "ai_result" in st.session_state:
            r = st.session_state["ai_result"]
            cols = st.columns(4)
            cols[0].metric("Signal", r.signal.upper())
            cols[1].metric("Trend", r.trend.upper())
            cols[2].metric("Confidence", f"{r.confidence}")
            cols[3].metric("Interval", interval)
            st.write("**Rationale:**")
            st.code(textwrap.fill(r.rationale, width=80), language="text")
            st.write("**Risk:**")
            st.code(textwrap.fill(r.risk, width=80), language="text")
        else:
            st.info("Run AI Analysis from the sidebar to populate this tab.")

        if "ai_history" in st.session_state and st.session_state["ai_history"]:
            st.write("**AI Signal History**")
            hist_df = pd.DataFrame(st.session_state["ai_history"])
            st.dataframe(hist_df, use_container_width=True)


if __name__ == "__main__":
    main()
