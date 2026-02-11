import argparse
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
                    st.session_state["ai_result"] = analyze_with_openai(df, interval, lookback=lookback, model=ai_model)
                except Exception as e:
                    st.session_state["ai_error"] = str(e)

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

        # Append to signal history
        history = st.session_state.get("ai_history", [])
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": r.signal,
                "trend": r.trend,
                "confidence": r.confidence,
                "interval": interval,
                "rationale": r.rationale,
                "risk": r.risk,
            }
        )
        st.session_state["ai_history"] = history[-50:]

    if "ai_history" in st.session_state and st.session_state["ai_history"]:
        with st.expander("AI Signal History", expanded=False):
            hist_df = pd.DataFrame(st.session_state["ai_history"])
            st.dataframe(hist_df, use_container_width=True)

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

    # Oscillators in a separate tab
    tab1, tab2 = st.tabs(["Price", "Oscillators"])

    with tab1:
        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
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


if __name__ == "__main__":
    main()
