import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class AnalysisResult:
    signal: str
    confidence: int
    trend: str
    rationale: str
    risk: str


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = tp.diff()

    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum().abs()

    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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


def _order_blocks(df: pd.DataFrame, mult: float = 1.8, lookahead: int = 6, atr_period: int = 14):
    a = _atr(df["high"], df["low"], df["close"], atr_period)
    body = (df["close"] - df["open"]).abs()

    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)

    highs = df["high"].values
    lows = df["low"].values
    bodies = body.values
    atrs = a.values

    for i in range(len(df)):
        if np.isnan(atrs[i]) or np.isnan(bodies[i]):
            continue
        if bodies[i] < mult * atrs[i]:
            continue
        end = min(i + 1 + lookahead, len(df))
        if np.any(highs[i + 1 : end] > highs[i]):
            bull.iat[i] = True
        if np.any(lows[i + 1 : end] < lows[i]):
            bear.iat[i] = True
    return bull, bear


def build_features(df: pd.DataFrame, lookback: int = 300) -> Dict[str, Any]:
    df = df.tail(lookback).copy()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    close = df["close"]
    returns = close.pct_change()

    macd_line = df.get("macd_line")
    macd_signal = df.get("macd_signal")
    macd_hist = df.get("macd_hist")
    if macd_line is None or macd_signal is None or macd_hist is None:
        macd_line, macd_signal, macd_hist = _macd(close)

    mfi = df.get("mfi_14")
    if mfi is None:
        mfi = _mfi(df["high"], df["low"], df["close"], df["volume"], period=14)

    ob_bull = df.get("ob_bull")
    ob_bear = df.get("ob_bear")
    if ob_bull is None or ob_bear is None:
        ob_bull, ob_bear = _order_blocks(df)

    last = df.iloc[-1]
    feats = {
        "last_close": float(last["close"]),
        "return_20": float((close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0),
        "volatility_20": float(returns.tail(20).std() if len(returns) > 20 else 0),
        "macd": {
            "line": float(macd_line.iloc[-1]),
            "signal": float(macd_signal.iloc[-1]),
            "hist": float(macd_hist.iloc[-1]),
        },
        "mfi": float(mfi.iloc[-1]) if len(mfi) else None,
        "order_blocks_last_50": {
            "bull": int(ob_bull.tail(50).sum()) if len(ob_bull) else 0,
            "bear": int(ob_bear.tail(50).sum()) if len(ob_bear) else 0,
        },
    }
    return feats


def analyze_with_openai(
    df: pd.DataFrame,
    interval: str,
    lookback: int = 300,
    model: str = "gpt-4o-mini",
) -> AnalysisResult:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run .\\.python\\python.exe -m pip install -r requirements.txt")
    if not os.getenv("OPENAI_API_KEY"):
        key_path = Path("config") / "openai_key.txt"
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Run set_api_key.bat or set the environment variable."
            )

    feats = build_features(df, lookback=lookback)
    system = (
        "You are a trading analysis assistant. Use the provided indicators only. "
        "Return concise JSON with signal, confidence (0-100), trend (bull/bear/range), "
        "rationale, and risk. No extra text."
    )
    user = {
        "interval": interval,
        "features": feats,
    }

    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
    )
    text = resp.output_text.strip()
    # Strip markdown code fences if the model wraps JSON
    if text.startswith("```"):
        text = text.strip("`")
        # remove optional language label
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except Exception:
        raise RuntimeError(f"Invalid JSON from model: {text}")

    return AnalysisResult(
        signal=str(data.get("signal", "hold")).lower(),
        confidence=int(data.get("confidence", 0)),
        trend=str(data.get("trend", "range")).lower(),
        rationale=str(data.get("rationale", "")),
        risk=str(data.get("risk", "")),
    )


def list_available_models() -> list[str]:
    if OpenAI is None:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        key_path = Path("config") / "openai_key.txt"
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
    if not os.getenv("OPENAI_API_KEY"):
        return []
    client = OpenAI()
    models = client.models.list()
    ids = []
    for m in models.data:
        if hasattr(m, "id"):
            ids.append(m.id)
    # Prefer common chat-capable models, but show all if filtering yields none
    preferred = [m for m in ids if m.startswith("gpt-")]
    return sorted(preferred or ids)
