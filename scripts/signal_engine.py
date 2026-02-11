import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = tp.diff()

    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum().abs()

    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


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


def order_blocks(df: pd.DataFrame, mult: float = 1.8, lookahead: int = 6, atr_period: int = 14):
    a = atr(df["high"], df["low"], df["close"], atr_period)
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


def build_signals(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    mfi_period: int = 14,
    mfi_lower: float = 40,
    mfi_upper: float = 80,
    ob_lookback: int = 12,
    ob_mult: float = 1.8,
    ob_lookahead: int = 6,
    sma_fast: int = 50,
    sma_slow: int = 200,
    trend_filter: bool = True,
):
    df = df.copy()
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(
        df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    df["mfi_14"] = mfi(df["high"], df["low"], df["close"], df["volume"], mfi_period)
    df["sma_fast"] = df["close"].rolling(sma_fast).mean()
    df["sma_slow"] = df["close"].rolling(sma_slow).mean()

    ob_bull, ob_bear = order_blocks(df, mult=ob_mult, lookahead=ob_lookahead)
    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear

    # Trend filter
    if trend_filter:
        df["trend_up"] = df["sma_fast"] > df["sma_slow"]
    else:
        df["trend_up"] = True

    # MACD cross
    df["macd_cross_up"] = (df["macd_line"] > df["macd_signal"]) & (
        df["macd_line"].shift(1) <= df["macd_signal"].shift(1)
    )
    df["macd_cross_dn"] = (df["macd_line"] < df["macd_signal"]) & (
        df["macd_line"].shift(1) >= df["macd_signal"].shift(1)
    )

    # Order block recent
    df["ob_bull_recent"] = df["ob_bull"].rolling(ob_lookback).max().fillna(False).astype(bool)
    df["ob_bear_recent"] = df["ob_bear"].rolling(ob_lookback).max().fillna(False).astype(bool)

    # Entry/exit rules (long-only baseline)
    df["buy"] = (
        df["trend_up"]
        & df["macd_cross_up"]
        & (df["mfi_14"] > mfi_lower)
        & (df["mfi_14"] < mfi_upper)
        & df["ob_bull_recent"]
    )
    df["sell"] = df["macd_cross_dn"] | (df["mfi_14"] > mfi_upper) | df["ob_bear_recent"]

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--output", default="data/signals/btc_1h_signals.csv")
    p.add_argument("--mfi-lower", type=float, default=40)
    p.add_argument("--mfi-upper", type=float, default=80)
    p.add_argument("--ob-lookback", type=int, default=12)
    p.add_argument("--trend-filter", action="store_true")
    p.add_argument("--no-trend-filter", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    trend_filter = True
    if args.no_trend_filter:
        trend_filter = False
    if args.trend_filter:
        trend_filter = True

    df = build_signals(
        df,
        mfi_lower=args.mfi_lower,
        mfi_upper=args.mfi_upper,
        ob_lookback=args.ob_lookback,
        trend_filter=trend_filter,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
