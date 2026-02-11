import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Cols:
    ts: str
    open: str
    high: str
    low: str
    close: str
    volume: str


def _to_datetime_series(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    # Detect ms vs s timestamps.
    if ts.max() > 10_000_000_000:  # > year 2286 in seconds, so assume ms
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, unit="s", utc=True)


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = tp.diff()

    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum().abs()

    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def bollinger(series: pd.Series, period: int, mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    up = mid + mult * std
    dn = mid - mult * std
    bw = (up - dn) / mid
    return mid, up, dn, bw


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series, fast: int, slow: int, signal: int):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
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


def order_blocks(df: pd.DataFrame, cols: Cols, atr_period: int, mult: float, lookahead: int):
    a = atr(df[cols.high], df[cols.low], df[cols.close], atr_period)
    body = (df[cols.close] - df[cols.open]).abs()

    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)

    highs = df[cols.high].values
    lows = df[cols.low].values
    bodies = body.values
    atrs = a.values

    for i in range(len(df)):
        if np.isnan(atrs[i]) or np.isnan(bodies[i]):
            continue
        if bodies[i] < mult * atrs[i]:
            continue

        end = min(i + 1 + lookahead, len(df))
        # Bull: break above candle high
        if np.any(highs[i + 1 : end] > highs[i]):
            bull.iat[i] = True
        # Bear: break below candle low
        if np.any(lows[i + 1 : end] < lows[i]):
            bear.iat[i] = True

    return bull, bear


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]

def pivots(high: pd.Series, low: pd.Series, left: int, right: int):
    n = len(high)
    ph = pd.Series(False, index=high.index)
    pl = pd.Series(False, index=low.index)
    hv = high.values
    lv = low.values
    for i in range(left, n - right):
        h = hv[i]
        l = lv[i]
        if np.all(h > hv[i - left : i]) and np.all(h >= hv[i + 1 : i + 1 + right]):
            ph.iat[i] = True
        if np.all(l < lv[i - left : i]) and np.all(l <= lv[i + 1 : i + 1 + right]):
            pl.iat[i] = True
    return ph, pl

def cluster_levels(levels: np.ndarray, tol: float) -> List[float]:
    if len(levels) == 0:
        return []
    levels = np.sort(levels)
    clusters = [levels[0]]
    counts = [1]
    for v in levels[1:]:
        if abs(v - clusters[-1]) <= tol:
            # update running mean
            counts[-1] += 1
            clusters[-1] = clusters[-1] + (v - clusters[-1]) / counts[-1]
        else:
            clusters.append(v)
            counts.append(1)
    return clusters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--col-ts", default="timestamp")
    p.add_argument("--col-open", default="open")
    p.add_argument("--col-high", default="high")
    p.add_argument("--col-low", default="low")
    p.add_argument("--col-close", default="close")
    p.add_argument("--col-volume", default="volume")

    p.add_argument("--rsi", type=int, default=None)
    p.add_argument("--mfi", type=int, default=None)
    p.add_argument("--bb", type=int, default=None)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--ma", default="")
    p.add_argument("--ema", default="")
    p.add_argument("--macd", action="store_true")
    p.add_argument("--macd-fast", type=int, default=12)
    p.add_argument("--macd-slow", type=int, default=26)
    p.add_argument("--macd-signal", type=int, default=9)

    p.add_argument("--order-blocks", action="store_true")
    p.add_argument("--ob-atr", type=int, default=14)
    p.add_argument("--ob-mult", type=float, default=1.8)
    p.add_argument("--ob-lookahead", type=int, default=6)

    p.add_argument("--sr", action="store_true")
    p.add_argument("--sr-left", type=int, default=3)
    p.add_argument("--sr-right", type=int, default=3)
    p.add_argument("--sr-cluster", type=float, default=0.0)
    p.add_argument("--sr-levels-out", default="")

    args = p.parse_args()

    cols = Cols(
        ts=args.col_ts,
        open=args.col_open,
        high=args.col_high,
        low=args.col_low,
        close=args.col_close,
        volume=args.col_volume,
    )

    df = pd.read_csv(args.input)

    # Normalize column names to lowercase for lookups
    df.columns = [c.lower() for c in df.columns]
    cols = Cols(
        ts=cols.ts.lower(),
        open=cols.open.lower(),
        high=cols.high.lower(),
        low=cols.low.lower(),
        close=cols.close.lower(),
        volume=cols.volume.lower(),
    )

    for c in [cols.ts, cols.open, cols.high, cols.low, cols.close, cols.volume]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Sort by time if needed
    df["_dt"] = _to_datetime_series(df[cols.ts])
    df = df.sort_values("_dt").reset_index(drop=True)
    df.drop(columns=["_dt"], inplace=True)

    if args.rsi:
        df[f"rsi_{args.rsi}"] = rsi(df[cols.close], args.rsi)

    if args.mfi:
        df[f"mfi_{args.mfi}"] = mfi(df[cols.high], df[cols.low], df[cols.close], df[cols.volume], args.mfi)

    if args.bb:
        mid, up, dn, bw = bollinger(df[cols.close], args.bb, args.bb_mult)
        df[f"bb_mid_{args.bb}"] = mid
        df[f"bb_up_{args.bb}"] = up
        df[f"bb_dn_{args.bb}"] = dn
        df[f"bb_bw_{args.bb}"] = bw

    ma_periods = parse_int_list(args.ma)
    for p_ in ma_periods:
        df[f"sma_{p_}"] = sma(df[cols.close], p_)

    ema_periods = parse_int_list(args.ema)
    for p_ in ema_periods:
        df[f"ema_{p_}"] = ema(df[cols.close], p_)

    if args.macd:
        macd_line, signal_line, hist = macd(
            df[cols.close], args.macd_fast, args.macd_slow, args.macd_signal
        )
        df["macd_line"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = hist

    if args.order_blocks:
        ob_bull, ob_bear = order_blocks(df, cols, args.ob_atr, args.ob_mult, args.ob_lookahead)
        df["ob_bull"] = ob_bull
        df["ob_bear"] = ob_bear

    if args.sr:
        ph, pl = pivots(df[cols.high], df[cols.low], args.sr_left, args.sr_right)
        df["sr_pivot_high"] = ph
        df["sr_pivot_low"] = pl

        if args.sr_cluster > 0:
            pivot_highs = df.loc[df["sr_pivot_high"], cols.high].values
            pivot_lows = df.loc[df["sr_pivot_low"], cols.low].values
            clustered_res = cluster_levels(pivot_highs, args.sr_cluster)
            clustered_sup = cluster_levels(pivot_lows, args.sr_cluster)
            if args.sr_levels_out:
                out = pd.DataFrame(
                    {
                        "type": ["resistance"] * len(clustered_res) + ["support"] * len(clustered_sup),
                        "price": [float(v) for v in clustered_res] + [float(v) for v in clustered_sup],
                    }
                )
                out.to_csv(args.sr_levels_out, index=False)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
