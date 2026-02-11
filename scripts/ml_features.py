import argparse
import sqlite3
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_12"] = df["close"].pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_48"] = df["ret_1"].rolling(48).std()

    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["mfi_14"] = mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["trend_up"] = (df["sma_50"] > df["sma_200"]).astype(int)

    ob_bull, ob_bear = order_blocks(df)
    df["ob_bull_recent"] = ob_bull.rolling(12).max().fillna(False).astype(int)
    df["ob_bear_recent"] = ob_bear.rolling(12).max().fillna(False).astype(int)

    # Price position relative to MAs
    df["close_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    df["close_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]

    return df


def label_outcomes(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.01) -> pd.DataFrame:
    future = df["close"].shift(-horizon)
    future_ret = (future - df["close"]) / df["close"]
    df["label"] = 0
    df.loc[future_ret > threshold, "label"] = 1
    df.loc[future_ret < -threshold, "label"] = -1
    return df


def write_to_db(df: pd.DataFrame, db_path: Path, table: str = "features"):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--db", default="data/ai_memory.db")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df = build_features(df)
    df = label_outcomes(df, horizon=args.horizon, threshold=args.threshold)

    # Drop rows with NaNs created by rolling windows or labels
    df = df.dropna().reset_index(drop=True)

    write_to_db(df, Path(args.db), table="features")
    print(f"Wrote features+labels to {args.db}")


if __name__ == "__main__":
    main()
