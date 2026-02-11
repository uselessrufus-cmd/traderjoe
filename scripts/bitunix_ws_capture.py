import argparse
import csv
import json
import math
import time
from pathlib import Path

import websocket

WS_URL = "wss://fapi.bitunix.com/public/"

INTERVAL_TO_CHANNEL = {
    "1m": "market_kline_1min",
    "5m": "market_kline_5min",
    "15m": "market_kline_15min",
    "1h": "market_kline_60min",
    "4h": "market_kline_4h",
    "12h": "market_kline_12h",
    "1d": "market_kline_1day",
    "1w": "market_kline_1week",
    "1M": "market_kline_1month",
}

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,  # approx 30d for bucketing
}


def bucket_ts(ts_ms: int, interval: str) -> int:
    sec = INTERVAL_SECONDS[interval]
    ts_s = int(ts_ms / 1000)
    return (ts_s // sec) * sec


def load_last_ts(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", newline="") as f:
            row = None
            for row in csv.DictReader(f):
                pass
            if row and "timestamp" in row:
                return int(row["timestamp"])
    except Exception:
        return 0
    return 0


def append_or_replace(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
            w.writeheader()
            w.writerow(row)
        return

    rows = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if rows and int(rows[-1]["timestamp"]) == int(row["timestamp"]):
        rows[-1] = row
    else:
        rows.append(row)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def trim_old(path: Path, retain_days: int):
    if not path.exists():
        return
    cutoff = int(time.time()) - retain_days * 86400
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r if int(row["timestamp"]) >= cutoff]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--intervals", default="1m,5m,15m,1h,4h,12h,1d,1w,1M")
    p.add_argument("--out-dir", default="data/bitunix/live")
    p.add_argument("--retain-days", type=int, default=7)
    args = p.parse_args()

    symbol = args.symbol
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    out_dir = Path(args.out_dir)

    def on_open(ws):
        subs = []
        for i in intervals:
            ch = INTERVAL_TO_CHANNEL.get(i)
            if ch:
                subs.append({"symbol": symbol, "ch": ch})
        msg = {"op": "subscribe", "args": subs}
        ws.send(json.dumps(msg))
        print(f"Subscribed: {symbol} {intervals}")

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except Exception:
            return

        if "data" not in msg or "ch" not in msg:
            return

        ch = msg.get("ch", "")
        data = msg.get("data", {})
        ts = data.get("ts")
        if ts is None:
            return

        interval = None
        for k, v in INTERVAL_TO_CHANNEL.items():
            if v == ch:
                interval = k
                break
        if interval is None:
            return

        row = {
            "timestamp": bucket_ts(int(ts), interval),
            "open": data.get("o"),
            "high": data.get("h"),
            "low": data.get("l"),
            "close": data.get("c"),
            "volume": data.get("b"),
        }

        out_path = out_dir / f"bitunix_{interval}.csv"
        append_or_replace(out_path, row)
        trim_old(out_path, args.retain_days)
        # Minimal heartbeat
        if interval == "1m":
            print(f"1m tick {row['timestamp']}")

    def on_error(ws, error):
        print("WS error:", error)

    def on_close(ws, *_):
        print("WS closed, reconnecting in 5s...")
        time.sleep(5)
        run()

    def run():
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=20, ping_timeout=10)

    run()


if __name__ == "__main__":
    main()
