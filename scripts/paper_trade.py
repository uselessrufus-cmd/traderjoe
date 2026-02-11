import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))

# Simple paper trader config
BASE_SIZE = 1.0
ADD_SIZE = 0.5
MAX_SIZE = 3.0
ALLOW_REVERSE = True
MIN_SIZE_PCT = 0.2
SIGNAL_CONF = 0.4


def get_latest_price(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "close"]].dropna()
    last = df.iloc[-1]
    return int(last["timestamp"]), float(last["close"])


def compute_position_size(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df = df[["close"]].dropna()
    if len(df) < 200:
        return 1.0
    close = df["close"]
    sma_fast = close.rolling(50).mean()
    sma_slow = close.rolling(200).mean()
    trend = 1.0 if sma_fast.iloc[-1] > sma_slow.iloc[-1] else -1.0
    ret = close.pct_change()
    mom = ret.tail(12).mean()
    vol = ret.tail(48).std()
    mom_score = 0.0 if pd.isna(vol) or vol == 0 else float(mom / vol)
    score = 0.7 * trend + 0.3 * mom_score
    score = max(-2.0, min(2.0, score))
    pct = (score + 2.0) / 4.0
    return float(pct)


def get_signal(signal_file: Path, fallback_csv: Path):
    if signal_file.exists():
        lines = signal_file.read_text(encoding="ascii", errors="ignore").splitlines()
        sig = {}
        for line in lines:
            if "=" in line:
                k, v = line.split("=", 1)
                sig[k.strip()] = v.strip()
        if sig.get("signal"):
            conf = float(sig.get("confidence", "0") or 0)
            if conf < SIGNAL_CONF:
                return "hold"
            return sig["signal"].lower()
    # fallback: rule-based signals from CSV (use best strategy if available)
    best_cfg = {}
    best_path = Path("data/models/best_strategy.txt")
    if best_path.exists():
        for line in best_path.read_text(encoding="ascii", errors="ignore").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                best_cfg[k.strip()] = v.strip()
    from signal_engine import build_signals

    df = pd.read_csv(fallback_csv)
    df.columns = [c.lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = build_signals(
        df,
        mfi_lower=int(float(best_cfg.get("mfi_lower", 40))),
        mfi_upper=int(float(best_cfg.get("mfi_upper", 80))),
        ob_lookback=int(float(best_cfg.get("ob_lookback", 12))),
        trend_filter=best_cfg.get("trend_filter", "true").lower() == "true",
    )
    last = df.iloc[-1]
    if bool(last.get("buy")):
        return "buy"
    if bool(last.get("sell")):
        return "sell"
    return "hold"


def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                side TEXT,
                size REAL,
                entry_price REAL,
                entry_ts INTEGER,
                last_action_ts INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT,
                size REAL,
                entry_price REAL,
                entry_ts INTEGER,
                exit_price REAL,
                exit_ts INTEGER,
                pnl REAL,
                pnl_pct REAL
            )
            """
        )
        con.commit()


def load_position(con):
    cur = con.execute("SELECT id, side, size, entry_price, entry_ts, last_action_ts FROM positions WHERE id=1")
    row = cur.fetchone()
    if row:
        return {
            "id": row[0],
            "side": row[1],
            "size": row[2],
            "entry_price": row[3],
            "entry_ts": row[4],
            "last_action_ts": row[5],
        }
    return None


def save_position(con, pos):
    con.execute(
        "INSERT OR REPLACE INTO positions (id, side, size, entry_price, entry_ts, last_action_ts) VALUES (1, ?, ?, ?, ?, ?)",
        (pos["side"], pos["size"], pos["entry_price"], pos["entry_ts"], pos["last_action_ts"]),
    )


def clear_position(con):
    con.execute("DELETE FROM positions WHERE id=1")


def record_trade(con, side, size, entry_price, entry_ts, exit_price, exit_ts):
    pnl = (exit_price - entry_price) * size if side == "long" else (entry_price - exit_price) * size
    pnl_pct = pnl / (entry_price * size) if entry_price > 0 else 0
    con.execute(
        "INSERT INTO trades (side, size, entry_price, entry_ts, exit_price, exit_ts, pnl, pnl_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (side, size, entry_price, entry_ts, exit_price, exit_ts, pnl, pnl_pct),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--price", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--signal", default="data/models/lgbm_latest_signal.txt")
    p.add_argument("--db", default="data/paper_trades.db")
    args = p.parse_args()

    # Load adaptive params if available
    params_path = Path("data/models/adaptive_params.txt")
    global MIN_SIZE_PCT, SIGNAL_CONF
    if params_path.exists():
        for line in params_path.read_text(encoding="ascii", errors="ignore").splitlines():
            if line.startswith("min_size_pct="):
                MIN_SIZE_PCT = float(line.split("=", 1)[1])
            if line.startswith("signal_conf="):
                SIGNAL_CONF = float(line.split("=", 1)[1])

    # Load best strategy preferences
    best_path = Path("data/models/best_strategy.txt")
    if best_path.exists():
        for line in best_path.read_text(encoding="ascii", errors="ignore").splitlines():
            pass

    price_ts, price = get_latest_price(Path(args.price))
    size_pct = compute_position_size(Path(args.price))
    signal = get_signal(Path(args.signal), Path(args.price))

    db_path = Path(args.db)
    init_db(db_path)

    with sqlite3.connect(db_path) as con:
        pos = load_position(con)

        # avoid duplicate action on same timestamp
        if pos and pos.get("last_action_ts") == price_ts:
            print("Already acted on this timestamp")
            return

        if pos is None:
            if signal == "buy":
                if size_pct < MIN_SIZE_PCT:
                    print("SKIP BUY (size too small)")
                    return
                pos = {
                    "side": "long",
                    "size": BASE_SIZE * size_pct,
                    "entry_price": price,
                    "entry_ts": price_ts,
                    "last_action_ts": price_ts,
                }
                save_position(con, pos)
                con.commit()
                print(f"OPEN LONG {BASE_SIZE} @ {price}")
            elif signal == "sell":
                if size_pct < MIN_SIZE_PCT:
                    print("SKIP SELL (size too small)")
                    return
                pos = {
                    "side": "short",
                    "size": BASE_SIZE * size_pct,
                    "entry_price": price,
                    "entry_ts": price_ts,
                    "last_action_ts": price_ts,
                }
                save_position(con, pos)
                con.commit()
                print(f"OPEN SHORT {BASE_SIZE} @ {price}")
            else:
                print("HOLD (no position)")
            return

        # position exists
        side = pos["side"]
        if signal == "hold":
            pos["last_action_ts"] = price_ts
            save_position(con, pos)
            con.commit()
            print("HOLD (position open)")
            return

        if signal == "buy":
            if side == "long":
                if pos["size"] + ADD_SIZE <= MAX_SIZE:
                    pos["size"] += ADD_SIZE * size_pct
                    pos["last_action_ts"] = price_ts
                    save_position(con, pos)
                    con.commit()
                    print(f"ADD LONG +{ADD_SIZE} (size={pos['size']})")
                else:
                    print("LONG at max size")
            else:
                # close short
                record_trade(con, "short", pos["size"], pos["entry_price"], pos["entry_ts"], price, price_ts)
                if ALLOW_REVERSE:
                    pos = {
                        "side": "long",
                        "size": BASE_SIZE,
                        "entry_price": price,
                        "entry_ts": price_ts,
                        "last_action_ts": price_ts,
                    }
                    save_position(con, pos)
                    print("REVERSE to LONG")
                else:
                    clear_position(con)
                    print("CLOSE SHORT")
                con.commit()
            return

        if signal == "sell":
            if side == "short":
                if pos["size"] + ADD_SIZE <= MAX_SIZE:
                    pos["size"] += ADD_SIZE * size_pct
                    pos["last_action_ts"] = price_ts
                    save_position(con, pos)
                    con.commit()
                    print(f"ADD SHORT +{ADD_SIZE} (size={pos['size']})")
                else:
                    print("SHORT at max size")
            else:
                # close long
                record_trade(con, "long", pos["size"], pos["entry_price"], pos["entry_ts"], price, price_ts)
                if ALLOW_REVERSE:
                    if size_pct < MIN_SIZE_PCT:
                        clear_position(con)
                        con.commit()
                        print("CLOSE LONG (size too small to reverse)")
                        return
                    pos = {
                        "side": "short",
                        "size": BASE_SIZE * size_pct,
                        "entry_price": price,
                        "entry_ts": price_ts,
                        "last_action_ts": price_ts,
                    }
                    save_position(con, pos)
                    print("REVERSE to SHORT")
                else:
                    clear_position(con)
                    print("CLOSE LONG")
                con.commit()
            return


if __name__ == "__main__":
    main()
