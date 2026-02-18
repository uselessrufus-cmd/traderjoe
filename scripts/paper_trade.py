import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import random
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))
from ml_features import build_features

# Simple paper trader config
BASE_SIZE = 1.0
ADD_SIZE = 0.5
MAX_SIZE = 1.0
ALLOW_REVERSE = True
MIN_SIZE_PCT = 0.2
SIGNAL_CONF = 0.4
RL_ALPHA = 0.2
RL_GAMMA = 0.95
RL_EPSILON = 0.1
RL_WARMUP = 30
RL_ACTIONS = ("buy", "sell", "hold")


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


def compute_market_state(csv_path: Path, pos_side: str | None) -> str:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df = df[["close"]].dropna()
    if len(df) < 240:
        return f"flat|lowvol|{pos_side or 'none'}"

    close = df["close"]
    sma_fast = close.rolling(50).mean()
    sma_slow = close.rolling(200).mean()
    trend = "bull" if sma_fast.iloc[-1] > sma_slow.iloc[-1] else "bear"
    ret = close.pct_change()
    mom_raw = ret.tail(12).mean()
    vol = ret.tail(48).std()
    mom = 0.0 if pd.isna(mom_raw) else float(mom_raw)
    vol_val = 0.0 if pd.isna(vol) else float(vol)
    if mom > 0.001:
        mom_bucket = "up"
    elif mom < -0.001:
        mom_bucket = "down"
    else:
        mom_bucket = "flat"
    vol_bucket = "highvol" if vol_val > 0.02 else "lowvol"
    side_bucket = pos_side if pos_side in {"long", "short"} else "none"
    return f"{trend}:{mom_bucket}|{vol_bucket}|{side_bucket}"


def _bucket_mfi(v: float) -> str:
    if v < 30:
        return "low"
    if v > 70:
        return "high"
    return "mid"


def _bucket_macd(v: float) -> str:
    return "up" if v >= 0 else "down"


def compute_policy_state(csv_path: Path, pos_side: str | None) -> str:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
        feat = build_features(df).dropna().reset_index(drop=True)
        if len(feat) == 0:
            side_bucket = pos_side if pos_side in {"long", "short"} else "none"
            return f"range|mfi:mid|macd:down|ob:mix|pos:{side_bucket}"
        row = feat.iloc[-1]
        trend = "bull" if int(row.get("trend_up", 0)) == 1 else "bear"
        mfi = _bucket_mfi(float(row.get("mfi_14", 50.0)))
        macd = _bucket_macd(float(row.get("macd_hist", 0.0)))
        ob_bull = int(row.get("ob_bull_recent", 0))
        ob_bear = int(row.get("ob_bear_recent", 0))
        ob = "bull" if ob_bull and not ob_bear else "bear" if ob_bear and not ob_bull else "mix"
        side_bucket = pos_side if pos_side in {"long", "short"} else "none"
        return f"{trend}|mfi:{mfi}|macd:{macd}|ob:{ob}|pos:{side_bucket}"
    except Exception:
        side_bucket = pos_side if pos_side in {"long", "short"} else "none"
        return f"range|mfi:mid|macd:down|ob:mix|pos:{side_bucket}"


def select_champion_action(policy_state: str, champions_root: Path) -> tuple[str, str]:
    regime = str(policy_state).split("|", 1)[0]
    order = []
    if regime in {"bull", "bear", "range"}:
        order.append(regime)
    order.append("global")
    for reg in order:
        pol = champions_root / reg / "policy.csv"
        if not pol.exists():
            continue
        try:
            pdf = pd.read_csv(pol)
            if len(pdf) == 0 or "state" not in pdf.columns or "action" not in pdf.columns:
                continue
            exact = pdf[pdf["state"] == policy_state]
            if not exact.empty:
                return str(exact.iloc[0]["action"]).lower(), f"champion:{reg}:exact"
            # fallback: same rationale context, different position side
            if "|pos:" in policy_state:
                rk = policy_state.split("|pos:", 1)[0] + "|pos:"
                cands = pdf[pdf["state"].astype(str).str.startswith(rk)]
                if not cands.empty:
                    if "q" in cands.columns:
                        cands = cands.sort_values("q", ascending=False)
                    return str(cands.iloc[0]["action"]).lower(), f"champion:{reg}:context"
        except Exception:
            continue
    return "hold", "none"


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
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_q (
                state TEXT NOT NULL,
                action TEXT NOT NULL,
                q REAL NOT NULL DEFAULT 0,
                updates INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (state, action)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_pending (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state TEXT,
                action TEXT
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
    return pnl_pct


def _rl_get_q(con, state: str, action: str) -> float:
    cur = con.execute("SELECT q FROM rl_q WHERE state=? AND action=?", (state, action))
    row = cur.fetchone()
    return float(row[0]) if row else 0.0


def _rl_set_q(con, state: str, action: str, q_value: float):
    con.execute(
        """
        INSERT INTO rl_q(state, action, q, updates)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(state, action) DO UPDATE SET
            q=excluded.q,
            updates=rl_q.updates + 1
        """,
        (state, action, q_value),
    )


def _rl_get_total_updates(con) -> int:
    cur = con.execute("SELECT COALESCE(SUM(updates), 0) FROM rl_q")
    row = cur.fetchone()
    return int(row[0] or 0)


def _rl_get_pending(con):
    cur = con.execute("SELECT state, action FROM rl_pending WHERE id=1")
    row = cur.fetchone()
    if row:
        return {"state": row[0], "action": row[1]}
    return None


def _rl_set_pending(con, state: str, action: str):
    con.execute(
        """
        INSERT INTO rl_pending(id, state, action)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET state=excluded.state, action=excluded.action
        """,
        (state, action),
    )


def _rl_clear_pending(con):
    con.execute("DELETE FROM rl_pending WHERE id=1")


def _rl_choose_action(con, state: str, base_signal: str, epsilon: float, warmup_updates: int):
    total_updates = _rl_get_total_updates(con)
    if total_updates < warmup_updates:
        return base_signal, total_updates, "warmup"

    if random.random() < epsilon:
        return random.choice(RL_ACTIONS), total_updates, "explore"

    q_map = {a: _rl_get_q(con, state, a) for a in RL_ACTIONS}
    best = max(q_map, key=lambda a: q_map[a])
    return best, total_updates, "exploit"


def _rl_update(con, reward: float, next_state: str):
    pending = _rl_get_pending(con)
    if pending is None:
        return
    state = pending["state"]
    action = pending["action"]
    current_q = _rl_get_q(con, state, action)
    next_max = max(_rl_get_q(con, next_state, a) for a in RL_ACTIONS)
    target = reward + RL_GAMMA * next_max
    new_q = current_q + RL_ALPHA * (target - current_q)
    _rl_set_q(con, state, action, new_q)
    _rl_clear_pending(con)


def _write_rl_status(
    path: Path,
    enabled: bool,
    state: str,
    base_signal: str,
    rl_signal: str,
    final_signal: str,
    mode: str,
    updates: int,
    champion_enabled: bool = False,
    champion_signal: str = "",
    champion_source: str = "",
    policy_state: str = "",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"enabled={enabled}\n"
            f"state={state}\n"
            f"policy_state={policy_state}\n"
            f"base_signal={base_signal}\n"
            f"champion_enabled={champion_enabled}\n"
            f"champion_signal={champion_signal}\n"
            f"champion_source={champion_source}\n"
            f"rl_signal={rl_signal}\n"
            f"final_signal={final_signal}\n"
            f"mode={mode}\n"
            f"updates={updates}\n"
            f"alpha={RL_ALPHA}\n"
            f"gamma={RL_GAMMA}\n"
            f"epsilon={RL_EPSILON}\n"
            f"warmup={RL_WARMUP}\n"
        ),
        encoding="ascii",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--price", default="data/bitstamp/ohlc/bitstamp_1h.csv")
    p.add_argument("--signal", default="data/models/lgbm_latest_signal.txt")
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--rl", action="store_true")
    p.add_argument("--champions", action="store_true")
    p.add_argument("--champions-root", default="data/models/champions")
    p.add_argument("--epsilon", type=float, default=RL_EPSILON)
    p.add_argument("--warmup-updates", type=int, default=RL_WARMUP)
    p.add_argument("--rl-status", default="data/models/rl_status.txt")
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

    price_path = Path(args.price)
    price_ts, price = get_latest_price(price_path)
    size_pct = compute_position_size(price_path)
    base_signal = get_signal(Path(args.signal), price_path)

    db_path = Path(args.db)
    init_db(db_path)

    with sqlite3.connect(db_path) as con:
        pos = load_position(con)
        state = compute_market_state(Path(args.price), pos["side"] if pos else None)
        policy_state = compute_policy_state(price_path, pos["side"] if pos else None)
        champion_signal = ""
        champion_source = "disabled"
        model_signal = base_signal
        if args.champions:
            champion_signal, champion_source = select_champion_action(policy_state, Path(args.champions_root))
            if champion_source != "none":
                model_signal = champion_signal
        signal = model_signal
        rl_signal = model_signal
        mode = "disabled"
        updates = _rl_get_total_updates(con)
        if args.rl:
            rl_signal, updates, mode = _rl_choose_action(con, state, model_signal, args.epsilon, args.warmup_updates)
            signal = rl_signal
        _write_rl_status(
            Path(args.rl_status),
            args.rl,
            state,
            base_signal,
            rl_signal,
            signal,
            mode,
            updates,
            champion_enabled=args.champions,
            champion_signal=champion_signal,
            champion_source=champion_source,
            policy_state=policy_state,
        )

        # avoid duplicate action on same timestamp
        if pos and pos.get("last_action_ts") == price_ts:
            print("Already acted on this timestamp")
            return

        if pos is None:
            if signal == "buy":
                if size_pct < MIN_SIZE_PCT:
                    print("SKIP BUY (size too small)")
                    return
                size = BASE_SIZE * size_pct
                pos = {
                    "side": "long",
                    "size": size,
                    "entry_price": price,
                    "entry_ts": price_ts,
                    "last_action_ts": price_ts,
                }
                save_position(con, pos)
                if args.rl:
                    _rl_set_pending(con, state, "buy")
                con.commit()
                print(f"OPEN LONG {size:.4f} @ {price}")
            elif signal == "sell":
                if size_pct < MIN_SIZE_PCT:
                    print("SKIP SELL (size too small)")
                    return
                size = BASE_SIZE * size_pct
                pos = {
                    "side": "short",
                    "size": size,
                    "entry_price": price,
                    "entry_ts": price_ts,
                    "last_action_ts": price_ts,
                }
                save_position(con, pos)
                if args.rl:
                    _rl_set_pending(con, state, "sell")
                con.commit()
                print(f"OPEN SHORT {size:.4f} @ {price}")
            else:
                if args.rl:
                    _rl_set_pending(con, state, "hold")
                print("HOLD (no position)")
            return

        # position exists
        side = pos["side"]
        if signal == "hold":
            pos["last_action_ts"] = price_ts
            save_position(con, pos)
            if args.rl:
                _rl_set_pending(con, state, "hold")
            con.commit()
            print("HOLD (position open)")
            return

        if signal == "buy":
            if side == "long":
                add_size = ADD_SIZE * size_pct
                if pos["size"] + add_size <= MAX_SIZE:
                    pos["size"] += add_size
                    pos["last_action_ts"] = price_ts
                    save_position(con, pos)
                    if args.rl:
                        _rl_set_pending(con, state, "buy")
                    con.commit()
                    print(f"ADD LONG +{add_size:.4f} (size={pos['size']:.4f})")
                else:
                    print("LONG at max size")
            else:
                # close short
                reward = record_trade(con, "short", pos["size"], pos["entry_price"], pos["entry_ts"], price, price_ts)
                if args.rl:
                    next_state = compute_market_state(Path(args.price), "long" if ALLOW_REVERSE else None)
                    _rl_update(con, reward, next_state)
                if ALLOW_REVERSE:
                    if size_pct < MIN_SIZE_PCT:
                        clear_position(con)
                        con.commit()
                        print("CLOSE SHORT (size too small to reverse)")
                        return
                    size = BASE_SIZE * size_pct
                    pos = {
                        "side": "long",
                        "size": size,
                        "entry_price": price,
                        "entry_ts": price_ts,
                        "last_action_ts": price_ts,
                    }
                    save_position(con, pos)
                    if args.rl:
                        _rl_set_pending(con, state, "buy")
                    print(f"REVERSE to LONG (size={size:.4f})")
                else:
                    clear_position(con)
                    print("CLOSE SHORT")
                con.commit()
            return

        if signal == "sell":
            if side == "short":
                add_size = ADD_SIZE * size_pct
                if pos["size"] + add_size <= MAX_SIZE:
                    pos["size"] += add_size
                    pos["last_action_ts"] = price_ts
                    save_position(con, pos)
                    if args.rl:
                        _rl_set_pending(con, state, "sell")
                    con.commit()
                    print(f"ADD SHORT +{add_size:.4f} (size={pos['size']:.4f})")
                else:
                    print("SHORT at max size")
            else:
                # close long
                reward = record_trade(con, "long", pos["size"], pos["entry_price"], pos["entry_ts"], price, price_ts)
                if args.rl:
                    next_state = compute_market_state(Path(args.price), "short" if ALLOW_REVERSE else None)
                    _rl_update(con, reward, next_state)
                if ALLOW_REVERSE:
                    if size_pct < MIN_SIZE_PCT:
                        clear_position(con)
                        con.commit()
                        print("CLOSE LONG (size too small to reverse)")
                        return
                    size = BASE_SIZE * size_pct
                    pos = {
                        "side": "short",
                        "size": size,
                        "entry_price": price,
                        "entry_ts": price_ts,
                        "last_action_ts": price_ts,
                    }
                    save_position(con, pos)
                    if args.rl:
                        _rl_set_pending(con, state, "sell")
                    print(f"REVERSE to SHORT (size={size:.4f})")
                else:
                    clear_position(con)
                    print("CLOSE LONG")
                con.commit()
            return


if __name__ == "__main__":
    main()
