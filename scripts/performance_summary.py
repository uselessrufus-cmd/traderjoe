import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/ai_memory.db")
    p.add_argument("--out", default="data/models/performance_summary.txt")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print("No database found")
        return

    with sqlite3.connect(db) as con:
        try:
            df = pd.read_sql_query("SELECT * FROM signal_history", con)
        except Exception:
            print("No signal_history yet")
            return

    if df.empty:
        print("No signal_history yet")
        return

    total = len(df)
    win = (df["ret"] > 0).sum()
    loss = (df["ret"] < 0).sum()
    avg = df["ret"].mean()
    win_rate = win / total if total else 0

    summary = (
        f"Total signals: {total}\n"
        f"Win rate: {win_rate:.2%}\n"
        f"Avg return: {avg:.2%}\n"
        f"Wins: {win} Losses: {loss}\n"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(summary, encoding="ascii")
    print(summary)


if __name__ == "__main__":
    main()
