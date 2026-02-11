import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--out", default="data/models/paper_summary.txt")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print("No paper trades yet")
        return

    with sqlite3.connect(db) as con:
        trades = pd.read_sql_query("SELECT * FROM trades", con)
        pos = pd.read_sql_query("SELECT * FROM positions", con)

    total = len(trades)
    win = (trades["pnl"] > 0).sum() if total else 0
    loss = (trades["pnl"] < 0).sum() if total else 0
    avg = trades["pnl_pct"].mean() if total else 0

    summary = (
        f"Total closed trades: {total}\n"
        f"Win rate: {(win/total if total else 0):.2%}\n"
        f"Avg trade: {avg:.2%}\n"
        f"Wins: {win} Losses: {loss}\n"
    )

    if not pos.empty:
        row = pos.iloc[0]
        summary += (
            f"Open position: {row['side']} size={row['size']} entry={row['entry_price']}\n"
        )
    else:
        summary += "Open position: none\n"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(summary, encoding="ascii")
    print(summary)


if __name__ == "__main__":
    main()
