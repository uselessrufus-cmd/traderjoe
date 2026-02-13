import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--delete-db", action="store_true")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    deleted = []

    if models_dir.exists():
        for f in models_dir.glob("*"):
            if f.is_file():
                f.unlink(missing_ok=True)
                deleted.append(str(f))

    if args.delete_db:
        for db in [Path("data/ai_memory.db"), Path("data/paper_trades.db")]:
            if db.exists():
                db.unlink(missing_ok=True)
                deleted.append(str(db))

    print("Reset complete.")
    print(f"Deleted files: {len(deleted)}")
    for d in deleted:
        print(f"- {d}")


if __name__ == "__main__":
    main()
