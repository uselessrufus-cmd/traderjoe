import argparse
import time
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=int, default=3600)
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    py = Path(".python") / "python.exe"
    py = py.resolve()

    while True:
        # update ML signal
        subprocess.run([str(py), str(base / "predict_lgbm.py")], check=False)
        # execute paper trade step
        subprocess.run([str(py), str(base / "paper_trade.py")], check=False)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
