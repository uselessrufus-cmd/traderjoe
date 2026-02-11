import argparse
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=int, default=60)
    p.add_argument("--intervals", default="1m,5m,15m,1h,4h,12h,1d,1w,1M")
    args = p.parse_args()

    script = Path(__file__).resolve().parent / "merge_data.py"
    while True:
        subprocess.run(
            ["python", str(script), "--intervals", args.intervals],
            check=False,
        )
        time.sleep(args.every)


if __name__ == "__main__":
    main()
