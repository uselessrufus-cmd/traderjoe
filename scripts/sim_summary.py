import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/models/sim_report.txt")
    p.add_argument("--out", default="data/models/sim_summary.txt")
    args = p.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print("No sim_report found")
        return

    lines = inp.read_text(encoding="ascii", errors="ignore").splitlines()
    accs = []
    for line in lines:
        m = re.search(r"acc=([0-9.]+)%", line)
        if m:
            accs.append(float(m.group(1)))

    if not accs:
        print("No accuracies found")
        return

    arr = np.array(accs)
    mean = float(arr.mean())
    med = float(np.median(arr))
    std = float(arr.std())
    mn = float(arr.min())
    mx = float(arr.max())
    stability = mean - std

    summary = (
        f"Windows: {len(arr)}\n"
        f"Mean acc: {mean:.2f}%\n"
        f"Median acc: {med:.2f}%\n"
        f"Std dev: {std:.2f}%\n"
        f"Min acc: {mn:.2f}%\n"
        f"Max acc: {mx:.2f}%\n"
        f"Stability score (mean-std): {stability:.2f}%\n"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(summary, encoding="ascii")

    # Append to history
    hist_path = Path("data/models/sim_history.csv")
    row = {
        "windows": len(arr),
        "mean": mean,
        "median": med,
        "std": std,
        "min": mn,
        "max": mx,
        "stability": stability,
    }
    if hist_path.exists():
        prev = pd.read_csv(hist_path)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(hist_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(hist_path, index=False)
    print(summary)


if __name__ == "__main__":
    main()
