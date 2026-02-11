import argparse
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="data/models/sim_summary.txt")
    p.add_argument("--perf", default="data/models/performance_summary.txt")
    p.add_argument("--out", default="data/models/adaptive_params.txt")
    args = p.parse_args()

    # Default thresholds
    min_size_pct = 0.2
    signal_conf = 0.4

    # Use simulation stability if available
    sim_path = Path(args.sim)
    if sim_path.exists():
        txt = sim_path.read_text(encoding="ascii", errors="ignore")
        m = re.search(r"Stability score \(mean-std\): ([0-9.]+)%", txt)
        if m:
            stability = float(m.group(1))
            # Higher stability -> lower confidence threshold (trade more)
            if stability > 55:
                signal_conf = 0.35
            elif stability > 50:
                signal_conf = 0.4
            else:
                signal_conf = 0.5

    # Use recent performance if available
    perf_path = Path(args.perf)
    if perf_path.exists():
        txt = perf_path.read_text(encoding="ascii", errors="ignore")
        m = re.search(r"Win rate: ([0-9.]+)%", txt)
        if m:
            win_rate = float(m.group(1))
            if win_rate < 40:
                min_size_pct = 0.4  # be more selective
            elif win_rate > 55:
                min_size_pct = 0.15

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"min_size_pct={min_size_pct}\n"
        f"signal_conf={signal_conf}\n",
        encoding="ascii",
    )
    print(out.read_text(encoding="ascii"))


if __name__ == "__main__":
    main()
