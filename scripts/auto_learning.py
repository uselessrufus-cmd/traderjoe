import argparse
import time
import subprocess
from pathlib import Path
import math
import sys
from pathlib import Path as _Path

sys.path.append(str((_Path(__file__).resolve().parent)))
from progress_bar import render_progress


def run(cmd):
    subprocess.run(cmd, check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=15)
    p.add_argument("--cycle", type=int, default=300)
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    py = (Path(".python") / "python.exe").resolve()

    start = time.time()
    total_cycles = max(1, int(math.ceil((args.minutes * 60) / max(1, args.cycle))))
    cycle_idx = 0
    while time.time() - start < args.minutes * 60:
        cycle_idx += 1
        print(f"Cycle {cycle_idx}/{total_cycles}")
        # historical simulation + summary
        run([str(py), str(base / "sim_train_loop.py")])
        run([str(py), str(base / "sim_summary.py")])
        # strategy exploration (heavier batch)
        run([str(py), str(base / "strategy_explore.py"), "--max-combos", "600", "--sleep", "0.02"])
        # historical paper-trading simulation
        run([str(py), str(base / "historical_sim.py")])
        # train + predict (ML)
        run([str(py), str(base / "ml_features.py")])
        run([str(py), str(base / "train_lgbm.py")])
        run([str(py), str(base / "predict_lgbm.py")])
        # adaptive params
        run([str(py), str(base / "adaptive_params.py")])
        print(f"Overall {render_progress(cycle_idx, total_cycles)}")
        time.sleep(args.cycle)


if __name__ == "__main__":
    main()
