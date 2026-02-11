import argparse
import time
import subprocess
from pathlib import Path


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
    while time.time() - start < args.minutes * 60:
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
        time.sleep(args.cycle)


if __name__ == "__main__":
    main()
