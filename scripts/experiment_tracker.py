import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


INDEX_COLUMNS = [
    "run_id",
    "created_at_utc",
    "source",
    "mode",
    "parent_run_id",
    "tag",
    "note",
    "objective",
    "train_return",
    "test_return",
    "test_dd",
    "win_rate",
    "run_dir",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id(prefix: str = "rl") -> str:
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="ascii")


def _copy_artifacts(run_dir: Path, artifacts: dict) -> dict:
    out = {}
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    for name, src in artifacts.items():
        if src is None:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        dst = art_dir / src_path.name
        shutil.copy2(src_path, dst)
        out[str(name)] = str(dst)
    return out


def _append_index(index_path: Path, row: dict):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    exists = index_path.exists()
    with index_path.open("a", encoding="ascii", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if not exists:
            w.writeheader()
        clean = {k: row.get(k, "") for k in INDEX_COLUMNS}
        w.writerow(clean)


def record_experiment(
    exp_root: Path,
    source: str,
    mode: str,
    params: dict,
    metrics: dict,
    artifacts: dict,
    parent_run_id: str = "",
    tag: str = "",
    note: str = "",
    run_prefix: str = "rl",
    started_at_utc: str = "",
) -> dict:
    run_id = new_run_id(run_prefix)
    created = _utc_now().isoformat()
    run_dir = exp_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_artifacts(run_dir, artifacts)
    _write_json(run_dir / "params.json", params)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "created_at_utc": created,
            "started_at_utc": started_at_utc,
            "source": source,
            "mode": mode,
            "parent_run_id": parent_run_id,
            "tag": tag,
            "note": note,
            "artifacts": copied,
        },
    )

    idx_row = {
        "run_id": run_id,
        "created_at_utc": created,
        "source": source,
        "mode": mode,
        "parent_run_id": parent_run_id,
        "tag": tag,
        "note": note,
        "objective": metrics.get("objective", ""),
        "train_return": metrics.get("train_return", ""),
        "test_return": metrics.get("test_return", ""),
        "test_dd": metrics.get("test_dd", ""),
        "win_rate": metrics.get("test_win_rate", ""),
        "run_dir": str(run_dir),
    }
    _append_index(exp_root / "index.csv", idx_row)
    return {"run_id": run_id, "run_dir": str(run_dir), "created_at_utc": created}
