"""Preload every model from models.json into local_models/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_manager import ensure_all_models, status_report


def log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    log("Preloading all catalogue models into local_models/ ...")
    results = ensure_all_models(progress_callback=log)
    log("")
    log("=== Summary ===")
    ok = sum(1 for r in results if r["ok"])
    failed = [r for r in results if not r["ok"]]
    log(f"Succeeded: {ok}/{len(results)}")
    if failed:
        log("Failures:")
        for item in failed:
            log(f" - {item['id']}: {item['error']}")
    log("")
    log("=== Local status ===")
    for row in status_report():
        mark = "READY" if row["ready"] else "MISSING"
        log(f"[{mark}] {row['name']} -> {row['path']}")
