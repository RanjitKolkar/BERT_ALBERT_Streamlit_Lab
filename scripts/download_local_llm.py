from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_manager import ensure_all_models


def log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    print("Preparing ALL catalogue models (including local LLM)...")
    results = ensure_all_models(progress_callback=log)
    ok = sum(1 for r in results if r["ok"])
    print(f"Done. {ok}/{len(results)} models ready.")
