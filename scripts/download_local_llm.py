"""
Download ONLY the local LLM (Qwen2.5-0.5B-Instruct).

Normal users should prefer:
  - Double-click setup.bat   (packages + this model)
  - or: python scripts/easy_setup.py

This script is the single-model download path if packages are already installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_manager import ensure_model, is_ready, local_path

LOCAL_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    log("One-model download: Local LLM (Qwen2.5-0.5B-Instruct)")
    log(f"Target folder: {local_path(LOCAL_LLM_ID)}")
    path = ensure_model(LOCAL_LLM_ID, progress_callback=log)
    if is_ready(LOCAL_LLM_ID):
        log("")
        log("SUCCESS — Local LLM is ready.")
        log(f"Path: {path}")
        log("Start the app with:  run_app.bat   or   streamlit run app.py")
    else:
        raise SystemExit(f"Download finished but model is not ready: {path}")
