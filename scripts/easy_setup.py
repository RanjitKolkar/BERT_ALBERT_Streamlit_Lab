"""
One-step setup for normal users.

Default (recommended):
  python scripts/easy_setup.py
  -> installs Python packages + downloads only the local LLM (Qwen)

Full classroom pack:
  python scripts/easy_setup.py --all
  -> packages + every model in models.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIREMENTS = ROOT / "requirements.txt"
LOCAL_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], step: str) -> None:
    log("")
    log(f"==> {step}")
    log("    " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"\nSetup failed during: {step}\n"
            f"Command exit code: {result.returncode}\n"
            "Fix the error above, then run this setup again."
        )


def install_packages() -> None:
    if not REQUIREMENTS.exists():
        raise SystemExit(f"Missing requirements file: {REQUIREMENTS}")
    run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Updating pip",
    )
    run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        "Installing app packages (one time)",
    )


def download_local_llm() -> None:
    from model_manager import ensure_model, is_ready, local_path

    log("")
    log("==> Downloading local LLM only (Qwen2.5-0.5B-Instruct)")
    log("    This is the single model download most users need.")
    path = ensure_model(LOCAL_LLM_ID, progress_callback=log)
    if not is_ready(LOCAL_LLM_ID):
        raise SystemExit(f"Download finished but model is not ready: {path}")
    log(f"Local LLM ready at: {path}")


def download_all_models() -> None:
    from model_manager import ensure_all_models, status_report

    log("")
    log("==> Downloading ALL catalogue models (larger, slower)")
    results = ensure_all_models(progress_callback=log)
    ok = sum(1 for r in results if r["ok"])
    log(f"Models ready: {ok}/{len(results)}")
    for row in status_report():
        mark = "READY" if row["ready"] else "MISSING"
        log(f"  [{mark}] {row['name']}")
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise SystemExit(
            "Some models failed. You can re-run setup later; already-downloaded models are kept."
        )


def print_done(all_models: bool) -> None:
    log("")
    log("=" * 60)
    log("SETUP COMPLETE")
    log("=" * 60)
    log("What was installed:")
    log("  1. Python packages from requirements.txt")
    if all_models:
        log("  2. All tutorial models + local LLM into local_models/")
    else:
        log("  2. Local LLM only (Qwen) into local_models/")
        log("     Tip: later run  python scripts/easy_setup.py --all  for NER/BERT too")
    log("")
    log("Next step — start the app:")
    log("  Double-click:  run_app.bat")
    log("  Or type:       streamlit run app.py")
    log("")
    log("In the app sidebar:")
    log("  - Document Context Chatbot  (ask questions on your files)")
    log("  - Tutorial Lab              (BERT / NER / ALBERT / Qwen demos)")
    log("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-step install for the Document Chatbot + Tutorial Lab."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also download every model in models.json (not only local LLM).",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="Skip package install (only download models).",
    )
    args = parser.parse_args()

    log("Easy setup starting...")
    log(f"Project folder: {ROOT}")
    log(f"Python: {sys.executable}")
    log(f"Mode: {'ALL models' if args.all else 'Local LLM only (recommended)'}")

    if not args.skip_pip:
        install_packages()
    else:
        log("Skipping pip install (--skip-pip).")

    if args.all:
        download_all_models()
    else:
        download_local_llm()

    print_done(all_models=args.all)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nSetup cancelled by user.")
