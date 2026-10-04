"""Install kit stage — ZIP + setup guide for the user's machine."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import streamlit as st

from ui_flow import callout, render_nav_buttons

APP_ROOT = Path(__file__).resolve().parent

try:
    from model_manager import (
        ensure_all_models,
        ensure_model,
        is_ready,
        local_path,
        status_report,
        all_models,
    )
except Exception as exc:  # pragma: no cover
    ensure_all_models = None
    ensure_model = None
    is_ready = lambda model_id: False  # noqa: E731
    local_path = lambda model_id: APP_ROOT / "local_models" / model_id.replace("/", "__")  # noqa: E731
    status_report = lambda: []  # noqa: E731
    all_models = lambda: []  # noqa: E731
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = None

MODEL_GUIDE: Dict[str, Dict[str, str]] = {
    "dslim/bert-base-NER": {
        "level": "Beginner",
        "best_for": "Standard English NER",
        "shortcoming": "Medium size vs DistilBERT",
        "start": "Second NER after DistilBERT NER",
    },
    "elastic/distilbert-base-uncased-finetuned-conll03-english": {
        "level": "Beginner ★",
        "best_for": "First NER — small & fast",
        "shortcoming": "Slightly less accurate than large NER",
        "start": "Best NER to begin",
    },
    "Jean-Baptiste/roberta-large-ner-english": {
        "level": "Advanced",
        "best_for": "Strong English NER compare",
        "shortcoming": "Large RAM/disk — install last",
        "start": "Only after small NER works",
    },
    "Davlan/xlm-roberta-base-ner-hrl": {
        "level": "Intermediate",
        "best_for": "Multilingual NER",
        "shortcoming": "Heavier; skip if English-only",
        "start": "When you need other languages",
    },
    "google-bert/bert-base-uncased": {
        "level": "Beginner",
        "best_for": "Classic fill-mask teaching",
        "shortcoming": "Larger than DistilBERT",
        "start": "After DistilBERT encoder",
    },
    "google-bert/bert-base-cased": {
        "level": "Beginner",
        "best_for": "Capitals matter",
        "shortcoming": "Same size as BERT uncased",
        "start": "Optional twin of uncased",
    },
    "distilbert/distilbert-base-uncased": {
        "level": "Beginner ★",
        "best_for": "Fastest encoder demo",
        "shortcoming": "Slightly less capacity than BERT",
        "start": "Best encoder to begin",
    },
    "FacebookAI/roberta-base": {
        "level": "Intermediate",
        "best_for": "Strong encoder baseline",
        "shortcoming": "Uses <mask> not [MASK]",
        "start": "After BERT/DistilBERT",
    },
    "albert/albert-base-v2": {
        "level": "Intermediate",
        "best_for": "Parameter-sharing story",
        "shortcoming": "Can surprise beginners",
        "start": "Teaching ALBERT, not first install",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "level": "Beginner ★",
        "best_for": "Local chat generation, no API key",
        "shortcoming": "Small LLM — limited reasoning",
        "start": "Best local LLM; Chat works without it",
    },
}

BEGINNER_IDS = [
    "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "distilbert/distilbert-base-uncased",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
CHAT_IDS = ["Qwen/Qwen2.5-0.5B-Instruct"]

KIT_FILES = [
    "SETUP_GUIDE.md",
    "INSTALL.md",
    "USER_GUIDE.md",
    "README.md",
    "requirements.txt",
    "requirements-lab.txt",
    "models.json",
    "model_manager.py",
    "runtime_memory.py",
    "setup.bat",
    "setup_all_models.bat",
    "run_app.bat",
    "app.py",
    "app1.py",
    "ui_flow.py",
    "chat_module.py",
    "learn_module.py",
    "install_module.py",
    "tutorial_lab.py",
    "scripts/easy_setup.py",
    "scripts/download_local_llm.py",
]


def guide_for(model_id: str) -> Dict[str, str]:
    return MODEL_GUIDE.get(
        model_id,
        {
            "level": "See catalogue",
            "best_for": "Listed in models.json",
            "shortcoming": "See Hugging Face model card",
            "start": "Install when needed",
        },
    )


def _add_path_to_zip(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.is_file():
        zf.write(path, arcname)
        return
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                rel = child.relative_to(path)
                zf.write(child, f"{arcname}/{rel.as_posix()}")


def build_setup_kit_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in KIT_FILES:
            path = APP_ROOT / rel
            if path.exists():
                _add_path_to_zip(zf, path, f"AI_Document_Lab/{rel}")
        how_to = (
            "AI Document Lab — unzip this folder, then:\r\n"
            "1. Read SETUP_GUIDE.md\r\n"
            "2. Double-click setup.bat  (packages + Local LLM)\r\n"
            "3. Double-click run_app.bat\r\n"
            "4. Follow the top journey: Start → Install → Learn → Practice → Chat\r\n"
        )
        zf.writestr("AI_Document_Lab/HOW_TO_START.txt", how_to)
    return buf.getvalue()


def build_models_zip(model_ids: Sequence[str], folder_name: str = "beginner_models") -> Optional[bytes]:
    paths = []
    for mid in model_ids:
        p = Path(local_path(mid))
        if p.exists() and any(p.iterdir()):
            paths.append((mid, p))
    if not paths:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        readme = (
            "Place extracted folders into your project local_models/ directory.\r\n"
            "Folder names must match model_manager layout (org__name).\r\n"
        )
        zf.writestr(f"{folder_name}/README_EXTRACT.txt", readme)
        for _mid, p in paths:
            arc_root = f"{folder_name}/{p.name}"
            for child in p.rglob("*"):
                if child.is_file():
                    zf.write(child, f"{arc_root}/{child.relative_to(p).as_posix()}")
    return buf.getvalue()


def folder_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def run_download_ids(ids: List[str], label: str) -> None:
    if ensure_model is None:
        st.error("model_manager unavailable. Install packages first (setup.bat).")
        return
    status = st.empty()
    bar = st.progress(0.0)
    ok = 0
    errors = []
    for i, mid in enumerate(ids, start=1):
        status.write(f"{label}: [{i}/{len(ids)}] {mid}")
        try:
            ensure_model(mid, progress_callback=lambda m: status.write(m))
            ok += 1
        except Exception as exc:
            errors.append(f"{mid}: {exc}")
        bar.progress(i / len(ids))
    bar.progress(1.0)
    if ok == len(ids):
        st.success(f"{label} complete — {ok}/{len(ids)} ready on this machine.")
    else:
        st.warning(f"{label} finished with issues — {ok}/{len(ids)} ready.")
        for e in errors:
            st.error(e)


def render_install() -> None:
    st.markdown("## Step 2 · Install kit — files on **your** machine")
    st.caption(
        "Primary path: download a **ZIP + setup guide**, unzip on the PC, run setup.bat. "
        "Optional: pull models here if this machine already has the project open."
    )

    st.markdown("### A · Download Setup Kit (ZIP)")
    callout(
        "<b>Best for new users / students:</b> get the kit on your disk, unzip, follow the guide. "
        "Models install locally via setup scripts — not forced through the browser."
    )

    guide_path = APP_ROOT / "SETUP_GUIDE.md"
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            kit_bytes = build_setup_kit_zip()
            st.download_button(
                label="Download Lab Setup Kit (ZIP)",
                data=kit_bytes,
                file_name="AI_Document_Lab_Setup_Kit.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                help="App + scripts + guide. No multi-GB model weights.",
            )
            st.caption(f"Kit size ≈ {len(kit_bytes) / 1024:.0f} KB")
        except Exception as exc:
            st.error(f"Could not build kit ZIP: {exc}")
    with c2:
        if guide_path.exists():
            st.download_button(
                label="Setup guide only (MD)",
                data=guide_path.read_bytes(),
                file_name="SETUP_GUIDE.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.warning("SETUP_GUIDE.md missing")
    with c3:
        install_md = APP_ROOT / "INSTALL.md"
        if install_md.exists():
            st.download_button(
                label="INSTALL.md",
                data=install_md.read_bytes(),
                file_name="INSTALL.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with st.expander("What is inside the Setup Kit ZIP?", expanded=False):
        st.markdown(
            """
- `SETUP_GUIDE.md` + `HOW_TO_START.txt`
- `setup.bat` / `setup_all_models.bat` / `run_app.bat`
- `requirements.txt`, `models.json`, `model_manager.py`
- Full app modules
- `scripts/easy_setup.py`

**Not included:** heavy `local_models/` weights (download on the target PC, or export below).
"""
        )

    st.markdown(
        """
**After you download the ZIP**

1. Unzip on **your** PC (example: `C:\\AI_Document_Lab`)
2. Open `SETUP_GUIDE.md`
3. Double-click **setup.bat** (packages + Qwen)
4. Double-click **run_app.bat**
5. Continue: **Learn → Practice → Chat**
"""
    )

    st.markdown("---")
    st.markdown("### B · Status on **this** machine")
    if _IMPORT_ERROR:
        st.error(f"Model manager issue: {_IMPORT_ERROR}")
    else:
        report = status_report()
        ready = sum(1 for r in report if r["ready"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Models ready", f"{ready}/{len(report)}")
        lm = APP_ROOT / "local_models"
        m2.metric("local_models size", f"{folder_size_mb(lm):.0f} MB")
        m3.metric("Setup guide", "Yes" if guide_path.exists() else "Missing")
        st.dataframe(
            [
                {
                    "Name": r["name"],
                    "ID": r["id"],
                    "Status": "READY" if r["ready"] else "missing",
                    "Level": guide_for(r["id"])["level"],
                    "Start with": guide_for(r["id"])["start"],
                }
                for r in report
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### C · Export models already on this PC (optional ZIP)")
    st.caption(
        "Copy weights to another lab PC without re-downloading. Large zips take time and disk."
    )
    beg_ready = [mid for mid in BEGINNER_IDS if is_ready(mid)]
    if beg_ready:
        sizes = sum(folder_size_mb(Path(local_path(m))) for m in beg_ready)
        st.write(f"Beginner models ready: **{len(beg_ready)}/3** · ~**{sizes:.0f} MB**")
        if st.button("Build beginner models ZIP", use_container_width=True):
            with st.spinner("Zipping beginner models…"):
                data = build_models_zip(BEGINNER_IDS, "beginner_local_models")
            if data is None:
                st.warning("No beginner model folders found to zip.")
            else:
                st.session_state["beginner_models_zip"] = data
                st.success(f"ZIP ready ({len(data) / (1024 * 1024):.1f} MB).")
        if st.session_state.get("beginner_models_zip"):
            st.download_button(
                "Download beginner_models.zip",
                data=st.session_state["beginner_models_zip"],
                file_name="AI_Document_Lab_beginner_models.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
    else:
        callout(
            "No beginner models on disk yet. Use section D, or run setup.bat after unzipping the kit.",
            kind="warn",
        )

    st.markdown("---")
    st.markdown("### D · Optional: download models into **this** project folder")
    st.caption("Only if you already run from the full project and have internet.")
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("Fetch Chat pack (Qwen)", use_container_width=True):
            with st.spinner("Downloading Qwen…"):
                run_download_ids(CHAT_IDS, "Chat pack")
                st.rerun()
    with d2:
        if st.button("Fetch Beginner pack", type="primary", use_container_width=True):
            with st.spinner("Downloading beginner pack…"):
                run_download_ids(BEGINNER_IDS, "Beginner pack")
                st.rerun()
    with d3:
        if st.button("Fetch everything", use_container_width=True):
            if ensure_all_models is None:
                st.error("ensure_all_models unavailable")
            else:
                status = st.empty()
                with st.spinner("Downloading full catalogue…"):
                    results = ensure_all_models(progress_callback=lambda m: status.write(m))
                ok = sum(1 for r in results if r["ok"])
                st.success(f"{ok}/{len(results)} ready")
                st.rerun()

    with st.expander("Model guide (when to install)", expanded=False):
        rows = []
        for item in all_models() or []:
            mid = item["id"]
            g = guide_for(mid)
            rows.append(
                {
                    "Name": item.get("name", mid),
                    "Level": g["level"],
                    "Best for": g["best_for"],
                    "Shortcoming": g["shortcoming"],
                    "Start": g["start"],
                    "Ready": "Yes" if is_ready(mid) else "No",
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Setup guide (preview)", expanded=False):
        if guide_path.exists():
            st.markdown(guide_path.read_text(encoding="utf-8")[:6000])
        else:
            st.warning("SETUP_GUIDE.md not found.")

    st.markdown("---")
    render_nav_buttons(
        prev_key="start",
        next_key="learn",
        next_label="Continue to Learn →",
        mark_done_key="install",
    )
