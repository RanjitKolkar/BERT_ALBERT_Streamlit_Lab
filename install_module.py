"""Install yourself — guided one-click model downloads with clear recommendations."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

try:
    from model_manager import (
        CATALOGUE,
        ensure_all_models,
        ensure_model,
        is_ready,
        local_path,
        status_report,
        all_models,
    )
except Exception as exc:  # pragma: no cover
    CATALOGUE = {}
    ensure_all_models = None
    ensure_model = None
    is_ready = lambda model_id: False  # noqa: E731
    local_path = lambda model_id: model_id  # noqa: E731
    status_report = lambda: []  # noqa: E731
    all_models = lambda: []  # noqa: E731
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = None

# Human guidance layered on models.json
MODEL_GUIDE: Dict[str, Dict[str, str]] = {
    "dslim/bert-base-NER": {
        "level": "Beginner ★★★",
        "best_for": "Standard English NER classroom demo",
        "shortcoming": "English-oriented; medium size vs DistilBERT",
        "start": "Good second NER after DistilBERT NER",
    },
    "elastic/distilbert-base-uncased-finetuned-conll03-english": {
        "level": "Beginner ★★★★★",
        "best_for": "First NER install — small & fast",
        "shortcoming": "Slightly less accurate than larger NER models",
        "start": "**Best NER to begin**",
    },
    "Jean-Baptiste/roberta-large-ner-english": {
        "level": "Advanced",
        "best_for": "Strong English NER comparisons",
        "shortcoming": "Large download & high RAM — install last",
        "start": "Only after small NER models work",
    },
    "Davlan/xlm-roberta-base-ner-hrl": {
        "level": "Intermediate",
        "best_for": "Multilingual NER",
        "shortcoming": "Heavier; overkill for English-only labs",
        "start": "When you need non-English entities",
    },
    "google-bert/bert-base-uncased": {
        "level": "Beginner ★★★★",
        "best_for": "Classic fill-mask / encoder teaching",
        "shortcoming": "Larger than DistilBERT",
        "start": "Core encoder after DistilBERT",
    },
    "google-bert/bert-base-cased": {
        "level": "Beginner ★★★",
        "best_for": "When capital letters matter",
        "shortcoming": "Same footprint as BERT uncased",
        "start": "Optional twin of uncased BERT",
    },
    "distilbert/distilbert-base-uncased": {
        "level": "Beginner ★★★★★",
        "best_for": "Fastest encoder demo",
        "shortcoming": "Distilled — a bit less capacity than BERT",
        "start": "**Best encoder to begin**",
    },
    "FacebookAI/roberta-base": {
        "level": "Intermediate",
        "best_for": "Strong base encoder; compare with BERT",
        "shortcoming": "Uses `<mask>` not `[MASK]`",
        "start": "After BERT/DistilBERT",
    },
    "albert/albert-base-v2": {
        "level": "Intermediate",
        "best_for": "Parameter-sharing encoder story",
        "shortcoming": "Behaviour can surprise beginners",
        "start": "Teaching ALBERT idea, not first install",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "level": "Beginner ★★★★ (for chat generation)",
        "best_for": "Local document Q&A without API key",
        "shortcoming": "Small LLM — limited reasoning; needs RAM",
        "start": "**Best local LLM to begin**; Chat still works without it",
    },
}

BEGINNER_IDS = [
    "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "distilbert/distilbert-base-uncased",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

CHAT_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


def guide_for(model_id: str) -> Dict[str, str]:
    return MODEL_GUIDE.get(
        model_id,
        {
            "level": "See catalogue",
            "best_for": "Listed in models.json",
            "shortcoming": "Check model card on Hugging Face",
            "start": "Install when needed",
        },
    )


def render_recommendation_board() -> None:
    st.subheader("What should I install first?")
    st.markdown(
        """
| Goal | One-click choice | Why |
|---|---|---|
| **Just try Chat** | Nothing required | Extractive mode needs **no** neural model |
| **Chat + fluent answers** | **Chat pack** (Qwen only) | One small local LLM |
| **Classroom starter** | **Beginner pack** | DistilBERT NER + DistilBERT encoder + Qwen |
| **Full paper / offline lab** | **Download everything** | All 10 catalogue models |
| **Low RAM PC** | Beginner pack only; skip RoBERTa-large NER | Large NER is the RAM hog |

### Honest shortcomings (read once)
- **More models ≠ better Chat** — Chat quality depends on your documents + retrieval  
- **RoBERTa-large NER** is powerful but painful on small laptops  
- **Qwen 0.5B** is a teaching LLM, not AGI and not a cloud-grade assistant  
- **Streamlit Cloud** often cannot host multi-GB `local_models/` — use Extractive there  
- Downloads need **internet** the first time; files stay in `local_models/` afterward  
"""
    )


def run_download_ids(ids: List[str], label: str) -> None:
    if ensure_model is None:
        st.error("model_manager unavailable. Check project install / requirements.")
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
        st.success(f"{label} complete — {ok}/{len(ids)} ready.")
    else:
        st.warning(f"{label} finished with issues — {ok}/{len(ids)} ready.")
        for e in errors:
            st.error(e)
    st.rerun()


def render_one_click_zone() -> None:
    st.subheader("One-click solutions")
    st.caption("Internet required on first run. Safe to re-run — already-ready models are skipped.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Chat pack")
        st.write("Only **Qwen** local LLM")
        st.caption("Best if you only need generative Chat")
        if st.button("⬇ One-click: Chat pack", type="primary", use_container_width=True):
            with st.spinner("Downloading Chat pack..."):
                run_download_ids(CHAT_IDS, "Chat pack")
    with c2:
        st.markdown("#### Beginner pack")
        st.write("DistilBERT NER + DistilBERT + Qwen")
        st.caption("**Recommended start** for most users")
        if st.button("⬇ One-click: Beginner pack", type="primary", use_container_width=True):
            with st.spinner("Downloading Beginner pack..."):
                run_download_ids(BEGINNER_IDS, "Beginner pack")
    with c3:
        st.markdown("#### Everything")
        st.write("All models in `models.json`")
        st.caption("Largest · slowest · full offline lab")
        if st.button("⬇ One-click: Download everything", use_container_width=True):
            if ensure_all_models is None:
                st.error("ensure_all_models unavailable")
            else:
                status = st.empty()
                bar = st.progress(0.0)

                def cb(msg: str):
                    status.write(msg)

                with st.spinner("Downloading full catalogue..."):
                    results = ensure_all_models(progress_callback=cb)
                bar.progress(1.0)
                good = sum(1 for r in results if r["ok"])
                st.success(f"Full catalogue: {good}/{len(results)} ready")
                st.rerun()

    st.markdown("#### Also on your PC (no Streamlit needed)")
    st.code("setup.bat                  # packages + Qwen\nsetup_all_models.bat       # packages + all models", language="text")


def render_library_table() -> None:
    st.subheader("Model library — guide + download")
    if _IMPORT_ERROR:
        st.error(_IMPORT_ERROR)
        return

    rows = []
    for item in all_models():
        mid = item["id"]
        g = guide_for(mid)
        rows.append(
            {
                "Name": item.get("name", mid),
                "Group": next(
                    (grp for grp, items in CATALOGUE.items() if any(i.get("id") == mid for i in items)),
                    "",
                ),
                "Status": "READY" if is_ready(mid) else "NOT INSTALLED",
                "Level": g["level"],
                "Best for": g["best_for"],
                "Shortcoming": g["shortcoming"],
                "When to start": g["start"],
                "Model ID": mid,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### Download one model")
    names = {f"{r['Name']} ({r['Status']})": r["Model ID"] for r in rows}
    pick = st.selectbox("Choose model", list(names.keys()))
    mid = names[pick]
    g = guide_for(mid)
    st.info(
        f"**{g['level']}**  \n"
        f"Best for: {g['best_for']}  \n"
        f"Shortcoming: {g['shortcoming']}  \n"
        f"Start guidance: {g['start']}  \n"
        f"Folder: `{local_path(mid)}`"
    )
    if is_ready(mid):
        st.success("Already installed.")
    elif st.button("Download selected model", type="primary", use_container_width=True):
        with st.spinner(f"Downloading {mid}..."):
            try:
                ensure_model(mid)
                st.success("Ready.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def render_status_strip() -> None:
    report = status_report()
    if not report:
        st.warning("No catalogue status available.")
        return
    ready = sum(1 for r in report if r["ready"])
    st.metric("Models ready in local_models/", f"{ready} / {len(report)}")
    if st.button("Refresh status", use_container_width=True):
        st.rerun()


def render_install() -> None:
    st.title("3 · Install yourself — easy model downloads")
    st.caption(
        "Guided library install. Prefer **one-click Beginner pack**. "
        "No API key. Files go to project folder `local_models/`."
    )

    if _IMPORT_ERROR:
        st.error(f"Install engine problem: {_IMPORT_ERROR}")
        st.code("pip install -r requirements.txt\npython scripts/easy_setup.py")
        return

    render_status_strip()
    st.markdown("---")
    render_recommendation_board()
    st.markdown("---")
    render_one_click_zone()
    st.markdown("---")
    render_library_table()

    st.markdown("---")
    with st.expander("Troubleshooting"):
        st.markdown(
            """
- **Slow download** — wait; re-run the same button (skips finished models)  
- **Disk space** — Beginner pack is much smaller than “everything”  
- **Out of memory when running** — use Chat Extractive; free model memory; avoid RoBERTa-large NER  
- **Cloud deploy** — large weights usually stay on your PC; Cloud Chat should stay Extractive  
- **Offline classroom** — run **Download everything** once on a machine with internet, then copy `local_models/`  
- Desktop shortcuts: `setup.bat`, `setup_all_models.bat`, `run_app.bat`  
"""
        )
