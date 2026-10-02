"""Tutorial lab modules using preloaded local_models catalogue."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    local_path = lambda model_id: Path("local_models") / model_id.replace("/", "__")  # noqa: E731
    status_report = lambda: []  # noqa: E731
    all_models = lambda: []  # noqa: E731
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = None

DEMO_NER = (
    "Dr. Ranjit Kolkar visited National Forensic Sciences University in Goa "
    "on 12 September 2026 for a Digital Forensics workshop. "
    "The team later travelled to New Delhi to meet officials from the Ministry of Home Affairs."
)
DEMO_Q = "What should a forensic report contain?"
DEMO_CONTEXT = """A forensic report should describe the scope of the examination, tools and versions used,
the acquisition method, examination steps, findings, limitations, and supporting evidence.
Screenshots, hash values, logs, and other reproducibility information can help another examiner
understand how the findings were obtained."""

NER_MODELS = {
    "BERT NER": "dslim/bert-base-NER",
    "DistilBERT NER": "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "RoBERTa NER": "Jean-Baptiste/roberta-large-ner-english",
    "XLM-R NER": "Davlan/xlm-roberta-base-ner-hrl",
}
ENCODER_MODELS = {
    "BERT base uncased": "google-bert/bert-base-uncased",
    "BERT base cased": "google-bert/bert-base-cased",
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
    "ALBERT": "albert/albert-base-v2",
}
LOCAL_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def prepare_selected_model(model_id: str, progress=None):
    if is_ready(model_id):
        if progress:
            progress.progress(45, "Using model already stored in project…")
        return local_path(model_id)
    if ensure_model is None:
        raise FileNotFoundError(
            f"Model not found locally and auto-download is unavailable: {model_id}"
        )
    if progress:
        progress.progress(20, "Preparing model from Hugging Face Hub…")
    path = ensure_model(model_id)
    if progress:
        progress.progress(55, "Model is ready locally…")
    return path


@st.cache_resource(show_spinner=False)
def load_pipe(task: str, model_id: str, aggregation: bool = True):
    from transformers import pipeline

    local = local_path(model_id)
    model_source = str(local) if is_ready(model_id) else model_id
    kw: Dict[str, Any] = dict(
        task=task, model=model_source, tokenizer=model_source, device=-1
    )
    if task in ("ner", "token-classification") and aggregation:
        kw["aggregation_strategy"] = "simple"
    return pipeline(**kw)


@st.cache_resource(show_spinner=False)
def load_local_llm(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    return tok, model


def local_generate(question: str, context: str, max_new_tokens: int = 220) -> str:
    import torch

    model_dir = local_path(LOCAL_LLM_ID)
    if not is_ready(LOCAL_LLM_ID):
        raise FileNotFoundError(
            f"Local Qwen missing. Expected: {model_dir}. Run scripts/preload_all_models.py"
        )
    tok, model = load_local_llm(str(model_dir))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a teaching assistant. Answer only from the supplied context. "
                "If the answer is absent, say so."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs.input_ids.shape[-1] :]
    return tok.decode(new, skip_special_tokens=True)


def progress_run(task: str, model_id: str, text: str, **kwargs):
    p = st.progress(0, "Checking local model library…")
    prepare_selected_model(model_id, p)
    p.progress(65, "Loading model into memory…")
    # drop None kwargs for pipeline
    clean = {k: v for k, v in kwargs.items() if v is not None}
    pipe = load_pipe(task, model_id)
    p.progress(80, "Running inference…")
    result = pipe(text, **clean)
    p.progress(100, "Complete")
    time.sleep(0.05)
    p.empty()
    return result


def entity_rows(entities: List[dict]) -> List[dict]:
    return [
        {
            "Entity": e.get("word", ""),
            "Label": e.get("entity_group", e.get("entity", "")),
            "Confidence": round(float(e.get("score", 0)), 4),
            "Start": e.get("start", ""),
            "End": e.get("end", ""),
        }
        for e in entities
    ]


def render_model_library() -> None:
    st.subheader("Tutorial model library")
    st.caption("Models live under local_models/. Prefer one-download setup for Local LLM.")
    with st.expander("Easy install + all models guide", expanded=True):
        st.markdown(
            """
**Windows (least work):**

1. Double-click `setup.bat` → packages + **one** Local LLM download  
2. Double-click `run_app.bat` → open the app  

**Optional full pack:** `setup_all_models.bat` (all 10 models)

**Catalogue (10 models)**  
- **NER (4):** BERT NER, DistilBERT NER, RoBERTa NER, XLM-R NER  
- **Encoders (5):** BERT uncased/cased, DistilBERT, RoBERTa, ALBERT  
- **Local LLM (1):** Qwen2.5-0.5B-Instruct  

**Docs in project folder:** `INSTALL.md` (install) · `USER_GUIDE.md` (how to use every model + context chatbot / RAG)
"""
        )
    if _IMPORT_ERROR:
        st.error(f"model_manager import failed: {_IMPORT_ERROR}")
        return

    report = status_report()
    ready_count = sum(1 for row in report if row["ready"])
    st.metric("Models ready locally", f"{ready_count}/{len(report)}")
    st.dataframe(
        [
            {
                "Group": next(
                    (
                        g
                        for g, items in CATALOGUE.items()
                        if any(i.get("id") == row["id"] for i in items)
                    ),
                    "",
                ),
                "Name": row["name"],
                "Model ID": row["id"],
                "Status": "READY" if row["ready"] else "MISSING",
                "Local path": row["path"],
            }
            for row in report
        ],
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Download Local LLM only", type="primary", use_container_width=True):
            if ensure_model is None:
                st.error("ensure_model is unavailable.")
            else:
                with st.spinner("Downloading Qwen (one model)..."):
                    try:
                        ensure_model(LOCAL_LLM_ID)
                        st.success("Local LLM ready.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
    with c2:
        if st.button("Preload ALL tutorial models", use_container_width=True):
            if ensure_all_models is None:
                st.error("ensure_all_models is unavailable.")
            else:
                status = st.empty()
                bar = st.progress(0.0)

                def cb(message: str):
                    status.write(message)

                results = ensure_all_models(progress_callback=cb)
                bar.progress(1.0)
                ok = sum(1 for r in results if r["ok"])
                st.success(f"Finished: {ok}/{len(results)} models ready.")
                st.rerun()
    with c3:
        if st.button("Refresh status", use_container_width=True):
            st.rerun()

    st.markdown("### Catalogue groups")
    for group, items in CATALOGUE.items():
        st.markdown(f"**{group}**")
        for item in items:
            mark = "READY" if is_ready(item["id"]) else "MISSING"
            st.write(f"- [{mark}] {item['name']} (`{item['id']}`)")


def render_ner_explorer() -> None:
    st.subheader("NER Explorer (tutorial models)")
    st.caption("Uses preloaded NER checkpoints from local_models/ when available.")
    label = st.selectbox("NER model", list(NER_MODELS.keys()))
    mid = NER_MODELS[label]
    st.write(f"Model ID: `{mid}`")
    st.write(f"Local status: **{'READY' if is_ready(mid) else 'MISSING'}**")
    txt = st.text_area("English input", DEMO_NER, height=160)
    if st.button("Run NER", type="primary", use_container_width=True):
        try:
            result = progress_run("ner", mid, txt)
            st.dataframe(entity_rows(result), use_container_width=True, hide_index=True)
            chips = " ".join(
                [
                    f"<span style='display:inline-block;padding:6px 9px;margin:3px;"
                    f"border-radius:8px;background:rgba(0,120,255,.12)'>"
                    f"<b>{e.get('word','')}</b> "
                    f"[{e.get('entity_group', e.get('entity'))}] "
                    f"{float(e.get('score', 0)):.2f}</span>"
                    for e in result
                ]
            )
            st.markdown(chips, unsafe_allow_html=True)
        except Exception as exc:
            st.error("NER run failed.")
            st.code(str(exc))


def render_encoder_explorer() -> None:
    st.subheader("BERT / ALBERT Explorer (tutorial encoders)")
    st.caption("Masked language modeling with base encoder checkpoints.")
    label = st.selectbox("Encoder model", list(ENCODER_MODELS.keys()))
    mid = ENCODER_MODELS[label]
    st.write(f"Model ID: `{mid}`")
    st.write(f"Local status: **{'READY' if is_ready(mid) else 'MISSING'}**")
    txt = st.text_input(
        "Input with [MASK]",
        "National Forensic Sciences University is located in [MASK].",
    )
    if st.button("Run fill-mask", type="primary", use_container_width=True):
        try:
            # ALBERT/RoBERTa may use <mask> — try smart replace for non-BERT tokenizers
            run_text = txt
            result = progress_run("fill-mask", mid, run_text, top_k=5)
            st.json(result)
        except Exception as exc:
            # Retry with <mask> for RoBERTa-like models
            try:
                alt = txt.replace("[MASK]", "<mask>")
                result = progress_run("fill-mask", mid, alt, top_k=5)
                st.caption("Retried with <mask> token.")
                st.json(result)
            except Exception as exc2:
                st.error("Fill-mask run failed.")
                st.code(str(exc2))


def render_local_llm_demo() -> None:
    st.subheader("Local LLM tutorial (Qwen)")
    path = local_path(LOCAL_LLM_ID)
    if is_ready(LOCAL_LLM_ID):
        st.success(f"Local Qwen ready: {path}")
    else:
        st.warning(f"Local Qwen missing: {path}")
        st.caption(
            "Easiest fix: double-click setup.bat (one Local LLM download), "
            "or click the button below."
        )
        if ensure_model is not None and st.button(
            "Download Qwen now (one model)", type="primary", use_container_width=True
        ):
            with st.spinner("Downloading Local LLM only..."):
                try:
                    ensure_model(LOCAL_LLM_ID)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    context = st.text_area("Context", DEMO_CONTEXT, height=180)
    question = st.text_input("Question", DEMO_Q)
    if st.button("Ask local LLM", type="primary", use_container_width=True):
        try:
            with st.spinner("Generating with local Qwen..."):
                answer = local_generate(question, context)
            st.success(answer)
        except Exception as exc:
            st.error("Local LLM failed.")
            st.code(str(exc))


def render_compare_ner() -> None:
    st.subheader("Compare tutorial NER models")
    txt = st.text_area("Same English input", DEMO_NER, height=150)
    selected = st.multiselect(
        "Models",
        list(NER_MODELS.keys()),
        default=["BERT NER", "DistilBERT NER"],
    )
    if st.button("Compare NER models", type="primary", use_container_width=True):
        if not selected:
            st.warning("Select at least one model.")
            return
        cols = st.columns(max(1, len(selected)))
        bar = st.progress(0, "Starting comparison…")
        for i, label in enumerate(selected):
            with cols[i]:
                st.markdown(f"### {label}")
                try:
                    result = progress_run("ner", NER_MODELS[label], txt)
                    st.dataframe(
                        entity_rows(result), use_container_width=True, hide_index=True
                    )
                except Exception as exc:
                    st.error(str(exc)[:400])
            bar.progress(int((i + 1) / len(selected) * 100))
        bar.empty()


def render_tutorial_lab() -> None:
    st.title("Tutorial Lab — BERT / NER / ALBERT / Local LLM")
    st.caption(
        "Classroom demos using models from models.json stored in local_models/. No API key required."
    )
    if _IMPORT_ERROR:
        st.warning(f"Model manager issue: {_IMPORT_ERROR}")

    tab = st.sidebar.radio(
        "Tutorial module",
        [
            "Model Library",
            "NER Explorer",
            "BERT / ALBERT Explorer",
            "Local LLM",
            "Compare NER",
        ],
        key="tutorial_module",
    )

    if tab == "Model Library":
        render_model_library()
    elif tab == "NER Explorer":
        render_ner_explorer()
    elif tab == "BERT / ALBERT Explorer":
        render_encoder_explorer()
    elif tab == "Local LLM":
        render_local_llm_demo()
    else:
        render_compare_ner()
