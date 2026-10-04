"""Tutorial lab modules using preloaded local_models catalogue.

Paper-oriented workflow:
  1) Shared document text (paste by default, or upload PDF/Word/etc.)
  2) Model library status
  3) Encoder demos (BERT / ALBERT family)
  4) NER on the same text
  5) Compare NER models
  6) Local LLM Q&A on the same context
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from runtime_memory import (
    clear_cached_resources,
    configure_torch_runtime,
    free_python_and_torch,
    memory_caption,
    model_load_kwargs,
)

# Keep tutorial demos lighter on RAM
MAX_NER_CHARS = 2500
MAX_LLM_CONTEXT_CHARS = 1800
MAX_NEW_TOKENS = 120

try:
    from model_manager import (
        CATALOGUE,
        ensure_all_models,
        ensure_model,
        is_ready,
        local_path,
        status_report,
    )
except Exception as exc:  # pragma: no cover
    CATALOGUE = {}
    ensure_all_models = None
    ensure_model = None
    is_ready = lambda model_id: False  # noqa: E731
    local_path = lambda model_id: Path("local_models") / model_id.replace("/", "__")  # noqa: E731
    status_report = lambda: []  # noqa: E731
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = None

# Default one-page forensic sample (generic — no personal author names).
# Shared text context across ALL tutorial models.
DEFAULT_FORENSIC_DOCUMENT = """\
DIGITAL FORENSICS EXAMINATION SUMMARY — CASE DF-2026-0142

Agency: State Cyber Crime Investigation Unit
Exhibit ID: HDD-SEAGATE-1TB-S/N-9WM3A7
Examination date: 18 March 2026
Location of seizure: Corporate office, Bengaluru, Karnataka, India
Legal authority: Search warrant issued by the City Sessions Court on 15 March 2026

1. Scope of examination
The laboratory received one sealed hard disk drive for imaging and analysis. The request
was to determine whether the drive contained unauthorised access tools, deleted logs, or
evidence of data exfiltration related to a suspected insider-threat incident at a financial
services company.

2. Acquisition method
A bit-for-bit forensic image was created using a write blocker. The image hash was recorded
as SHA-256: 7f3a91c2e8b04d56a1f90e2c4b7d83a5e6f10294c8a7b3d1e0f5a69284c7d3e1.
The original exhibit remained sealed after imaging. Tool versions used for acquisition were
documented in the laboratory case notes.

3. Examination steps
Directory enumeration, deleted-file recovery, registry and event-log review, and keyword
searching were performed. Timeline analysis covered user activity between 01 February 2026
and 14 March 2026. Browser artefacts and cloud-sync folders were inspected for large outbound
file transfers.

4. Key findings
- A portable remote-administration utility was located in a user Temp folder created on
  03 March 2026 at 21:14 IST.
- Three compressed archives containing customer account spreadsheets were deleted on
  12 March 2026; recoverable fragments matched filenames used on the finance file server.
- Windows Security event logs showed repeated failed logons from workstation WS-FIN-07
  followed by a successful logon outside normal business hours on 11 March 2026.
- No evidence of full-disk encryption bypass was observed on this exhibit.

5. Limitations
Volatile memory was not collected at the scene. Network packet captures for the relevant
period were unavailable. Therefore conclusions about live remote sessions are limited to
disk-resident artefacts only.

6. Supporting evidence retained
Forensic image file, hash verification record, tool version list, screenshots of recovered
artefacts, exported event-log subsets, and this summary report. Another examiner should be
able to repeat the hash verification and re-open the image with the listed tools.

7. Preliminary conclusion
Disk-resident artefacts are consistent with unauthorised tool use and attempted removal of
customer data files. Findings should be correlated with server logs and HR access records
before final attribution.
"""

DEFAULT_QUESTION = (
    "What acquisition method was used, and what were the key findings about deleted files?"
)

DEFAULT_MASK_SENTENCE = (
    "A bit-for-bit forensic image was created using a write [MASK]."
)

DEFAULT_NER_SNIPPET = (
    "The laboratory received exhibit HDD-SEAGATE-1TB at the State Cyber Crime Investigation "
    "Unit in Bengaluru, Karnataka, India on 18 March 2026 under a warrant from the City "
    "Sessions Court. Analysis covered workstation WS-FIN-07 and activity between "
    "01 February 2026 and 14 March 2026."
)

SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md"]

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

WORKFLOW_STEPS = [
    "1 · Shared document text",
    "2 · Model library",
    "3 · Encoder demos (BERT / ALBERT)",
    "4 · NER explorer",
    "5 · Compare NER",
    "6 · Local LLM Q&A",
]


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def init_lab_state() -> None:
    if "lab_document_text" not in st.session_state:
        st.session_state.lab_document_text = DEFAULT_FORENSIC_DOCUMENT
    if "lab_document_source" not in st.session_state:
        st.session_state.lab_document_source = "Default forensic sample (built-in text)"
    if "lab_question" not in st.session_state:
        st.session_state.lab_question = DEFAULT_QUESTION
    if "lab_mask_sentence" not in st.session_state:
        st.session_state.lab_mask_sentence = DEFAULT_MASK_SENTENCE
    if "lab_ner_text" not in st.session_state:
        st.session_state.lab_ner_text = DEFAULT_NER_SNIPPET


def get_shared_text() -> str:
    init_lab_state()
    return st.session_state.lab_document_text or ""


def extract_text_from_upload(uploaded_file) -> str:
    """Load plain text from PDF / Word / Excel / CSV / TXT for shared lab context."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(tmp_path)
            parts = [(page.extract_text() or "") for page in reader.pages]
            return clean_text("\n\n".join(parts))
        if suffix in {".docx", ".doc"}:
            try:
                import docx2txt
            except Exception as exc:
                raise RuntimeError("docx2txt is required for Word files.") from exc
            return clean_text(docx2txt.process(tmp_path) or "")
        if suffix in {".xlsx", ".xls"}:
            import pandas as pd

            book = pd.read_excel(tmp_path, sheet_name=None, dtype=str)
            blocks = []
            for sheet_name, frame in book.items():
                frame = frame.fillna("")
                if frame.empty:
                    continue
                header = " | ".join(map(str, frame.columns))
                rows = [" | ".join(map(str, row)) for row in frame.values.tolist()]
                blocks.append(
                    f"Sheet: {sheet_name}\nColumns: {header}\n" + "\n".join(rows)
                )
            return clean_text("\n\n".join(blocks))
        if suffix == ".csv":
            import pandas as pd

            frame = pd.read_csv(tmp_path, dtype=str).fillna("")
            header = " | ".join(map(str, frame.columns))
            rows = [" | ".join(map(str, row)) for row in frame.values.tolist()]
            return clean_text(f"Columns: {header}\n" + "\n".join(rows))
        if suffix in {".txt", ".md"}:
            return clean_text(Path(tmp_path).read_text(encoding="utf-8", errors="ignore"))
        raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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


@st.cache_resource(max_entries=1, show_spinner=False)
def load_pipe(task: str, model_id: str, aggregation: bool = True):
    """Cache at most ONE pipeline so switching models drops the previous weights."""
    from transformers import pipeline

    configure_torch_runtime()
    local = local_path(model_id)
    model_source = str(local) if is_ready(model_id) else model_id
    kw: Dict[str, Any] = dict(
        task=task,
        model=model_source,
        tokenizer=model_source,
        device=-1,
        model_kwargs=model_load_kwargs(),
    )
    if task in ("ner", "token-classification") and aggregation:
        kw["aggregation_strategy"] = "simple"
    return pipeline(**kw)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_local_llm(model_dir: str):
    """Cache at most ONE local LLM copy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_torch_runtime()
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        **model_load_kwargs(),
    )
    model.eval()
    return tok, model


def free_lab_model_memory() -> str:
    return clear_cached_resources(load_pipe, load_local_llm)


def local_generate(question: str, context: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch

    # Drop encoder/NER pipeline before loading Qwen (one heavy model at a time)
    try:
        load_pipe.clear()
    except Exception:
        pass
    free_python_and_torch()

    model_dir = local_path(LOCAL_LLM_ID)
    if not is_ready(LOCAL_LLM_ID):
        raise FileNotFoundError(
            f"Local Qwen missing. Expected: {model_dir}. Run setup.bat or easy_setup.py"
        )
    tok, model = load_local_llm(str(model_dir))
    ctx = context.strip()
    if len(ctx) > MAX_LLM_CONTEXT_CHARS:
        ctx = ctx[:MAX_LLM_CONTEXT_CHARS] + "\n…[truncated to save memory]"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a forensic teaching assistant. Answer only from the supplied "
                "document context. Reply in clear English. If the answer is not in the "
                "context, say that it is not stated in the document."
            ),
        },
        {"role": "user", "content": f"Document context:\n{ctx}\n\nQuestion:\n{question}"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new = out[0][inputs.input_ids.shape[-1] :]
    answer = tok.decode(new, skip_special_tokens=True).strip()
    # Free generation tensors promptly
    del inputs, out, new
    free_python_and_torch()
    return answer


def progress_run(task: str, model_id: str, text: str, unload_after: bool = False, **kwargs):
    p = st.progress(0, "Checking local model library…")
    prepare_selected_model(model_id, p)
    p.progress(65, "Loading model into memory (replaces previous model)…")
    clean = {k: v for k, v in kwargs.items() if v is not None}
    # Unload LLM before encoder/NER so RAM is not stacked
    try:
        load_local_llm.clear()
    except Exception:
        pass
    free_python_and_torch()
    pipe = load_pipe(task, model_id)
    p.progress(80, "Running inference…")
    # Bound NER input length
    run_text = text
    if task == "ner" and len(run_text) > MAX_NER_CHARS:
        run_text = run_text[:MAX_NER_CHARS]
    result = pipe(run_text, **clean)
    p.progress(100, "Complete")
    time.sleep(0.05)
    p.empty()
    if unload_after:
        try:
            load_pipe.clear()
        except Exception:
            pass
        free_python_and_torch()
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


def render_workflow_banner(current: str) -> None:
    st.markdown("#### Paper workflow order")
    bits = []
    for step in WORKFLOW_STEPS:
        if step == current:
            bits.append(f"**→ {step}**")
        else:
            bits.append(step)
    st.caption("  ·  ".join(bits))


def render_shared_document_panel(show_editor: bool = True) -> str:
    """Default = editable text. Optional upload fills the same shared text for all models."""
    init_lab_state()
    st.subheader("Shared document text (used by all tutorial models)")
    st.caption(
        "Default input is **plain text**. Paste any forensic (or other) content, "
        "or upload a PDF / Word / Excel / CSV / TXT file to load text into this box. "
        "Every module below reads the same shared context."
    )

    if show_editor:
        c1, c2 = st.columns([2, 1])
        with c1:
            input_mode = st.radio(
                "How do you want to provide content?",
                [
                    "Paste / edit text (default)",
                    "Upload file → fill text box",
                ],
                horizontal=True,
                key="lab_input_mode",
            )
        with c2:
            if st.button("Reset to sample forensic page", use_container_width=True):
                st.session_state.lab_document_text = DEFAULT_FORENSIC_DOCUMENT
                st.session_state.lab_document_text_area = DEFAULT_FORENSIC_DOCUMENT
                st.session_state.lab_document_source = (
                    "Default forensic sample (built-in text)"
                )
                st.session_state.lab_ner_text = DEFAULT_NER_SNIPPET
                st.session_state.lab_mask_sentence = DEFAULT_MASK_SENTENCE
                st.session_state.lab_question = DEFAULT_QUESTION
                st.rerun()

        if input_mode.startswith("Upload"):
            uploaded = st.file_uploader(
                "Upload PDF, Word, Excel, CSV, or text",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=False,
                key="lab_shared_upload",
            )
            if uploaded is not None and st.button(
                "Load file into shared text", type="primary", use_container_width=True
            ):
                try:
                    text = extract_text_from_upload(uploaded)
                    if not text:
                        st.error("No text could be extracted from that file.")
                    else:
                        st.session_state.lab_document_text = text
                        st.session_state.lab_document_text_area = text
                        st.session_state.lab_document_source = f"Uploaded: {uploaded.name}"
                        st.session_state.lab_ner_text = text[:900]
                        st.success(
                            f"Loaded {len(text)} characters from {uploaded.name}. "
                            "You can still edit the text below."
                        )
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")

        if "lab_document_text_area" not in st.session_state:
            st.session_state.lab_document_text_area = st.session_state.lab_document_text

        st.text_area(
            "Document text (default context for all models)",
            height=320,
            key="lab_document_text_area",
            help="Paste content here. This text is the default context for NER, encoders, and Local LLM.",
        )
        st.session_state.lab_document_text = st.session_state.lab_document_text_area

    text = get_shared_text()
    src = st.session_state.get("lab_document_source", "Shared text")
    chars = len(text)
    words = len(text.split()) if text else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Characters", f"{chars:,}")
    m2.metric("Approx. words", f"{words:,}")
    m3.write(f"**Source:** {src}")
    if not text.strip():
        st.warning("Shared document text is empty. Paste or upload content to continue.")
    return text


def render_context_preview(max_chars: int = 500) -> None:
    text = get_shared_text()
    with st.expander("Preview shared document (read-only)", expanded=False):
        st.caption(st.session_state.get("lab_document_source", ""))
        preview = text if len(text) <= max_chars else text[:max_chars] + "…"
        st.text(preview)
        st.caption(
            "Edit the full text in workflow step **1 · Shared document text**, "
            "or paste a custom slice in the boxes below."
        )


def render_model_library() -> None:
    st.subheader("Step 2 — Model library (status & preload)")
    st.caption("Confirm local_models/ before running paper experiments.")
    with st.expander("Install notes", expanded=False):
        st.markdown(
            """
- `setup.bat` → packages + Local LLM only
- `setup_all_models.bat` → all 10 tutorial models
- Guides: `INSTALL.md`, `USER_GUIDE.md`
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


def render_encoder_explorer() -> None:
    st.subheader("Step 3 — Encoder demos (BERT / DistilBERT / RoBERTa / ALBERT)")
    st.caption(
        "Masked language modeling on a sentence drawn from the shared forensic text "
        "(or your own sentence with [MASK])."
    )
    render_context_preview()
    label = st.selectbox("Encoder model", list(ENCODER_MODELS.keys()))
    mid = ENCODER_MODELS[label]
    st.write(f"Model ID: `{mid}` · Local: **{'READY' if is_ready(mid) else 'MISSING'}**")

    if st.button("Use default mask sentence from sample", use_container_width=True):
        st.session_state.lab_mask_sentence = DEFAULT_MASK_SENTENCE
        st.rerun()

    txt = st.text_input(
        "Sentence with [MASK]",
        value=st.session_state.get("lab_mask_sentence", DEFAULT_MASK_SENTENCE),
        key="lab_mask_input",
    )
    st.session_state.lab_mask_sentence = txt

    if st.button("Run fill-mask", type="primary", use_container_width=True):
        try:
            result = progress_run("fill-mask", mid, txt, top_k=5)
            st.json(result)
        except Exception:
            try:
                alt = txt.replace("[MASK]", "<mask>")
                result = progress_run("fill-mask", mid, alt, top_k=5)
                st.caption("Retried with <mask> token (RoBERTa-style).")
                st.json(result)
            except Exception as exc2:
                st.error("Fill-mask run failed.")
                st.code(str(exc2))


def render_ner_explorer() -> None:
    st.subheader("Step 4 — NER explorer")
    st.caption(
        "Named-entity recognition on text from the shared document (editable). "
        "Default text is forensic sample content — paste or load any text you prefer."
    )
    render_context_preview()
    label = st.selectbox("NER model", list(NER_MODELS.keys()))
    mid = NER_MODELS[label]
    st.write(f"Model ID: `{mid}` · Local: **{'READY' if is_ready(mid) else 'MISSING'}**")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Use full shared document", use_container_width=True):
            st.session_state.lab_ner_text = get_shared_text()[:4000]
            st.rerun()
    with b2:
        if st.button("Use short forensic NER sample", use_container_width=True):
            st.session_state.lab_ner_text = DEFAULT_NER_SNIPPET
            st.rerun()

    txt = st.text_area(
        "Text for NER (default = document-related text)",
        value=st.session_state.get("lab_ner_text", DEFAULT_NER_SNIPPET),
        height=200,
        key="lab_ner_input",
    )
    st.session_state.lab_ner_text = txt

    if st.button("Run NER", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("Enter or paste text first.")
            return
        try:
            result = progress_run("ner", mid, txt)
            st.dataframe(entity_rows(result), use_container_width=True, hide_index=True)
            chips = " ".join(
                [
                    (
                        "<span style='display:inline-block;padding:6px 9px;margin:3px;"
                        "border-radius:8px;background:rgba(0,120,255,.12)'>"
                        f"<b>{e.get('word', '')}</b> "
                        f"[{e.get('entity_group', e.get('entity'))}] "
                        f"{float(e.get('score', 0)):.2f}</span>"
                    )
                    for e in result
                ]
            )
            st.markdown(chips, unsafe_allow_html=True)
        except Exception as exc:
            st.error("NER run failed.")
            st.code(str(exc))


def render_compare_ner() -> None:
    st.subheader("Step 5 — Compare NER models (same text)")
    st.caption("Run multiple NER checkpoints on one shared passage for the paper comparison.")
    render_context_preview()

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Load full shared document into compare box", use_container_width=True):
            st.session_state.lab_ner_text = get_shared_text()[:4000]
            st.rerun()
    with b2:
        if st.button("Load short forensic sample", use_container_width=True):
            st.session_state.lab_ner_text = DEFAULT_NER_SNIPPET
            st.rerun()

    txt = st.text_area(
        "Same English input for all selected models",
        value=st.session_state.get("lab_ner_text", DEFAULT_NER_SNIPPET),
        height=180,
        key="lab_compare_ner_input",
    )
    st.session_state.lab_ner_text = txt

    st.info(
        "Memory-safe compare: models run **one after another**. "
        "Only one stays loaded. Prefer 2 models if RAM is tight. "
        "RoBERTa-large NER uses the most memory."
    )
    selected = st.multiselect(
        "Models",
        list(NER_MODELS.keys()),
        default=["BERT NER", "DistilBERT NER"],
    )
    if st.button("Compare NER models", type="primary", use_container_width=True):
        if not selected:
            st.warning("Select at least one model.")
            return
        if not txt.strip():
            st.warning("Enter text first.")
            return
        cols = st.columns(max(1, len(selected)))
        bar = st.progress(0, "Starting comparison (one model at a time)…")
        for i, name in enumerate(selected):
            with cols[i]:
                st.markdown(f"### {name}")
                try:
                    # unload_after=True frees weights before the next model
                    result = progress_run(
                        "ner",
                        NER_MODELS[name],
                        txt,
                        unload_after=True,
                    )
                    st.dataframe(
                        entity_rows(result), use_container_width=True, hide_index=True
                    )
                except Exception as exc:
                    st.error(str(exc)[:400])
            bar.progress(int((i + 1) / len(selected) * 100))
        bar.empty()
        st.caption(free_lab_model_memory())


def render_local_llm_demo() -> None:
    st.subheader("Step 6 — Local LLM Q&A on shared document")
    st.caption(
        "Qwen answers using the **shared document text** as context (RAG-style teaching demo). "
        "Paste/upload in step 1; ask questions here."
    )
    path = local_path(LOCAL_LLM_ID)
    if is_ready(LOCAL_LLM_ID):
        st.success(f"Local Qwen ready: {path}")
    else:
        st.warning(f"Local Qwen missing: {path}")
        st.caption("Easiest fix: double-click setup.bat, or download below.")
        if ensure_model is not None and st.button(
            "Download Qwen now (one model)", type="primary", use_container_width=True
        ):
            with st.spinner("Downloading Local LLM only..."):
                try:
                    ensure_model(LOCAL_LLM_ID)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    render_context_preview(max_chars=700)

    st.markdown("**Context for this question** (defaults to full shared document text)")
    use_full = st.checkbox("Use full shared document as context", value=True)
    if use_full:
        context = get_shared_text()
        st.text_area(
            "Context (from shared document)",
            value=context,
            height=220,
            disabled=True,
            key="lab_llm_context_view",
        )
    else:
        context = st.text_area(
            "Custom context (paste any text)",
            value=get_shared_text(),
            height=220,
            key="lab_llm_custom_context",
        )

    question = st.text_input(
        "Question",
        value=st.session_state.get("lab_question", DEFAULT_QUESTION),
        key="lab_llm_question",
    )
    st.session_state.lab_question = question

    if st.button("Ask local LLM", type="primary", use_container_width=True):
        if not (context or "").strip():
            st.warning("Context is empty. Paste text or upload a file in step 1.")
            return
        if not question.strip():
            st.warning("Enter a question.")
            return
        try:
            with st.spinner("Generating with local Qwen from document context..."):
                answer = local_generate(question, context)
            st.success(answer)
        except Exception as exc:
            st.error("Local LLM failed.")
            st.code(str(exc))


def render_tutorial_lab() -> None:
    from ui_flow import callout, lesson_stepper, mark_step_done, render_nav_buttons

    init_lab_state()
    configure_torch_runtime()
    st.markdown("## Step 4 · Practice — hands-on tutorial")
    st.caption(
        "One shared document flows through every demo: text → models → encoder → NER → compare → local LLM. "
        "No left menu. No API key."
    )
    callout(
        "<b>Do these mini-steps in order.</b> Keep the same forensic text so paper comparisons stay fair."
    )
    st.caption(memory_caption())

    cmem1, cmem2 = st.columns([3, 1])
    with cmem2:
        if st.button("Free model memory", use_container_width=True, key="lab_free_mem"):
            st.success(free_lab_model_memory())
    with cmem1:
        st.caption("Loads **one model at a time**. Free memory after heavy demos.")

    if _IMPORT_ERROR:
        st.warning(f"Model manager issue: {_IMPORT_ERROR}")

    with st.expander("Recommended order for the paper", expanded=False):
        st.markdown(
            """
1. **Shared document text** — paste content (default) or upload PDF/Word/Excel/TXT
2. **Model library** — confirm models READY
3. **Encoder demos** — BERT / ALBERT fill-mask on a sentence from the document
4. **NER explorer** — entities from the same document text
5. **Compare NER** — side-by-side models on one passage
6. **Local LLM Q&A** — ask questions grounded in the shared document

Default sample is a **one-page generic digital forensics summary** (no personal names).
"""
        )

    idx = lesson_stepper(WORKFLOW_STEPS, "practice_idx", total_label="Practice step")
    tab = WORKFLOW_STEPS[idx]
    render_workflow_banner(tab)

    if tab == WORKFLOW_STEPS[0]:
        render_shared_document_panel(show_editor=True)
        st.info(
            "Next: open **2 · Model library**, then run encoder / NER / LLM steps. "
            "They all reuse this shared text."
        )
    elif tab == WORKFLOW_STEPS[1]:
        render_model_library()
    elif tab == WORKFLOW_STEPS[2]:
        render_encoder_explorer()
    elif tab == WORKFLOW_STEPS[3]:
        render_ner_explorer()
    elif tab == WORKFLOW_STEPS[4]:
        render_compare_ner()
    else:
        render_local_llm_demo()

    if idx >= len(WORKFLOW_STEPS) - 1:
        callout(
            "Practice path complete. Continue to <b>Chat</b> to try your own documents.",
            kind="ok",
        )
        mark_step_done("practice")

    st.markdown("---")
    render_nav_buttons(
        prev_key="learn",
        next_key="chat",
        next_label="Continue to Chat →",
        mark_done_key="practice",
    )
