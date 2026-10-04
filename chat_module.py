"""Chat mode — document Q&A using available project models (no API key)."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from runtime_memory import (
    clear_cached_resources,
    configure_torch_runtime,
    free_python_and_torch,
    memory_caption,
    model_load_kwargs,
)

try:
    import docx2txt
except Exception:  # pragma: no cover
    docx2txt = None

try:
    from model_manager import (
        ensure_model,
        is_ready as mm_is_ready,
        local_path as mm_local_path,
        status_report,
    )
except Exception as exc:  # pragma: no cover
    ensure_model = None
    mm_is_ready = None
    mm_local_path = None
    status_report = lambda: []  # noqa: E731
    _MM_ERROR = str(exc)
else:
    _MM_ERROR = None

APP_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = APP_ROOT / "local_models"
LOCAL_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_LLM_DIRNAMES = [
    "Qwen__Qwen2.5-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen-Qwen2.5-0.5B-Instruct",
]

MAX_LLM_CONTEXT_CHARS = 1800
MAX_NEW_TOKENS = 120
SUPPORTED_TYPES = ["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md"]
NO_ANSWER = "I could not find that information in the uploaded document(s)."


@dataclass
class Chunk:
    text: str
    source: str
    page: Optional[int] = None
    sheet: Optional[str] = None


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "chunks": [],
        "vectorizer": None,
        "matrix": None,
        "doc_names": [],
        "chunk_count": 0,
        "answer_mode": "extractive",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    page: Optional[int] = None,
    sheet: Optional[str] = None,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    chunks: List[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, page=page, sheet=sheet))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf(path: str, source: str) -> List[Chunk]:
    reader = PdfReader(path)
    out: List[Chunk] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.extend(chunk_text(text, source=source, page=i))
    return out


def load_docx(path: str, source: str) -> List[Chunk]:
    if docx2txt is None:
        raise RuntimeError("docx2txt is not installed. Add docx2txt to requirements.")
    text = docx2txt.process(path) or ""
    return chunk_text(text, source=source)


def load_excel(path: str, source: str) -> List[Chunk]:
    out: List[Chunk] = []
    book = pd.read_excel(path, sheet_name=None, dtype=str)
    for sheet_name, frame in book.items():
        frame = frame.fillna("")
        if frame.empty:
            continue
        header = " | ".join(map(str, frame.columns))
        rows = [" | ".join(map(str, row)) for row in frame.values.tolist()]
        text = (
            f"Spreadsheet: {source}\nSheet: {sheet_name}\nColumns: {header}\n"
            + "\n".join(rows)
        )
        out.extend(chunk_text(text, source=source, sheet=str(sheet_name)))
    return out


def load_csv(path: str, source: str) -> List[Chunk]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.empty:
        return []
    header = " | ".join(map(str, frame.columns))
    rows = [" | ".join(map(str, row)) for row in frame.values.tolist()]
    text = f"CSV file: {source}\nColumns: {header}\n" + "\n".join(rows)
    return chunk_text(text, source=source)


def load_text(path: str, source: str) -> List[Chunk]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    return chunk_text(raw, source=source)


def load_uploaded_file(uploaded_file) -> List[Chunk]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return load_pdf(tmp_path, uploaded_file.name)
        if suffix in {".docx", ".doc"}:
            return load_docx(tmp_path, uploaded_file.name)
        if suffix in {".xlsx", ".xls"}:
            return load_excel(tmp_path, uploaded_file.name)
        if suffix == ".csv":
            return load_csv(tmp_path, uploaded_file.name)
        if suffix in {".txt", ".md"}:
            return load_text(tmp_path, uploaded_file.name)
        raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, object]:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(question: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
    chunks: List[Chunk] = st.session_state.chunks
    vectorizer: TfidfVectorizer = st.session_state.vectorizer
    matrix = st.session_state.matrix
    if not chunks or vectorizer is None or matrix is None:
        return []
    q = vectorizer.transform([question])
    scores = cosine_similarity(q, matrix).ravel()
    order = scores.argsort()[::-1][:top_k]
    results = []
    for idx in order:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append((chunks[idx], score))
    return results


def label_chunk(chunk: Chunk) -> str:
    label = chunk.source
    if chunk.page is not None:
        label += f" (page {chunk.page + 1})"
    if chunk.sheet:
        label += f" (sheet: {chunk.sheet})"
    return label


def extractive_answer(question: str, hits: List[Tuple[Chunk, float]]) -> str:
    if not hits or hits[0][1] < 0.05:
        return NO_ANSWER
    lines = [
        "Based only on your uploaded document(s), here is the most relevant information:",
        "",
    ]
    for i, (chunk, score) in enumerate(hits[:3], start=1):
        snippet = " ".join(chunk.text.split())
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + "..."
        lines.append(f"**{i}. From {label_chunk(chunk)}** (match {score:.2f})")
        lines.append(snippet)
        lines.append("")
    lines.append(
        "_Extractive mode — closest passages from your files. No API key._"
    )
    return "\n".join(lines).strip()


def qwen_candidate_dirs() -> List[Path]:
    dirs = [MODEL_ROOT / name for name in LOCAL_LLM_DIRNAMES]
    if mm_local_path is not None:
        try:
            dirs.insert(0, Path(mm_local_path(LOCAL_LLM_ID)))
        except Exception:
            pass
    seen = set()
    out: List[Path] = []
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def folder_has_llm_weights(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return (
        any(path.glob("model*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin.index.json").exists()
    )


def resolve_qwen_dir() -> Optional[Path]:
    for path in qwen_candidate_dirs():
        if folder_has_llm_weights(path):
            return path
    if mm_is_ready is not None and mm_local_path is not None:
        try:
            if mm_is_ready(LOCAL_LLM_ID):
                return Path(mm_local_path(LOCAL_LLM_ID))
        except Exception:
            pass
    return None


@st.cache_resource(max_entries=1, show_spinner=False)
def load_local_llm(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_torch_runtime()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        **model_load_kwargs(),
    )
    model.eval()
    return tokenizer, model


def free_chatbot_model_memory() -> str:
    return clear_cached_resources(load_local_llm)


def local_llm_answer(
    question: str, hits: List[Tuple[Chunk, float]]
) -> Tuple[Optional[str], Optional[str]]:
    if not hits or hits[0][1] < 0.05:
        return NO_ANSWER, None

    model_dir = resolve_qwen_dir()
    if model_dir is None:
        return None, (
            "Local Qwen not found. Open **Install yourself** and download the Local LLM, "
            f"or run setup.bat. Expected under: {MODEL_ROOT}"
        )

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return None, f"Need torch + transformers. Use Install yourself. Details: {exc}"

    try:
        tokenizer, model = load_local_llm(str(model_dir))
    except Exception as exc:
        return None, f"Failed to load local Qwen from {model_dir}: {exc}"

    import torch

    context = "\n\n".join(
        f"Source: {label_chunk(chunk)}\n{chunk.text}" for chunk, _ in hits[:3]
    )
    if len(context) > MAX_LLM_CONTEXT_CHARS:
        context = context[:MAX_LLM_CONTEXT_CHARS] + "\n…[truncated to save memory]"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Answer ONLY from the supplied context. "
                "Reply in clear English. If the answer is not in the context, say exactly: "
                f"{NO_ANSWER}"
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([prompt], return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        del inputs, output, new_tokens
        free_python_and_torch()
        return (text or NO_ANSWER), None
    except Exception as exc:
        free_python_and_torch()
        return None, f"Local Qwen generation failed: {exc}"


def answer_question(question: str) -> Dict[str, object]:
    hits = retrieve(question, top_k=4)
    sources = []
    for chunk, score in hits:
        snippet = " ".join(chunk.text.split())[:180]
        sources.append(f"**{label_chunk(chunk)}** (score {score:.2f}): {snippet}...")

    mode = st.session_state.answer_mode
    answer = None
    used_mode = "extractive"
    error = None

    if mode == "local_llm":
        answer, error = local_llm_answer(question, hits)
        if answer is not None:
            used_mode = "local_llm"

    if answer is None:
        answer = extractive_answer(question, hits)
        used_mode = "extractive"
        if error and mode == "local_llm":
            answer = (
                f"Local Qwen unavailable ({error})\n\n"
                f"Fell back to extractive mode:\n\n{answer}"
            )

    return {"answer": answer, "sources": sources, "mode": used_mode, "error": error}


def reset_chat() -> None:
    st.session_state.messages = []


def clear_knowledge_base() -> None:
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.matrix = None
    st.session_state.doc_names = []
    st.session_state.chunk_count = 0


def render_available_models_panel() -> None:
    st.subheader("Available models in this project")
    st.caption(
        "Status from `local_models/` + `models.json`. "
        "Chat works with **Extractive** even if neural models are missing."
    )
    try:
        rows = status_report()
    except Exception:
        rows = []
    if not rows:
        st.warning("Could not read model catalogue. Open **Install yourself**.")
        return
    ready = sum(1 for r in rows if r["ready"])
    st.metric("Ready locally", f"{ready}/{len(rows)}")
    st.dataframe(
        [
            {
                "Name": r["name"],
                "ID": r["id"],
                "Status": "READY" if r["ready"] else "NOT INSTALLED",
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    qwen = resolve_qwen_dir()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Chat engines**")
        st.write("- Extractive search (always on, low memory)")
        if qwen:
            st.write(f"- Local Qwen READY: `{qwen.name}`")
        else:
            st.write("- Local Qwen: not installed → use **Install yourself**")
    with c2:
        st.markdown("**Need a model?**")
        st.write("Go to sidebar → **3 · Install yourself**")
        st.write("One-click: download beginner pack or everything")


def render_chat() -> None:
    """Main Chat mode UI."""
    init_session_state()
    configure_torch_runtime()

    st.title("1 · Chat — document context assistant")
    st.caption(
        "Start here. Upload project files (PDF/Word/Excel/TXT), then ask English questions. "
        "Uses models already in this project when available. **No API key.**"
    )

    with st.expander("Available models (this PC / project)", expanded=True):
        render_available_models_panel()

    qwen_dir = resolve_qwen_dir()
    local_ready = qwen_dir is not None

    with st.sidebar:
        st.header("Chat setup")
        st.success("API key: not required")

        mode_options = {
            "Extractive (begin here — low memory)": "extractive",
            "Local Qwen LLM (needs install)": "local_llm",
        }
        labels = list(mode_options.keys())
        current = st.session_state.get("answer_mode", "extractive")
        default_label = labels[0] if current != "local_llm" else labels[1]
        chosen = st.radio(
            "Answer engine",
            labels,
            index=labels.index(default_label),
            help="Extractive = best first step. Local Qwen = fluent answers from retrieved text.",
        )
        st.session_state.answer_mode = mode_options[chosen]
        st.caption(memory_caption())
        if st.button("Free model memory", use_container_width=True, key="chat_free_mem"):
            st.success(free_chatbot_model_memory())

        if st.session_state.answer_mode == "local_llm":
            if local_ready:
                st.info(f"Qwen ready:\n{qwen_dir}")
            else:
                st.warning("Qwen not installed yet.")
                st.caption("Open **Install yourself** → Beginner pack or Local LLM only.")
                if ensure_model is not None and st.button(
                    "Quick download Qwen", type="primary", use_container_width=True
                ):
                    with st.spinner("Downloading Local LLM..."):
                        try:
                            ensure_model(LOCAL_LLM_ID)
                            st.success("Downloaded. Ask again.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        st.markdown("---")
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "PDF / Word / Excel / CSV / TXT",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key="chat_uploads",
        )
        process_clicked = st.button(
            "Process documents", type="primary", use_container_width=True
        )
        st.button("Clear chat", on_click=reset_chat, use_container_width=True)
        st.button(
            "Clear documents", on_click=clear_knowledge_base, use_container_width=True
        )

        if st.session_state.doc_names:
            st.success("Loaded")
            for name in st.session_state.doc_names:
                st.write(f"- {name}")
            st.caption(f"Chunks: {st.session_state.chunk_count}")

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            with st.spinner("Reading files and building local search index..."):
                try:
                    all_chunks: List[Chunk] = []
                    names: List[str] = []
                    for uploaded in uploaded_files:
                        parts = load_uploaded_file(uploaded)
                        if not parts:
                            st.warning(f"No text from: {uploaded.name}")
                            continue
                        all_chunks.extend(parts)
                        names.append(uploaded.name)
                    if not all_chunks:
                        st.error("Could not extract text.")
                    else:
                        vectorizer, matrix = build_index(all_chunks)
                        st.session_state.chunks = all_chunks
                        st.session_state.vectorizer = vectorizer
                        st.session_state.matrix = matrix
                        st.session_state.doc_names = names
                        st.session_state.chunk_count = len(all_chunks)
                        st.session_state.messages = []
                        st.success(
                            f"Ready: {len(names)} file(s), {len(all_chunks)} chunk(s). Ask below."
                        )
                except Exception as exc:
                    st.error(f"Failed to process: {exc}")

    if not st.session_state.chunks:
        st.info(
            """
**Quick start**
1. Upload documents in the sidebar  
2. Click **Process documents**  
3. Ask questions in the chat box  

**Recommended path:** Chat (Extractive) → Learn → Install more models if you need NER/BERT demos.
"""
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    user_question = st.chat_input("Ask an English question about your document(s)...")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = answer_question(user_question)
                answer = str(result["answer"])
                sources = result.get("sources") or []
                mode = result.get("mode", "extractive")
                st.caption(f"Engine: {mode}")
                st.markdown(answer)
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"- {source}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                error_text = f"Could not answer right now. Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )
