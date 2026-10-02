"""
Document Context Chatbot (no API key required)
----------------------------------------------
Answers English questions using ONLY uploaded document content
(PDF, Word, Excel, CSV, TXT/MD).

Retrieval: TF-IDF (scikit-learn)
Answering:
  1) Local Qwen2.5-0.5B-Instruct if present in local_models/
  2) Otherwise extractive answers from the best matching passages
"""

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

try:
    import docx2txt
except Exception:  # pragma: no cover
    docx2txt = None

try:
    from model_manager import is_ready, local_path
except Exception:  # pragma: no cover - Cloud can still run extractive mode
    def is_ready(model_id: str) -> bool:
        return False

    def local_path(model_id: str) -> Path:
        return Path("local_models") / model_id.replace("/", "__")

st.set_page_config(
    page_title="Document Context Chatbot",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_TYPES = ["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md"]
LOCAL_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
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


def chunk_text(text: str, source: str, page: Optional[int] = None, sheet: Optional[str] = None,
               chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
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
        text = f"Spreadsheet: {source}\nSheet: {sheet_name}\nColumns: {header}\n" + "\n".join(rows)
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


def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, any]:
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
    if not hits:
        return NO_ANSWER
    # Very weak match → treat as missing
    if hits[0][1] < 0.05:
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
        "_No API key is used. This extractive mode returns the closest passages from your files._"
    )
    return "\n".join(lines).strip()


@st.cache_resource(show_spinner=False)
def load_local_llm():
    """Load preloaded Qwen model from local_models/ when available."""
    if not is_ready(LOCAL_LLM_ID):
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_dir = str(local_path(LOCAL_LLM_ID))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def local_llm_answer(question: str, hits: List[Tuple[Chunk, float]]) -> Optional[str]:
    if not hits or hits[0][1] < 0.05:
        return NO_ANSWER
    try:
        tokenizer, model = load_local_llm()
    except Exception:
        return None
    if tokenizer is None or model is None:
        return None

    import torch

    context = "\n\n".join(
        f"Source: {label_chunk(chunk)}\n{chunk.text}" for chunk, _ in hits[:4]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Answer ONLY from the supplied context. "
                "Reply in clear English. If the answer is not in the context, say exactly: "
                f"{NO_ANSWER}"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text or NO_ANSWER


def answer_question(question: str) -> Dict[str, object]:
    hits = retrieve(question, top_k=4)
    sources = []
    for chunk, score in hits:
        snippet = " ".join(chunk.text.split())[:180]
        sources.append(f"**{label_chunk(chunk)}** (score {score:.2f}): {snippet}...")

    mode = st.session_state.answer_mode
    answer = None
    used_mode = "extractive"

    if mode == "local_llm":
        answer = local_llm_answer(question, hits)
        if answer is not None:
            used_mode = "local_llm"

    if answer is None:
        answer = extractive_answer(question, hits)
        used_mode = "extractive"

    return {"answer": answer, "sources": sources, "mode": used_mode}


def reset_chat() -> None:
    st.session_state.messages = []


def clear_knowledge_base() -> None:
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.matrix = None
    st.session_state.doc_names = []
    st.session_state.chunk_count = 0


def main() -> None:
    init_session_state()

    local_ready = is_ready(LOCAL_LLM_ID)

    st.title("📚 Document Context Chatbot")
    st.caption(
        "No API key required. Upload documents and ask English questions. "
        "Answers come only from your uploaded content."
    )

    with st.sidebar:
        st.header("Setup")
        st.success("API key: not required")

        default_mode = "local_llm" if local_ready else "extractive"
        mode_options = {
            "Extractive (works everywhere, no model download)": "extractive",
            "Local Qwen LLM (uses preloaded local_models)": "local_llm",
        }
        labels = list(mode_options.keys())
        default_label = labels[0] if default_mode == "extractive" else labels[1]
        chosen = st.radio(
            "Answer mode",
            labels,
            index=labels.index(default_label),
            help="Extractive mode always works. Local LLM needs Qwen in local_models/.",
        )
        st.session_state.answer_mode = mode_options[chosen]

        if st.session_state.answer_mode == "local_llm":
            if local_ready:
                st.info("Local Qwen model is ready.")
            else:
                st.warning(
                    "Local Qwen is not in local_models/. "
                    "Run: python scripts/preload_all_models.py — or use Extractive mode."
                )

        st.markdown("---")
        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "PDF / Word / Excel / CSV / TXT",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
        )
        process_clicked = st.button("Process documents", type="primary", use_container_width=True)
        st.button("Clear chat", on_click=reset_chat, use_container_width=True)
        st.button("Clear documents", on_click=clear_knowledge_base, use_container_width=True)

        st.markdown("---")
        st.markdown("**Supported formats**")
        st.write("`.pdf` `.docx` `.xlsx` `.xls` `.csv` `.txt` `.md`")
        st.markdown("**Language**")
        st.write("Questions and answers: English")

        if st.session_state.doc_names:
            st.markdown("---")
            st.success("Loaded documents")
            for name in st.session_state.doc_names:
                st.write(f"• {name}")
            st.caption(f"Indexed chunks: {st.session_state.chunk_count}")

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Reading documents and building a local search index..."):
                try:
                    all_chunks: List[Chunk] = []
                    names: List[str] = []
                    for uploaded in uploaded_files:
                        parts = load_uploaded_file(uploaded)
                        if not parts:
                            st.warning(f"No text extracted from: {uploaded.name}")
                            continue
                        all_chunks.extend(parts)
                        names.append(uploaded.name)

                    if not all_chunks:
                        st.error("Could not extract text from the uploaded file(s).")
                    else:
                        vectorizer, matrix = build_index(all_chunks)
                        st.session_state.chunks = all_chunks
                        st.session_state.vectorizer = vectorizer
                        st.session_state.matrix = matrix
                        st.session_state.doc_names = names
                        st.session_state.chunk_count = len(all_chunks)
                        st.session_state.messages = []
                        st.success(
                            f"Processed {len(names)} file(s) into {len(all_chunks)} chunk(s). "
                            "Ask English questions below."
                        )
                except Exception as exc:
                    st.error(f"Failed to process documents: {exc}")

    if not st.session_state.chunks:
        st.markdown(
            """
            ### How to use (no API key)
            1. Upload one or more documents (PDF, Word, Excel, CSV, TXT).
            2. Click **Process documents**.
            3. Ask English questions in the chat box.

            **Extractive mode** works on Streamlit Cloud with no keys and no GPU.  
            **Local Qwen mode** uses your preloaded model under `local_models/` (best on your PC).
            """
        )
        st.stop()

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
                st.caption(f"Answer mode: {mode}")
                st.markdown(answer)
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"- {source}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                error_text = f"Sorry, I could not answer that right now. Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
