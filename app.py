"""
Document Context Chatbot
------------------------
Answers English questions using ONLY the content of uploaded documents
(PDF, Word, Excel, CSV, plain text).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(
    page_title="Document Context Chatbot",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_TYPES = ["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md"]

SYSTEM_PROMPT = """You are a careful document Q&A assistant.

Rules:
1. Answer ONLY using the retrieved document context below.
2. The user asks questions in English. Reply in clear English.
3. If the answer is not present in the context, say exactly:
   "I could not find that information in the uploaded document(s)."
4. Do not invent facts, names, numbers, or dates.
5. Prefer short, direct answers. Use bullet points when listing items.
6. When helpful, mention which part of the document the answer comes from.

Context:
{context}
"""


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "rag_chain": None,
        "doc_names": [],
        "chunk_count": 0,
        "processed_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_excel_as_documents(file_path: str, source_name: str) -> List[Document]:
    """Convert each Excel sheet into text documents."""
    documents: List[Document] = []
    workbook = pd.read_excel(file_path, sheet_name=None, dtype=str)

    for sheet_name, frame in workbook.items():
        frame = frame.fillna("")
        if frame.empty:
            continue

        # Keep tabular meaning while remaining searchable as text.
        header = " | ".join(str(col) for col in frame.columns)
        rows = [" | ".join(str(value) for value in row) for row in frame.values.tolist()]
        text = f"Spreadsheet: {source_name}\nSheet: {sheet_name}\nColumns: {header}\n" + "\n".join(rows)
        documents.append(
            Document(
                page_content=text,
                metadata={"source": source_name, "sheet": sheet_name, "type": "excel"},
            )
        )
    return documents


def load_csv_as_documents(file_path: str, source_name: str) -> List[Document]:
    frame = pd.read_csv(file_path, dtype=str).fillna("")
    if frame.empty:
        return []

    header = " | ".join(str(col) for col in frame.columns)
    rows = [" | ".join(str(value) for value in row) for row in frame.values.tolist()]
    text = f"CSV file: {source_name}\nColumns: {header}\n" + "\n".join(rows)
    return [Document(page_content=text, metadata={"source": source_name, "type": "csv"})]


def load_uploaded_file(uploaded_file) -> List[Document]:
    """Save upload temporarily and load it with the right parser."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif suffix in {".docx", ".doc"}:
            docs = Docx2txtLoader(tmp_path).load()
        elif suffix in {".xlsx", ".xls"}:
            docs = load_excel_as_documents(tmp_path, uploaded_file.name)
        elif suffix == ".csv":
            docs = load_csv_as_documents(tmp_path, uploaded_file.name)
        elif suffix in {".txt", ".md"}:
            docs = TextLoader(tmp_path, encoding="utf-8").load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_rag_chain(documents: List[Document], api_key: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(documents)
    if not splits:
        raise ValueError("No readable text was found in the uploaded file(s).")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain, len(splits)


def reset_chat() -> None:
    st.session_state.messages = []


def clear_knowledge_base() -> None:
    st.session_state.messages = []
    st.session_state.rag_chain = None
    st.session_state.doc_names = []
    st.session_state.chunk_count = 0
    st.session_state.processed_key = None


def main() -> None:
    init_session_state()

    st.title("📚 Document Context Chatbot")
    st.caption(
        "Upload PDF, Word, Excel, CSV, or text files. Ask questions in English. "
        "Answers are generated only from your uploaded content."
    )

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Required for embeddings and answers.")
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
        st.markdown("**Grounding rule**")
        st.write("If it is not in the document, the bot will say it does not know.")

        if st.session_state.doc_names:
            st.markdown("---")
            st.success("Loaded documents")
            for name in st.session_state.doc_names:
                st.write(f"• {name}")
            st.caption(f"Indexed chunks: {st.session_state.chunk_count}")

    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to begin.")
        st.stop()

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            key = tuple((f.name, f.size) for f in uploaded_files)
            with st.spinner("Reading documents and building the knowledge base..."):
                try:
                    all_docs: List[Document] = []
                    names: List[str] = []
                    for uploaded in uploaded_files:
                        docs = load_uploaded_file(uploaded)
                        if not docs:
                            st.warning(f"No text extracted from: {uploaded.name}")
                            continue
                        all_docs.extend(docs)
                        names.append(uploaded.name)

                    if not all_docs:
                        st.error("Could not extract text from the uploaded file(s).")
                    else:
                        chain, chunk_count = build_rag_chain(all_docs, api_key)
                        st.session_state.rag_chain = chain
                        st.session_state.doc_names = names
                        st.session_state.chunk_count = chunk_count
                        st.session_state.processed_key = key
                        st.session_state.messages = []
                        st.success(
                            f"Processed {len(names)} file(s) into {chunk_count} searchable chunk(s). "
                            "You can now ask English questions about the content."
                        )
                except Exception as exc:
                    st.error(f"Failed to process documents: {exc}")

    if st.session_state.rag_chain is None:
        st.markdown(
            """
            ### How to use
            1. Enter your OpenAI API key in the sidebar.
            2. Upload one or more documents (PDF, Word, Excel, CSV, TXT).
            3. Click **Process documents**.
            4. Ask English questions in the chat box.

            The chatbot will answer only from the uploaded context.
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
                response = st.session_state.rag_chain.invoke({"input": user_question})
                answer = response.get("answer", "").strip() or "I could not find that information in the uploaded document(s)."

                source_docs = response.get("context", []) or []
                sources = []
                for doc in source_docs:
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page")
                    sheet = doc.metadata.get("sheet")
                    label = source
                    if page is not None:
                        label += f" (page {int(page) + 1})"
                    if sheet:
                        label += f" (sheet: {sheet})"
                    snippet = " ".join(doc.page_content.split())[:180]
                    sources.append(f"**{label}**: {snippet}...")

                st.markdown(answer)
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"- {source}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_text = f"Sorry, I could not answer that right now. Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
