import hashlib
import io
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

st.set_page_config(
    page_title="BERT & ALBERT Teaching Lab",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
MODELS = {
    "BERT": {
        "id": "bert-base-uncased",
        "title": "BERT",
        "full_name": "Bidirectional Encoder Representations from Transformers",
        "parameters": "~110M",
        "layers": "12",
        "hidden": "768",
        "heads": "12",
        "focus": "Bidirectional contextual language representation",
        "description": (
            "BERT is a Transformer encoder model that learns contextual "
            "representations of language. It reads context from both sides "
            "of a token."
        ),
        "training": "Masked Language Modeling + original Next Sentence Prediction",
        "architecture": "Transformer Encoder",
    },
    "ALBERT": {
        "id": "albert-base-v2",
        "title": "ALBERT",
        "full_name": "A Lite BERT",
        "parameters": "~12M",
        "layers": "12",
        "hidden": "768",
        "heads": "12",
        "focus": "Parameter-efficient BERT-style language representation",
        "description": (
            "ALBERT keeps the BERT-style encoder idea but reduces parameters "
            "through factorized embedding parameterization and cross-layer "
            "parameter sharing."
        ),
        "training": "Sentence Order Prediction + masked language modeling",
        "architecture": "Transformer Encoder",
    },
}

# ---------------------------------------------------------
# Streamlit-safe model loading
# Only one selected model is loaded at a time.
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def extract_document(uploaded_file):
    data = uploaded_file.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore"), sha256

    if name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, sha256

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, sha256

    raise ValueError("Please upload a TXT, PDF or DOCX file.")


def run_inference(model_id, text):
    tokenizer, model = load_model(model_id)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.inference_mode():
        outputs = model(**encoded)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    ids = encoded["input_ids"][0].tolist()
    hidden = outputs.last_hidden_state
    pooled = hidden.mean(dim=1)[0]

    return tokens, ids, hidden, pooled


def architecture_box(model_name):
    if model_name == "BERT":
        st.code(
"""INPUT TEXT
    ↓
TOKENIZER
    ↓
TOKEN IDs
    ↓
TOKEN EMBEDDINGS
    ↓
POSITION + SEGMENT INFORMATION
    ↓
12 × TRANSFORMER ENCODER
    ├── Multi-Head Self-Attention
    ├── Feed-Forward Network
    └── Normalization
    ↓
CONTEXTUAL REPRESENTATIONS
"""
        )
    else:
        st.code(
"""INPUT TEXT
    ↓
TOKENIZER
    ↓
FACTORISED EMBEDDING
    ↓
SHARED TRANSFORMER PARAMETERS
    ↕
Repeated across encoder layers
    ↓
CONTEXTUAL REPRESENTATIONS
"""
        )


def model_information(model_name):
    m = MODELS[model_name]

    st.subheader(f"About {model_name}")
    st.write(m["description"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", m["parameters"])
    c2.metric("Layers", m["layers"])
    c3.metric("Hidden size", m["hidden"])
    c4.metric("Attention heads", m["heads"])

    with st.expander("Training objective"):
        st.write(m["training"])

    with st.expander("Architecture"):
        st.write(m["architecture"])
        architecture_box(model_name)


def main():
    with st.sidebar:
        st.title("🧠 Transformer Lab")
        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🤖 Model Explorer",
                "⚖️ BERT vs ALBERT",
                "🔤 Tokenization Lab",
                "📖 Teaching Guide",
            ]
        )

        st.divider()
        st.caption("Streamlit-ready CPU demonstration")
        st.caption("Models are downloaded from Hugging Face on first use.")

    if page == "🏠 Home":
        st.title("🧠 BERT & ALBERT Teaching Laboratory")
        st.subheader("Learn → Experiment → Observe → Explain")

        st.write(
            "An interactive teaching application for demonstrating how "
            "Transformer encoder models process language."
        )

        c1, c2, c3 = st.columns(3)
        c1.info("📚 **Learn**\n\nUnderstand architecture and training.")
        c2.success("🧪 **Experiment**\n\nEnter text or upload a document.")
        c3.warning("🔍 **Observe**\n\nInspect tokens and representations.")

        st.divider()

        st.subheader("Available models")

        for name in MODELS:
            m = MODELS[name]
            with st.expander(name):
                st.write(m["full_name"])
                st.write(m["description"])

        st.subheader("Teaching methodology")
        st.code(
"""1. Introduce the NLP problem
2. Explain tokenization
3. Explain embeddings
4. Explain self-attention
5. Run a pretrained model
6. Inspect the tokens
7. Inspect the tensor representation
8. Compare BERT and ALBERT
9. Discuss model limitations
10. Introduce fine-tuning for task-specific applications
"""
        )

        st.warning(
            "Important: BERT and ALBERT are pretrained language representation "
            "models. Generic BERT is NOT automatically a phishing, forensic, "
            "sentiment or cybercrime classifier."
        )

    elif page == "🤖 Model Explorer":
        st.title("🤖 Model Explorer")

        selected = st.selectbox(
            "Select the model",
            list(MODELS.keys())
        )

        model_information(selected)

        st.divider()

        st.subheader("Your input")

        uploaded = st.file_uploader(
            "Upload TXT, PDF or DOCX",
            type=["txt", "pdf", "docx"]
        )

        default_text = (
            "Artificial intelligence is increasingly being used "
            "in cybersecurity and digital forensic investigations."
        )

        text = st.text_area(
            "Or type / paste text or a question",
            value=default_text,
            height=180
        )

        if uploaded is not None:
            try:
                extracted, sha256 = extract_document(uploaded)

                if extracted.strip():
                    text = extracted
                    st.success(f"Loaded: {uploaded.name}")
                    st.caption(f"SHA-256: {sha256}")

                    with st.expander("View extracted text"):
                        st.text(extracted[:12000])

            except Exception as e:
                st.error(f"File processing failed: {e}")

        if st.button("🚀 Run selected model", type="primary",
                     use_container_width=True):

            if not text.strip():
                st.warning("Please enter some text or upload a document.")
                st.stop()

            try:
                with st.spinner(
                    f"Downloading/loading {selected} and running inference..."
                ):
                    tokens, ids, hidden, pooled = run_inference(
                        MODELS[selected]["id"], text
                    )

                st.success(f"{selected} inference completed.")

                st.subheader("1️⃣ Tokenization")

                token_rows = []
                for i, (tok, token_id) in enumerate(
                    zip(tokens, ids)
                ):
                    token_rows.append(
                        {
                            "Position": i,
                            "Token": tok,
                            "Token ID": token_id
                        }
                    )

                st.dataframe(
                    token_rows,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"{len(tokens)} tokens were generated "
                    "(maximum 256 tokens for this demonstration)."
                )

                st.subheader("2️⃣ Transformer representation")

                st.write(
                    f"Output tensor shape: "
                    f"**{tuple(hidden.shape)}**"
                )

                st.write(
                    "This represents: **batch × tokens × hidden dimensions**."
                )

                st.subheader("3️⃣ Example vector")

                vector_rows = [
                    {
                        "Dimension": i + 1,
                        "Value": round(float(v), 6)
                    }
                    for i, v in enumerate(pooled[:20])
                ]

                st.dataframe(
                    vector_rows,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("4️⃣ What happened?")

                st.markdown(
"""**Your text**

↓  

**Tokenizer**

The text was split into tokens.

↓

**Token IDs**

Each token was represented by a numerical ID.

↓

**Embeddings**

The IDs were converted into numerical vectors.

↓

**Transformer encoder**

Self-attention allowed the model to construct contextual representations.

↓

**Output**

The model produced a vector representation for each input token.
"""
                )

                if selected == "ALBERT":
                    st.success(
                        "ALBERT teaching point: it follows the BERT-style "
                        "encoder approach but uses parameter-sharing and "
                        "factorized embeddings to reduce parameters."
                    )
                else:
                    st.success(
                        "BERT teaching point: each token representation "
                        "is contextual and can use information from "
                        "both directions of the input."
                    )

            except Exception as e:
                st.error(f"Model execution failed: {e}")
                st.info(
                    "On Streamlit Cloud, this normally means the model could "
                    "not be downloaded or the application ran out of memory. "
                    "Try a shorter input and confirm that the deployment "
                    "has internet access."
                )

    elif page == "⚖️ BERT vs ALBERT":
        st.title("⚖️ BERT vs ALBERT")

        st.table(
            {
                "Feature": [
                    "Architecture",
                    "Bidirectional context",
                    "Layers",
                    "Approx. parameters",
                    "Hidden size",
                    "Attention heads",
                    "Key idea",
                ],
                "BERT": [
                    "Transformer Encoder",
                    "Yes",
                    "12",
                    "~110M",
                    "768",
                    "12",
                    "Contextual representation",
                ],
                "ALBERT": [
                    "Transformer Encoder",
                    "Yes",
                    "12",
                    "~12M",
                    "768",
                    "12",
                    "Parameter efficiency",
                ],
            }
        )

        st.subheader("Why ALBERT was developed")

        st.markdown(
"""BERT can contain a large number of parameters.

ALBERT reduces this through two important ideas:

**1. Factorized embedding parameterization**

Large vocabulary embeddings are separated from the hidden representation size.

**2. Cross-layer parameter sharing**

The same Transformer parameters can be reused across layers.

Therefore:

```text
BERT
Layer 1 → Parameters A
Layer 2 → Parameters B
Layer 3 → Parameters C
...

ALBERT
Shared parameters
      ↓
Layer 1
Layer 2
Layer 3
...
```
"""
        )

    elif page == "🔤 Tokenization Lab":
        st.title("🔤 Tokenization Laboratory")

        selected = st.selectbox(
            "Choose tokenizer",
            ["BERT", "ALBERT"]
        )

        text = st.text_area(
            "Enter a sentence",
            "Digital forensics helps investigators analyze electronic evidence."
        )

        if st.button("Tokenize", type="primary"):
            try:
                tokenizer, _ = load_model(
                    MODELS[selected]["id"]
                )

                tokens = tokenizer.tokenize(text)
                ids = tokenizer.convert_tokens_to_ids(tokens)

                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Tokens")
                    st.code(" | ".join(tokens))

                with c2:
                    st.subheader("Token IDs")
                    st.code(str(ids))

                st.metric("Number of tokens", len(tokens))

                st.info(
                    "A Transformer does not directly process English words. "
                    "Text is converted into tokens and numerical IDs first."
                )

            except Exception as e:
                st.error(f"Tokenizer failed: {e}")

    else:
        st.title("📖 Teaching Guide")

        st.markdown(
"""
## Suggested 60–90 minute demonstration

### Part 1 — Problem

Why can't a computer directly understand a sentence?

### Part 2 — Tokenization

Use the Tokenization Lab and demonstrate that text becomes tokens and IDs.

### Part 3 — Embeddings

Explain how token IDs become numerical representations.

### Part 4 — Self-attention

Explain how the representation of a word depends on surrounding context.

### Part 5 — BERT

Run BERT with a sentence supplied by the students.

### Part 6 — Inspect

Show the tokens, tensor dimensions and example vector.

### Part 7 — ALBERT

Run the same input through ALBERT.

### Part 8 — Compare

Discuss why ALBERT can have substantially fewer parameters.

### Part 9 — Beyond pretrained models

Explain that a pretrained encoder is a foundation for many applications.
For a specific classification task, it normally needs an appropriate
task head and/or fine-tuning using labelled examples.
"""
        )


if __name__ == "__main__":
    main()
