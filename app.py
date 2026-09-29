import io
import re
import hashlib
from collections import Counter

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)

st.set_page_config(
    page_title="Transformer AI Teaching Lab",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# MODEL CATALOG
# ============================================================
MODELS = {
    "BERT": {
        "base": "bert-base-uncased",
        "qa": "deepset/bert-base-cased-squad2",
        "ner": "dslim/bert-base-NER",
        "kind": "Encoder",
        "description": "Bidirectional Transformer encoder for contextual language understanding.",
        "parameters": "~110M",
        "layers": "12",
        "hidden": "768",
        "heads": "12",
        "strength": "Contextual text understanding",
    },
    "ALBERT": {
        "base": "albert-base-v2",
        "qa": "twmkn9/albert-base-v2-squad2",
        "ner": "Jean-Baptiste/albert-xxlarge-v2-ner",
        "kind": "Encoder",
        "description": "BERT-style encoder designed to reduce parameters through parameter sharing and factorized embeddings.",
        "parameters": "~12M for base-v2",
        "layers": "12",
        "hidden": "768",
        "heads": "12",
        "strength": "Parameter efficiency",
    },
    "RoBERTa": {
        "base": "roberta-base",
        "qa": "deepset/roberta-base-squad2",
        "ner": "Jean-Baptiste/roberta-large-ner-english",
        "kind": "Encoder",
        "description": "BERT-style encoder trained with an improved pre-training recipe.",
        "parameters": "~125M",
        "layers": "12",
        "hidden": "768",
        "heads": "12",
        "strength": "Robust contextual representations",
    },
}

NER_MODELS = {
    "BERT NER": "dslim/bert-base-NER",
    "RoBERTa NER": "Jean-Baptiste/roberta-large-ner-english",
    "DistilBERT NER": "elastic/distilbert-base-uncased-finetuned-conll03-english",
}

QA_MODELS = {
    "BERT QA": "deepset/bert-base-cased-squad2",
    "ALBERT QA": "twmkn9/albert-base-v2-squad2",
    "RoBERTa QA": "deepset/roberta-base-squad2",
}

GEN_MODEL = "google/flan-t5-small"

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_encoder(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_qa(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_ner(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_generation():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    model.eval()
    return tokenizer, model


# ============================================================
# FILE HANDLING
# ============================================================
def extract_file(uploaded):
    data = uploaded.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    name = uploaded.name.lower()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore"), sha256

    if name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n[Page {page_no}]\n{page_text}")
        text = "\n".join(pages)
        return text, sha256

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, sha256

    raise ValueError("Supported files: TXT, PDF, DOCX.")


# ============================================================
# QA
# ============================================================
def split_context(text, max_chars=3200, overlap=350):
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + 100:
                end = boundary

        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def answer_question(model_id, question, context):
    """Extractive QA over manageable text chunks.

    Important Streamlit/Transformers fix:
    - Tokenization uses padding='max_length' with return_tensors='pt'
      so overflow features have consistent tensor dimensions.
    - offset mappings are retained for answer-span extraction.
    """
    tokenizer, model = load_qa(model_id)
    chunks = split_context(context)
    candidates = []

    for chunk_index, chunk in enumerate(chunks):
        try:
            encoded = tokenizer(
                question,
                chunk,
                return_tensors="pt",
                truncation="only_second",
                max_length=384,
                stride=96,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
        except Exception:
            # Some tokenizer/model combinations can be more robust when
            # overflow features are created without an immediate tensor
            # conversion. Pad each feature below if necessary.
            raw = tokenizer(
                question,
                chunk,
                truncation="only_second",
                max_length=384,
                stride=96,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            encoded = tokenizer.pad(raw, return_tensors="pt")

        feature_count = encoded["input_ids"].shape[0]

        for feature_index in range(feature_count):
            inputs = {
                "input_ids": encoded["input_ids"][feature_index:feature_index+1],
                "attention_mask": encoded["attention_mask"][feature_index:feature_index+1],
            }

            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"][feature_index:feature_index+1]

            with torch.inference_mode():
                output = model(**inputs)

            start_logits = output.start_logits[0]
            end_logits = output.end_logits[0]
            offsets = encoded["offset_mapping"][feature_index].tolist()
            sequence_ids = encoded.sequence_ids(feature_index)

            context_positions = [
                i for i, sid in enumerate(sequence_ids) if sid == 1
            ]

            if not context_positions:
                continue

            start_top = torch.topk(
                start_logits[context_positions],
                min(12, len(context_positions))
            )
            end_top = torch.topk(
                end_logits[context_positions],
                min(12, len(context_positions))
            )

            for sv, sl in zip(
                start_top.values.tolist(),
                start_top.indices.tolist()
            ):
                s_pos = context_positions[sl]

                for ev, el in zip(
                    end_top.values.tolist(),
                    end_top.indices.tolist()
                ):
                    e_pos = context_positions[el]

                    if e_pos < s_pos or e_pos - s_pos > 30:
                        continue

                    start_char = offsets[s_pos][0]
                    end_char = offsets[e_pos][1]

                    if end_char <= start_char:
                        continue

                    answer = chunk[int(start_char):int(end_char)].strip()

                    if answer:
                        candidates.append({
                            "answer": answer,
                            "score": float(sv + ev),
                            "context": chunk,
                            "chunk": chunk_index,
                        })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)

    top = candidates[0]
    scores = torch.tensor([x["score"] for x in candidates[:20]])
    probs = torch.softmax(scores - scores.max(), dim=0)
    top["confidence"] = float(probs[0]) * 100

    return top


# ============================================================
# NER
# ============================================================
def run_ner(model_id, text):
    tokenizer, model = load_ner(model_id)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.inference_mode():
        output = model(**encoded)

    probabilities = torch.softmax(output.logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)[0]

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    word_ids = encoded.word_ids(batch_index=0)

    entities = []
    current = None

    for i, token in enumerate(tokens):
        if word_ids[i] is None:
            continue

        label = model.config.id2label[int(predictions[i])]
        confidence = float(probabilities[0, i, predictions[i]])

        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix, entity_type = "B", label

        clean_token = token.replace("##", "")

        if prefix == "B" or current is None or current["type"] != entity_type:
            if current:
                entities.append(current)

            current = {
                "type": entity_type,
                "text": clean_token,
                "confidence": confidence,
            }
        else:
            if token.startswith("##"):
                current["text"] += clean_token
            else:
                current["text"] += " " + clean_token

            current["confidence"] = (
                current["confidence"] + confidence
            ) / 2

    if current:
        entities.append(current)

    return entities


# ============================================================
# GENERATION / CHAT
# ============================================================
def generate_answer(prompt):
    tokenizer, model = load_generation()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================================
# LEARNING MODULE
# ============================================================
def learning_module():
    st.title("📚 Learning Module")

    topic = st.selectbox(
        "Choose a topic",
        [
            "What are Transformers?",
            "BERT",
            "ALBERT",
            "Named Entity Recognition (NER)",
            "Question Answering",
            "Tokenization",
            "Self-Attention",
            "Embeddings",
            "BERT vs ALBERT",
            "Encoder vs Generative Models",
        ]
    )

    if topic == "What are Transformers?":
        st.header("What are Transformers?")
        st.write(
            "Transformers are neural-network architectures built around "
            "attention mechanisms. They allow a model to process relationships "
            "between tokens efficiently."
        )
        st.code(
"""Text
 ↓
Tokenizer
 ↓
Embeddings
 ↓
Self-Attention
 ↓
Transformer Layers
 ↓
Task-specific output"""
        )

    elif topic == "BERT":
        st.header("BERT")
        st.write(
            "BERT is a bidirectional Transformer encoder. It was designed "
            "primarily for language understanding rather than free-form generation."
        )
        st.subheader("Key ideas")
        st.markdown(
            "- Bidirectional contextual representations\n"
            "- Transformer encoder\n"
            "- Masked Language Modeling\n"
            "- Fine-tuning for downstream tasks\n"
            "- NER, classification, QA and other applications"
        )
        st.subheader("Teaching architecture")
        st.code(
"""Input
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer Encoder × 12
 ↓
Contextual representations
 ↓
Task-specific head"""
        )

    elif topic == "ALBERT":
        st.header("ALBERT — A Lite BERT")
        st.write(
            "ALBERT retains the BERT-style Transformer encoder while "
            "reducing parameters through factorized embeddings and "
            "cross-layer parameter sharing."
        )
        st.subheader("Why ALBERT?")
        st.markdown(
            "- Reduce parameter count\n"
            "- Reduce memory requirements\n"
            "- Reuse parameters across layers\n"
            "- Maintain the Transformer encoder approach"
        )
        st.code(
"""BERT:
Layer 1 → A
Layer 2 → B
Layer 3 → C
...

ALBERT:
Shared parameters
       ↓
Layer 1
Layer 2
Layer 3
..."""
        )

    elif topic == "Named Entity Recognition (NER)":
        st.header("Named Entity Recognition — NER")
        st.write(
            "NER identifies important entities in text and assigns an entity type."
        )
        st.code(
"""Input:
"Rahul visited NFSU Goa in Ponda."

NER:

Rahul       → PERSON
NFSU Goa    → ORGANIZATION
Ponda       → LOCATION"""
        )
        st.markdown(
            """
Typical entity classes include:

- **PER** — Person
- **ORG** — Organization
- **LOC** — Location
- **MISC** — Miscellaneous entity

NER is useful for information extraction, document analysis,
cybersecurity and forensic text processing.
"""
        )

    elif topic == "Question Answering":
        st.header("Question Answering")
        st.write(
            "Extractive QA models identify an answer span from supplied context."
        )
        st.code(
"""Context
   +
Question
   ↓
Transformer Encoder
   ↓
Start position + End position
   ↓
Answer span"""
        )

    elif topic == "Tokenization":
        st.header("Tokenization")
        st.write(
            "Tokenization converts text into tokens that can be mapped to numerical IDs."
        )
        st.code(
""""The investigator examined the phone."

       ↓

["The", "investigator", "examined", ...]
       ↓
[101, 1996, 12345, ...]
       ↓
Embeddings"""
        )

    elif topic == "Self-Attention":
        st.header("Self-Attention")
        st.write(
            "Self-attention allows a token to assign different importance "
            "to other tokens in the sequence."
        )
        st.code(
"""The suspect went to the bank because he needed money.

"bank" attends to:
    suspect
    went
    money
    ..."""
        )

    elif topic == "Embeddings":
        st.header("Embeddings")
        st.write(
            "Embeddings represent tokens, sentences or documents as numerical vectors."
        )
        st.code(
"""Text
 ↓
Vector

[0.21, -0.44, 0.83, ...]"""
        )

    elif topic == "BERT vs ALBERT":
        st.header("BERT vs ALBERT")
        st.table(
            {
                "Feature": [
                    "Architecture",
                    "Parameters",
                    "Main idea",
                    "Bidirectional",
                    "Typical tasks",
                ],
                "BERT": [
                    "Transformer Encoder",
                    "~110M",
                    "Contextual representation",
                    "Yes",
                    "NER, QA, classification",
                ],
                "ALBERT": [
                    "Transformer Encoder",
                    "~12M base-v2",
                    "Parameter sharing",
                    "Yes",
                    "NER, QA, classification",
                ],
            }
        )

    else:
        st.header("Encoder vs Generative Models")
        st.write(
            "Encoder models such as BERT are primarily designed to understand "
            "and represent input. Generative models produce new sequences."
        )
        st.code(
"""BERT / ALBERT
Input → Encoder → Representation → Task

T5 / FLAN-T5
Input → Encoder/Decoder → Generated text"""
        )


# ============================================================
# APPLICATION MODULE
# ============================================================
def application_module():
    st.title("🧪 Application Module")

    task = st.selectbox(
        "Select application",
        [
            "🔎 Question Answering",
            "🏷️ Named Entity Recognition",
            "🧠 Text Representation",
            "💬 Chat / General AI",
        ]
    )

    if task == "🔎 Question Answering":
        question_answering_app()

    elif task == "🏷️ Named Entity Recognition":
        ner_app()

    elif task == "🧠 Text Representation":
        representation_app()

    else:
        chat_app()


def input_area(default):
    uploaded = st.file_uploader(
        "Upload TXT, PDF or DOCX",
        type=["txt", "pdf", "docx"],
        key=st.session_state.get("input_key", "file")
    )

    text = st.text_area(
        "Or type / paste your input",
        value=default,
        height=180,
    )

    if uploaded:
        try:
            extracted, sha = extract_file(uploaded)
            if extracted.strip():
                text = extracted
                st.success(f"Loaded {uploaded.name}")
                st.caption(f"SHA-256: {sha}")
                with st.expander("View extracted text"):
                    st.text(extracted[:12000])
        except Exception as e:
            st.error(f"File processing failed: {e}")

    return text


def question_answering_app():
    st.header("🔎 Question Answering")

    model_name = st.selectbox(
        "Select QA model",
        list(QA_MODELS.keys())
    )

    context = input_area(
        "National Forensic Sciences University was established in 2020. "
        "The university focuses on forensic science and cybersecurity."
    )

    question = st.text_input(
        "Question",
        "When was National Forensic Sciences University established?"
    )

    if st.button("Find Answer", type="primary", use_container_width=True):
        if not context.strip() or not question.strip():
            st.warning("Provide both context and a question.")
            return

        with st.spinner("Running Question Answering model..."):
            result = answer_question(
                QA_MODELS[model_name],
                question,
                context
            )

        if result:
            a, b = st.columns([3, 1])
            a.success(result["answer"])
            b.metric("Score indicator", f"{result['confidence']:.1f}%")

            st.subheader("Supporting context")
            st.write(result["context"])

            with st.expander("How did the model answer?"):
                st.write(
                    "The fine-tuned QA head predicts likely start and end "
                    "positions of an answer span in the context."
                )
        else:
            st.warning("No answer span was found.")


def ner_app():
    st.header("🏷️ Named Entity Recognition")

    model_name = st.selectbox(
        "Select NER model",
        list(NER_MODELS.keys())
    )

    text = input_area(
        "Dr. Rahul Sharma visited National Forensic Sciences University "
        "in Goa on 15 September 2026."
    )

    if st.button("Extract Entities", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Provide text or upload a document.")
            return

        with st.spinner("Running NER model..."):
            entities = run_ner(
                NER_MODELS[model_name],
                text
            )

        if not entities:
            st.info("No entities were identified.")
            return

        st.subheader("Extracted entities")

        rows = []
        for entity in entities:
            rows.append(
                {
                    "Entity": entity["text"],
                    "Type": entity["type"],
                    "Confidence": f"{entity['confidence']*100:.1f}%"
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

        counts = Counter(e["type"] for e in entities)

        st.subheader("Entity summary")
        st.write(dict(counts))

        st.subheader("How NER works")

        st.code(
"""Input text
    ↓
Tokenizer
    ↓
Transformer Encoder
    ↓
Token classification
    ↓
PERSON / ORG / LOC / MISC
"""
        )


def representation_app():
    st.header("🧠 Text Representation")

    model_name = st.selectbox(
        "Select encoder",
        ["BERT", "ALBERT", "RoBERTa"]
    )

    text = input_area(
        "Artificial intelligence is increasingly used in cybersecurity."
    )

    if st.button("Generate Representation", type="primary",
                 use_container_width=True):

        if not text.strip():
            st.warning("Provide text.")
            return

        with st.spinner("Generating contextual representation..."):
            tokenizer, model = load_encoder(
                MODELS[model_name]["base"]
            )

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            with torch.inference_mode():
                output = model(**encoded)

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0]
        )

        st.success("Representation generated.")

        st.write(
            f"Output tensor shape: **{tuple(output.last_hidden_state.shape)}**"
        )

        st.subheader("Tokens")
        st.code(" | ".join(tokens))

        pooled = output.last_hidden_state.mean(dim=1)[0]

        st.subheader("Example vector dimensions")

        st.dataframe(
            {
                "Dimension": list(range(1, 21)),
                "Value": [
                    round(float(x), 6)
                    for x in pooled[:20]
                ]
            },
            use_container_width=True,
            hide_index=True
        )


def chat_app():
    st.header("💬 Chat / General AI")

    st.info(
        "This mode uses FLAN-T5-small for text generation. "
        "BERT and ALBERT are encoder models and are therefore shown "
        "separately in the QA, NER and representation applications."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(
        "Ask a general question..."
    )

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    answer = generate_answer(
                        "answer: " + prompt
                    )
                    st.write(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    with st.sidebar:
        st.title("🧠 Transformer AI Lab")

        module = st.radio(
            "Select Module",
            [
                "📚 Learning Module",
                "🧪 Application Module",
            ]
        )

        st.divider()

        st.caption(
            "Teaching + hands-on Transformer experimentation"
        )

    if module == "📚 Learning Module":
        learning_module()
    else:
        application_module()


if __name__ == "__main__":
    main()
