
import os, sys, time
from pathlib import Path
import streamlit as st
import json
from model_manager import (
    CATALOGUE,
    ensure_model,
    ensure_all_models,
    local_path,
    is_ready,
    status_report,
    all_models,
)

st.set_page_config(page_title="Hugging Face BERT, NER & Local LLM Lab",
                   page_icon="🤗", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
LOCAL_MODEL = local_path("Qwen/Qwen2.5-0.5B-Instruct")

st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1450px}
[data-testid="stSidebar"]{min-width:285px;max-width:300px}
.panel{border:1px solid rgba(128,128,128,.25);border-radius:15px;padding:17px;
background:rgba(128,128,128,.045);margin-bottom:12px}
.entity{display:inline-block;padding:6px 9px;margin:3px;border-radius:8px;
background:rgba(0,120,255,.12)}
.small{opacity:.72;font-size:.86rem}
</style>
""", unsafe_allow_html=True)

DEMO_NER = ("Dr. Ranjit Kolkar visited National Forensic Sciences University in Goa "
            "on 12 September 2026 for a Digital Forensics workshop. "
            "The team later travelled to New Delhi to meet officials from the Ministry of Home Affairs.")
DEMO_Q = "What should a forensic report contain?"
DEMO_CONTEXT = """A forensic report should describe the scope of the examination, tools and versions used,
the acquisition method, examination steps, findings, limitations, and supporting evidence.
Screenshots, hash values, logs, and other reproducibility information can help another examiner
understand how the findings were obtained."""

NER_MODELS = {
    "BERT NER — dslim": "dslim/bert-base-NER",
    "DistilBERT NER": "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "RoBERTa NER": "Jean-Baptiste/roberta-large-ner-english",
}
BERT_MODELS = {
    "BERT base uncased": "google-bert/bert-base-uncased",
    "BERT base cased": "google-bert/bert-base-cased",
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
    "ALBERT": "albert/albert-base-v2",
}


def prepare_selected_model(model_id, progress=None):
    """Use an existing project model; otherwise automatically download it."""
    if is_ready(model_id):
        if progress:
            progress.progress(45, "Using model already stored in project…")
        return local_path(model_id)
    if progress:
        progress.progress(20, "Preparing model from Hugging Face Hub…")
    p = ensure_model(model_id)
    if progress:
        progress.progress(55, "Model is ready locally…")
    return p

@st.cache_resource(show_spinner=False)
def load_pipe(task, model_id, aggregation=True):
    from transformers import pipeline
    local = local_path(model_id)
    model_source = str(local) if is_ready(model_id) else model_id
    kw = dict(task=task, model=model_source, tokenizer=model_source, device=-1)
    if task in ("ner","token-classification") and aggregation:
        kw["aggregation_strategy"] = "simple"
    return pipeline(**kw)

@st.cache_resource(show_spinner=False)
def load_local_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    if not LOCAL_MODEL.exists():
        raise FileNotFoundError("Local Qwen model is not installed. Run scripts/download_local_llm.py")
    tok = AutoTokenizer.from_pretrained(str(LOCAL_MODEL), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_MODEL), local_files_only=True, torch_dtype="auto"
    )
    return tok, model

def local_generate(question, context, max_new_tokens=220):
    import torch
    tok, model = load_local_llm()
    messages = [
        {"role":"system","content":"You are a teaching assistant. Answer only from the supplied context. If the answer is absent, say so."},
        {"role":"user","content":f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs.input_ids.shape[-1]:]
    return tok.decode(new, skip_special_tokens=True)

def progress_run(task, model_id, text, **kwargs):
    p = st.progress(0, "Checking local model library…")
    prepare_selected_model(model_id, p)
    p.progress(65, "Loading model into memory…")
    pipe = load_pipe(task, model_id)
    p.progress(80, "Running inference…")
    result = pipe(text, **kwargs)
    p.progress(100, "Complete")
    time.sleep(.1)
    p.empty()
    return result

def rows(entities):
    return [{"Entity":e.get("word",""),"Label":e.get("entity_group",e.get("entity","")),
             "Confidence":round(float(e.get("score",0)),4),
             "Start":e.get("start",""),"End":e.get("end","")} for e in entities]

st.sidebar.title("🤗 HF AI Teaching Lab")
st.sidebar.caption("BERT • NER • Local LLM")
page = st.sidebar.radio("Modules",[
    "🏠 Dashboard","📚 Learning","🧩 NER Explorer","🧠 BERT Explorer",
    "🤖 Local LLM","🔬 Compare Models","⚙️ Model Library"
])
st.sidebar.divider()
st.sidebar.caption("All catalogue models can be preloaded into local_models/.")

st.title("Hugging Face BERT, NER & Local LLM Teaching Lab")
st.caption("A classroom application for understanding how pretrained and fine-tuned Transformer models are selected, downloaded and used.")

if page == "🏠 Dashboard":
    st.subheader("Laboratory overview")
    for c,(icon,title,desc) in zip(st.columns(4),[
        ("🧩","NER","Identify people, places, organisations and other entities."),
        ("🧠","BERT","Explore base Transformer checkpoints and task heads."),
        ("🤖","Local LLM","Ask questions using a model stored inside this project."),
        ("🔬","Compare","Run the same input through multiple models.")
    ]):
        with c: st.markdown(f'<div class="panel"><h3>{icon} {title}</h3><div class="small">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown("### Ready-made demonstration")
    st.info(DEMO_NER)
    if st.button("▶ Run BERT NER demonstration",type="primary",use_container_width=True):
        try:
            result=progress_run("ner",NER_MODELS["BERT NER — dslim"],DEMO_NER)
            st.dataframe(rows(result),use_container_width=True,hide_index=True)
        except Exception as e: st.error(str(e))

    st.markdown("### Learning path")
    st.markdown("""
    **1. Transformer → 2. BERT encoder → 3. Tokenization → 4. Fine-tuning →
    5. NER → 6. Local LLM → 7. RAG → 8. Model comparison**

    The key teaching idea is that a **base BERT model is not automatically an NER model**.
    A task-specific checkpoint adds a task head and training for the desired task.
    """)

elif page == "📚 Learning":
    st.subheader("📚 Detailed Learning Module")
    topic=st.selectbox("Choose a lesson",[
        "1. What is a Transformer?",
        "2. What is BERT?",
        "3. Encoder vs Decoder",
        "4. Tokenization",
        "5. Self-Attention",
        "6. Pretraining and Fine-Tuning",
        "7. What is NER?",
        "8. How BERT performs NER",
        "9. Base Model vs Task Model",
        "10. Hugging Face Hub and Model Loading",
        "11. Confidence Scores",
        "12. Local LLMs",
        "13. RAG with a Local LLM",
        "14. BERT QA vs RAG",
        "15. Model Selection and Trade-offs",
    ])
    lessons={
"1. What is a Transformer?":(
"Transformers are neural-network architectures built around attention mechanisms.",
"""A Transformer processes a sequence by representing tokens as vectors and allowing tokens
to interact through attention. Unlike older recurrent architectures, the main computation
can be highly parallelised.

Pipeline:
Text → Tokenizer → Token IDs → Embeddings → Attention layers → Task head → Output

Teaching example:
“The officer visited Goa.”
Attention helps the model represent relationships between words such as “officer”, “visited” and “Goa”."""
),
"2. What is BERT?":(
"BERT means Bidirectional Encoder Representations from Transformers.",
"""BERT is an encoder-only Transformer designed primarily for language understanding.
It reads context from both directions.

Important characteristics:
• Encoder-only architecture
• Bidirectional contextual representations
• Pretrained on large text corpora
• Adaptable through fine-tuning
• Useful for classification, NER, similarity, QA and embeddings

BERT is not inherently a chatbot. A base BERT checkpoint produces representations; a
task-specific model adds an appropriate prediction head."""
),
"3. Encoder vs Decoder":(
"Encoder and decoder Transformers solve different classes of problems.",
"""Encoder models such as BERT are strong at understanding and representing text.
Decoder/causal models such as Qwen are designed to generate text one token at a time.

Encoder:
Text → contextual representation → classification / NER / QA

Decoder:
Prompt → next-token prediction → generated text

This distinction explains why BERT NER and a local Qwen chatbot are separate modules."""
),
"4. Tokenization":(
"Tokenization converts human text into pieces that a Transformer can process.",
"""Example:
“National Forensic Sciences University”

may be represented as whole words or subwords depending on the tokenizer.

The tokenizer produces:
• tokens
• token IDs
• attention masks
• sometimes special tokens

NER is performed at the token level and then token predictions are aggregated into human-readable entities."""
),
"5. Self-Attention":(
"Self-attention lets each token assign different importance to other tokens.",
"""Conceptually:
Query = what this token is looking for
Key = what another token represents
Value = information contributed by that token

Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V

Multiple attention heads allow the model to learn different relationships simultaneously.

Classroom experiment:
Change a sentence and observe how entity recognition or classification changes."""
),
"6. Pretraining and Fine-Tuning":(
"Pretraining learns general language patterns; fine-tuning adapts the model to a specific task.",
"""Example:
BERT base → general language representation

BERT + NER training → token labels such as:
B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC

BERT + sentiment training → POSITIVE / NEGATIVE

Therefore, two models can share the BERT architecture but behave very differently because
their task-specific weights are different."""
),
"7. What is NER?":(
"Named Entity Recognition identifies spans of text and assigns semantic categories.",
"""Typical categories include:
PERSON — Dr. Ranjit Kolkar
ORGANIZATION — National Forensic Sciences University
LOCATION — Goa
DATE — 12 September 2026

NER is usually formulated as token classification.

Applications:
• Digital forensics
• Legal document analysis
• Intelligence/OSINT
• News analysis
• Information extraction
• Case-file indexing"""),
"8. How BERT performs NER":(
"BERT NER predicts a label for each token after contextual encoding.",
"""Example:
“Ranjit visited Goa”

Tokenizer → [Ranjit] [visited] [Goa]
             ↓        ↓        ↓
BERT contextual representations
             ↓        ↓        ↓
NER classifier
             ↓        ↓        ↓
           PERSON   O      LOCATION

A post-processing step combines subword predictions into entity spans."""
),
"9. Base Model vs Task Model":(
"A base checkpoint and a fine-tuned checkpoint should not be treated as interchangeable.",
"""Base:
google-bert/bert-base-uncased

Task-specific:
dslim/bert-base-NER

The first is a general BERT encoder. The second has been adapted for NER.
Selecting a model therefore requires checking:
1. architecture
2. task
3. tokenizer
4. labels
5. language
6. model size
7. license
8. intended use"""),
"10. Hugging Face Hub and Model Loading":(
"The Hugging Face Hub is a repository of pretrained and fine-tuned models.",
"""The application accepts a model ID such as:
dslim/bert-base-NER

Transformers can download model files from the Hub and cache them locally.
A local directory can also be supplied to from_pretrained(), allowing offline inference.

Teaching flow:
Hub model ID → download → local cache/folder → tokenizer + weights → pipeline → inference"""),
"11. Confidence Scores":(
"A pipeline score is a model prediction score, not a guarantee that the prediction is factually correct.",
"""For NER:
PERSON — 0.98

This indicates strong model preference for the label/span under that model's scoring
procedure. It does NOT prove that the person actually exists or that the information is true.

For forensic applications, model output should therefore be treated as an analytical aid,
not independent proof."""),
"12. Local LLMs":(
"A local LLM runs on the user's machine or controlled infrastructure.",
"""This project includes a setup for Qwen2.5-0.5B-Instruct.

Benefits:
• Can run without sending document content to a cloud API
• Useful for classroom demonstrations
• Model files can be kept with the project
• Reproducible environment

Limitations:
• Small models are less capable than larger models
• CPU inference can be slow
• RAM/storage requirements increase with model size"""),
"13. RAG with a Local LLM":(
"RAG combines retrieval with generation.",
"""Document → chunks → embeddings/search → relevant passages → local LLM → answer

The LLM is not expected to memorize the uploaded document.
Instead, the application supplies retrieved evidence in the prompt.

This is especially useful for long documents because the complete document does not need
to be placed in every prompt."""),
"14. BERT QA vs RAG":(
"BERT QA and RAG solve document-question tasks differently.",
"""BERT extractive QA:
Question + context → predict start/end positions → answer span

RAG:
Question → retrieve relevant chunks → LLM generates answer

BERT QA is excellent for teaching extractive question answering.
RAG is better suited to conversational document assistants, especially when answers need
synthesis across multiple passages."""),
"15. Model Selection and Trade-offs":(
"Choosing a model is an engineering decision, not simply a search for the largest model.",
"""Consider:
• Task: NER, classification, generation, QA
• Language
• Model size
• CPU/GPU availability
• Latency
• Accuracy on the relevant domain
• License
• Privacy requirements
• Hallucination risk
• Explainability and evidence requirements

For teaching, start small and make the model's behaviour visible."""
)}
    a,b=lessons[topic]
    st.markdown(f"### {topic}")
    st.info(a)
    st.write(b)

elif page == "🧩 NER Explorer":
    st.subheader("🧩 NER Explorer — any compatible Hugging Face model")
    mode=st.radio("Select model",["Recommended","Enter model ID"],horizontal=True)
    if mode=="Recommended":
        label=st.selectbox("Model",list(NER_MODELS))
        mid=NER_MODELS[label]
    else:
        mid=st.text_input("Hugging Face model ID","dslim/bert-base-NER")
    txt=st.text_area("Input",DEMO_NER,height=170)
    if st.button("🚀 Download / Load & Run NER",type="primary",use_container_width=True):
        try:
            result=progress_run("ner",mid.strip(),txt)
            st.markdown("### Entities")
            st.dataframe(rows(result),use_container_width=True,hide_index=True)
            st.markdown("### Visual output")
            st.markdown(" ".join([f'<span class="entity"><b>{e["word"]}</b> [{e.get("entity_group",e.get("entity"))}] {e["score"]:.2f}</span>' for e in result]),unsafe_allow_html=True)
        except Exception as e:
            st.error("The selected model is not compatible with this NER pipeline or could not be downloaded.")
            st.code(str(e))

elif page == "🧠 BERT Explorer":
    st.subheader("🧠 BERT Model Explorer")
    st.info("Base BERT is an encoder. For a specific task, select a compatible fine-tuned checkpoint.")
    task=st.selectbox("Task",["Masked Language Modeling","Sentiment Classification","Feature Extraction"])
    if task=="Masked Language Modeling":
        mid=st.text_input("Model","google-bert/bert-base-uncased")
        txt=st.text_input("Input","National Forensic Sciences University is located in [MASK].")
        t="fill-mask"
    elif task=="Sentiment Classification":
        mid=st.text_input("Model","distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        txt=st.text_input("Input","The forensic analysis was accurate and well documented.")
        t="text-classification"
    else:
        mid=st.text_input("Model","google-bert/bert-base-uncased")
        txt=st.text_input("Input","Digital forensic evidence must be handled carefully.")
        t="feature-extraction"
    if st.button("🚀 Run BERT demonstration",type="primary",use_container_width=True):
        try:
            r=progress_run(t,mid,txt,top_k=5 if t=="fill-mask" else None)
            st.json(r if t!="feature-extraction" else {"tokens":len(r[0]),"embedding_dimension":len(r[0][0])})
        except Exception as e: st.error(str(e))

elif page == "🤖 Local LLM":
    st.subheader("🤖 Local LLM — project-folder model")
    if not LOCAL_MODEL.exists() or not any(LOCAL_MODEL.iterdir()):
        st.info("The local model is not yet stored in this project. The app will prepare it automatically when you run the Local LLM for the first time.")
    else:
        st.success(f"Local model found in project: `{LOCAL_MODEL}`")
    context=st.text_area("Context",DEMO_CONTEXT,height=180)
    q=st.text_input("Question",DEMO_Q)
    if st.button("▶ Ask local LLM",type="primary",use_container_width=True):
        try:
            with st.spinner("Loading local model from project folder…"):
                answer=local_generate(q,context)
            st.markdown("### Answer")
            st.success(answer)
        except Exception as e:
            st.error("Local LLM could not be loaded.")
            st.code(str(e))

elif page == "🔬 Compare Models":
    st.subheader("🔬 Compare BERT NER models")
    txt=st.text_area("Same test input for every model",DEMO_NER,height=160)
    selected=st.multiselect("Models",list(NER_MODELS),default=["BERT NER — dslim","DistilBERT NER"])
    if st.button("▶ Compare",type="primary",use_container_width=True):
        cols=st.columns(max(1,len(selected)))
        bar=st.progress(0,"Starting comparison…")
        for i,label in enumerate(selected):
            with cols[i]:
                st.markdown(f"### {label}")
                try:
                    r=progress_run("ner",NER_MODELS[label],txt)
                    st.dataframe(rows(r),use_container_width=True,hide_index=True)
                except Exception as e: st.error(str(e)[:400])
            bar.progress(int((i+1)/len(selected)*100),f"Completed {i+1}/{len(selected)}")
        bar.empty()

elif page == "⚙️ Model Library":
    st.subheader("⚙️ Local model library")
    st.markdown("""
    This project is designed so **all catalogue models are preloaded** into
    `local_models/`. Once ready, modules can run from local files without
    re-downloading each time.

    ### One-command preload

    ```bash
    python scripts/preload_all_models.py
    ```

    This downloads every model listed in `models.json`:
    - NER models
    - BERT / DistilBERT / RoBERTa / ALBERT encoders
    - Local LLM (Qwen2.5-0.5B-Instruct)

    ### Model lifecycle

    Hugging Face Hub → project `local_models/` → local tokenizer/model → inference
    """)

    report = status_report()
    ready_count = sum(1 for row in report if row["ready"])
    st.metric("Models ready locally", f"{ready_count}/{len(report)}")
    st.dataframe(
        [
            {
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

    if st.button("⬇ Preload ALL catalogue models", type="primary", use_container_width=True):
        progress = st.progress(0, "Starting full model preload…")
        status_box = st.empty()
        models = all_models()
        total = max(len(models), 1)
        done = {"n": 0}

        def cb(message: str):
            status_box.write(message)
            # advance roughly when a model finishes or is already ready
            if message.startswith("Ready:") or message.startswith("Already ready:") or message.startswith("FAILED:"):
                done["n"] += 1
                progress.progress(min(done["n"] / total, 1.0), message)

        try:
            results = ensure_all_models(progress_callback=cb)
            progress.progress(1.0, "Preload finished")
            ok = sum(1 for r in results if r["ok"])
            failed = [r for r in results if not r["ok"]]
            if failed:
                st.warning(f"Completed with issues: {ok}/{len(results)} succeeded.")
                for item in failed:
                    st.error(f"{item['id']}: {item['error']}")
            else:
                st.success(f"All {ok} catalogue models are ready in local_models/.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
