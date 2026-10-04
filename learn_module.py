"""Learn mode — guided path from basics to Transformers, RAG, LLMs, and AGI context."""

from __future__ import annotations

import streamlit as st

LEARN_LESSONS = [
    {
        "id": "L0",
        "title": "0 · How this lab is organised",
        "mins": "3",
        "body": """
### Three doors in the sidebar

| Mode | Purpose |
|---|---|
| **1 · Chat** | Use document Q&A with whatever is already installed |
| **2 · Learn** | Understand ideas before / while you install models |
| **3 · Install yourself** | One-click downloads + honest guidance on each model |

### Suggested learning order
1. Read this Learn path (L0 → L8)  
2. Try **Chat** with Extractive mode on a small PDF  
3. Install **Beginner pack** (small models only)  
4. Try hands-on demos (NER / fill-mask / local LLM)  
5. Only then install large models if you need them  

### What you will *not* get here
- No paid API key is required for the core lab  
- This is **not** AGI — we use narrow, specialised models  
- Local Qwen is a **small** instruct model for teaching, not a production assistant
""",
    },
    {
        "id": "L1",
        "title": "1 · AI, ML, DL — plain language",
        "mins": "5",
        "body": """
### Artificial Intelligence (AI)
Any system that performs tasks that look “smart” to humans (search, classify, generate text).

### Machine Learning (ML)
AI that **learns patterns from data** instead of only hand-written rules.

### Deep Learning (DL)
ML with **multi-layer neural networks**. Modern NLP (BERT, GPT-style models) is DL.

```text
AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  Transformers / LLMs
```

### Shortcoming to remember
Bigger name ≠ better for your task. A small NER model can beat a general chatbot at finding person names in a police report.
""",
    },
    {
        "id": "L2",
        "title": "2 · NLP tasks you will meet",
        "mins": "6",
        "body": """
| Task | What it does | Lab model family |
|---|---|---|
| **Classification** | Assign a label to text | (concept) |
| **NER** | Find people, orgs, places, dates | BERT / DistilBERT / RoBERTa / XLM-R NER |
| **Fill-mask** | Guess a hidden word | BERT, ALBERT, RoBERTa encoders |
| **Generation** | Write new text | Local Qwen (small LLM) |
| **Retrieval** | Find relevant passages | TF-IDF in Chat (no neural net) |
| **RAG** | Retrieve then generate/answer | Chat = retrieve + extractive or Qwen |

### Forensic / document angle
You usually want **grounded** answers: only what the file says. That is why Chat uses retrieval first.
""",
    },
    {
        "id": "L3",
        "title": "3 · Tokens, embeddings, attention (intuition)",
        "mins": "7",
        "body": """
### Tokens
Text is split into pieces (words or subwords) called **tokens**. Models read numbers, not letters.

### Embeddings
Each token becomes a vector (list of numbers) that captures meaning in context.

### Attention
The model learns **which tokens matter to each other**  
Example: in “the image hash was verified”, “hash” attends strongly to “image” and “verified”.

### Why BERT vs GPT-style differs
- **Encoders (BERT family):** see left *and* right context → great for NER, classification, fill-mask  
- **Decoders (GPT / Qwen style):** predict next token → great for chat and generation  

This lab includes **both**.
""",
    },
    {
        "id": "L4",
        "title": "4 · Transformers architecture (core idea)",
        "mins": "8",
        "body": """
### The Transformer (2017 idea, still the backbone)
A stack of layers that mix **self-attention** + small feed-forward networks.

```text
Input tokens → embeddings → N × (Attention + FFN) → task head
```

### Hugging Face Transformers library
In this project, the Python package `transformers` loads:

- tokenizer (text → ids)  
- model weights (from `local_models/` when installed)  
- optional `pipeline()` helpers for NER / fill-mask / generation  

### What “fine-tuned” means
A **base** model learns general language.  
A **fine-tuned** model (e.g. BERT-NER) was further trained for one task (entities).

### Shortcoming
Transformers are powerful but **heavy** on disk/RAM. Always start small (see Install guide).
""",
    },
    {
        "id": "L5",
        "title": "5 · Encoders in this lab (BERT → ALBERT)",
        "mins": "6",
        "body": """
| Model | Good for | Beginner? | Shortcoming |
|---|---|---|---|
| **DistilBERT** | Fast demos, less RAM | **Yes — start here** | Slightly weaker than full BERT |
| **BERT uncased** | Classic teaching reference | Yes | Larger than DistilBERT |
| **BERT cased** | When capitals matter | Medium | Same size as BERT |
| **RoBERTa base** | Strong encoder baseline | Medium | Different mask token (`<mask>`) |
| **ALBERT** | Parameter-efficient encoder | Medium | Can feel different / slower to grasp |

### Hands-on later
Learn tab → **Try encoder fill-mask** (needs model installed).
""",
    },
    {
        "id": "L6",
        "title": "6 · NER models — pick wisely",
        "mins": "6",
        "body": """
| Model | Beginner? | Strength | Shortcoming |
|---|---|---|---|
| **DistilBERT NER** | **Best first NER** | Small, fast | English-focused, compact |
| **BERT NER** | Great second step | Standard classroom NER | Medium size |
| **XLM-R NER** | If you need other languages | Multilingual | Heavier; not first install |
| **RoBERTa-large NER** | Advanced compare only | Often strong English NER | **Large RAM/disk** — install last |

### Teaching tip
Compare 2 small models on the **same sentence** before adding large ones.
""",
    },
    {
        "id": "L7",
        "title": "7 · Chat, RAG, and local LLM (Qwen)",
        "mins": "8",
        "body": """
### Best pattern for document chat = RAG
```text
Your files → chunks → retrieve top passages → answer ONLY from them
   ├─ Extractive: show passages (safest, low memory)
   └─ Local Qwen: rewrite a fluent answer from those passages
```

### Qwen2.5-0.5B-Instruct in this lab
- Small **local** generative model (no cloud API key)  
- Good for **teaching** grounded Q&A  
- **Not** a replacement for large commercial assistants  

| Approach | Best when | Shortcoming |
|---|---|---|
| Extractive | Always start here | Less “chatty” |
| Local Qwen | Fluent summary of retrieved text | Uses more RAM; can still err |
| Whole PDF into LLM | Almost never on small models | Truncation + hallucinations |

### Install tip
Chat works **without** Qwen. Install Qwen only when you want generative mode.
""",
    },
    {
        "id": "L8",
        "title": "8 · LLMs, agents, and AGI — honest map",
        "mins": "7",
        "body": """
### Large Language Models (LLMs)
Generative models trained on broad text. Scale ranges from &lt;1B params (this lab’s Qwen) to hundreds of billions.

### What people call “agents”
Systems that **plan + use tools** (search, code, APIs) in a loop. This lab focuses on **models + RAG**, not full agents.

### AGI (Artificial General Intelligence)
A hypothetical system with broad human-like competence across many tasks.  

**Important:** nothing in this repository is AGI.  
BERT NER, ALBERT, and Qwen-0.5B are **narrow tools**. Treating them as AGI causes bad science and bad forensics.

### Responsible use checklist
- Prefer **document-grounded** answers for casework-style teaching  
- Show **sources**  
- Know model **limits** (language, size, domain)  
- Do not invent facts not present in evidence text  

### Where to go next in the app
1. **Chat** — practice RAG on a forensic-style PDF  
2. **Install yourself** — Beginner pack first  
3. Return here, then open **Hands-on practice** below  
""",
    },
]


def render_learn() -> None:
    st.title("2 · Learn — from basics to Transformers & beyond")
    st.caption(
        "Read in order. No download required for the theory lessons. "
        "Hands-on practice needs models from **Install yourself**."
    )

    lesson_titles = [f"{x['title']}  ·  ~{x['mins']} min" for x in LEARN_LESSONS]
    choice = st.selectbox("Lesson", lesson_titles, key="learn_lesson_select")
    idx = lesson_titles.index(choice)
    lesson = LEARN_LESSONS[idx]

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Previous", disabled=idx == 0, use_container_width=True):
            st.session_state.learn_lesson_select = lesson_titles[idx - 1]
            st.rerun()
    with c2:
        st.metric("Progress", f"{idx + 1} / {len(LEARN_LESSONS)}")
    with c3:
        if st.button(
            "Next →", disabled=idx >= len(LEARN_LESSONS) - 1, use_container_width=True
        ):
            st.session_state.learn_lesson_select = lesson_titles[idx + 1]
            st.rerun()

    st.markdown(lesson["body"])

    st.markdown("---")
    st.subheader("Hands-on practice (optional)")
    st.caption("Uses installed local models. If missing, the app will guide you to Install.")

    practice = st.radio(
        "Practice area",
        [
            "Overview only",
            "Encoder fill-mask (BERT family)",
            "NER explorer",
            "Compare NER",
            "Local LLM on shared text",
            "Full paper workflow lab",
        ],
        key="learn_practice",
    )

    if practice == "Overview only":
        st.info(
            "Finish the lessons, then try Chat. When ready for models, open "
            "**Install yourself** → one-click Beginner pack."
        )
        return

    try:
        from tutorial_lab import (
            render_compare_ner,
            render_encoder_explorer,
            render_local_llm_demo,
            render_ner_explorer,
            render_tutorial_lab,
        )
    except Exception as exc:
        st.error(f"Hands-on lab could not load: {exc}")
        return

    if practice == "Encoder fill-mask (BERT family)":
        render_encoder_explorer()
    elif practice == "NER explorer":
        render_ner_explorer()
    elif practice == "Compare NER":
        render_compare_ner()
    elif practice == "Local LLM on shared text":
        render_local_llm_demo()
    else:
        render_tutorial_lab()
