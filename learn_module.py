"""Learn stage — ordered theory path (single page, no drawer)."""

from __future__ import annotations

import streamlit as st

from ui_flow import callout, lesson_stepper, render_nav_buttons

LEARN_LESSONS = [
    {
        "id": "L0",
        "title": "How this lab is organised",
        "mins": "3",
        "body": """
### The one-page journey

| Step | Purpose |
|---|---|
| **1 Start** | Map of the lab |
| **2 Install kit** | ZIP + setup guide on **your machine** |
| **3 Learn** | Theory (this stage) |
| **4 Practice** | Hands-on on one shared document |
| **5 Chat** | Your own files Q&A |

### Suggested order
1. Finish Learn lessons L0 → L8  
2. Open **Practice** and walk its 6 mini-steps  
3. Use **Chat** with Extractive mode  
4. Only install large models if you need them  

### What you will *not* get
- No paid API key for core use  
- Not AGI — narrow specialised models  
- Local Qwen is a **small** teaching LLM  
""",
    },
    {
        "id": "L1",
        "title": "AI, ML, DL — plain language",
        "mins": "5",
        "body": """
### Artificial Intelligence (AI)
Systems that perform tasks that look “smart” (search, classify, generate text).

### Machine Learning (ML)
AI that **learns patterns from data** instead of only hand-written rules.

### Deep Learning (DL)
ML with **multi-layer neural networks**. Modern NLP (BERT, GPT-style) is DL.

```text
AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  Transformers / LLMs
```

### Remember
Bigger name ≠ better for your task. A small NER model can beat a general chatbot at finding person names in a report.
""",
    },
    {
        "id": "L2",
        "title": "NLP tasks in this lab",
        "mins": "6",
        "body": """
| Task | What it does | Where in the lab |
|---|---|---|
| **NER** | Find people, orgs, places, dates | Practice → NER |
| **Fill-mask** | Guess a hidden word | Practice → Encoder |
| **Generation** | Write new text | Practice / Chat (Qwen) |
| **Retrieval** | Find relevant passages | Chat (TF-IDF) |
| **RAG** | Retrieve then answer | Chat = retrieve + extractive or Qwen |

### Document angle
You usually want **grounded** answers: only what the file says. Chat retrieves first.
""",
    },
    {
        "id": "L3",
        "title": "Tokens, embeddings, attention",
        "mins": "7",
        "body": """
### Tokens
Text is split into pieces (words or subwords). Models read numbers, not letters.

### Embeddings
Each token becomes a vector that captures meaning in context.

### Attention
The model learns **which tokens matter to each other**.

### BERT vs GPT-style
- **Encoders (BERT family):** left *and* right context → NER, fill-mask  
- **Decoders (Qwen style):** next-token prediction → chat / generation  

This lab includes **both**.
""",
    },
    {
        "id": "L4",
        "title": "Transformers architecture",
        "mins": "8",
        "body": """
### Core idea
A stack of layers that mix **self-attention** + small feed-forward networks.

```text
Input tokens → embeddings → N × (Attention + FFN) → task head
```

### Hugging Face `transformers` here
- tokenizer (text → ids)  
- model weights (from `local_models/` when installed)  
- `pipeline()` helpers for NER / fill-mask / generation  

### Fine-tuned
A **base** model learns general language. A **fine-tuned** model (e.g. BERT-NER) was trained further for one task.

### Shortcoming
Powerful but **heavy** on disk/RAM. Always start small (Install kit → Beginner).
""",
    },
    {
        "id": "L5",
        "title": "Encoders (BERT → ALBERT)",
        "mins": "6",
        "body": """
| Model | Beginner? | Good for | Shortcoming |
|---|---|---|---|
| **DistilBERT** | **Start here** | Fast demos | Slightly weaker than BERT |
| **BERT uncased** | Yes | Classic teaching | Larger than DistilBERT |
| **BERT cased** | Yes | Capitals matter | Same size as BERT |
| **RoBERTa** | Medium | Strong baseline | Uses `<mask>` |
| **ALBERT** | Medium | Parameter sharing | Can feel different |

Next: **Practice → Encoder demos**.
""",
    },
    {
        "id": "L6",
        "title": "NER models — pick wisely",
        "mins": "6",
        "body": """
| Model | Beginner? | Strength | Shortcoming |
|---|---|---|---|
| **DistilBERT NER** | **Best first** | Small, fast | Compact English focus |
| **BERT NER** | Great 2nd | Standard classroom NER | Medium size |
| **XLM-R NER** | Later | Multilingual | Heavier |
| **RoBERTa-large NER** | Last | Strong English | **Large RAM** |

Tip: compare **2 small models** on the same sentence before adding large ones.
""",
    },
    {
        "id": "L7",
        "title": "Chat, RAG, and local LLM",
        "mins": "8",
        "body": """
### Best pattern = RAG
```text
Files → chunks → retrieve top passages → answer ONLY from them
   ├─ Extractive: show passages (safest)
   └─ Local Qwen: fluent rewrite from those passages
```

### Qwen2.5-0.5B-Instruct
- Small **local** generative model  
- Good for **teaching** grounded Q&A  
- **Not** a cloud-grade assistant  

| Approach | When | Shortcoming |
|---|---|---|
| Extractive | Always start | Less “chatty” |
| Local Qwen | Fluent summary | More RAM; can still err |
| Whole PDF into LLM | Almost never | Truncation + hallucinations |

Chat works **without** Qwen.
""",
    },
    {
        "id": "L8",
        "title": "LLMs, agents, AGI — honest map",
        "mins": "7",
        "body": """
### LLMs
Generative models trained on broad text. This lab’s Qwen is small (&lt;1B class).

### Agents
Systems that plan + use tools in a loop. This lab focuses on **models + RAG**, not full agents.

### AGI
Hypothetical broad human-like competence. **Nothing here is AGI.**

### Responsible checklist
- Prefer document-grounded answers  
- Show sources  
- Know model limits  
- Do not invent facts absent from evidence  

### Next
Open **Practice** (shared text → encoder → NER → LLM), then **Chat**.
""",
    },
]


def render_learn() -> None:
    st.markdown("## Step 3 · Learn — concepts in order")
    st.caption("Read one lesson at a time. No model download required for theory.")

    callout(
        "Use <b>Previous / Next</b> below. Finish L0→L8, then continue to <b>Practice</b>."
    )

    titles = [f"{x['id']} · {x['title']} (~{x['mins']} min)" for x in LEARN_LESSONS]
    idx = lesson_stepper(titles, "learn_idx", total_label="Lesson")
    lesson = LEARN_LESSONS[idx]

    st.markdown(f"### {lesson['id']} · {lesson['title']}")
    st.markdown(lesson["body"])

    # Lesson jump chips
    with st.expander("Jump to any lesson", expanded=False):
        cols = st.columns(3)
        for i, title in enumerate(titles):
            with cols[i % 3]:
                if st.button(title, key=f"learn_jump_{i}", use_container_width=True):
                    st.session_state.learn_idx = i
                    st.rerun()

    if idx >= len(LEARN_LESSONS) - 1:
        callout(
            "Theory complete. Continue to <b>Practice</b> for the hands-on paper workflow.",
            kind="ok",
        )

    st.markdown("---")
    render_nav_buttons(
        prev_key="install",
        next_key="practice",
        next_label="Continue to Practice →",
        mark_done_key="learn",
    )
