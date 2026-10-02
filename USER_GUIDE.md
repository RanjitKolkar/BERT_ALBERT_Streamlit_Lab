# User guide — models, how to use, context chatbot

This guide lists **every model**, how to use the app, how to **check installed models**, and the **best way** to solve context-based document Q&A.

---

## 1. Installed models status (this PC)

Checked with `model_manager.status_report()`:

| # | Group | Display name | Hugging Face ID | Local status |
|---|---|---|---|---|
| 1 | NER | BERT NER | `dslim/bert-base-NER` | **READY** |
| 2 | NER | DistilBERT NER | `elastic/distilbert-base-uncased-finetuned-conll03-english` | **READY** |
| 3 | NER | RoBERTa NER | `Jean-Baptiste/roberta-large-ner-english` | **READY** |
| 4 | NER | XLM-R NER | `Davlan/xlm-roberta-base-ner-hrl` | **READY** |
| 5 | Encoder | BERT base uncased | `google-bert/bert-base-uncased` | **READY** |
| 6 | Encoder | BERT base cased | `google-bert/bert-base-cased` | **READY** |
| 7 | Encoder | DistilBERT | `distilbert/distilbert-base-uncased` | **READY** |
| 8 | Encoder | RoBERTa | `FacebookAI/roberta-base` | **READY** |
| 9 | Encoder | ALBERT | `albert/albert-base-v2` | **READY** |
| 10 | Local LLM | Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | **READY** |

**Summary on this machine: 10 / 10 READY** under `local_models/`.

### Re-check anytime

In the app:

1. Open **Tutorial Lab**
2. Open **Model Library**
3. Click **Refresh status**

Or in a terminal from the project folder:

```bash
python -c "from model_manager import status_report; [print(('READY' if r['ready'] else 'MISSING'), r['name']) for r in status_report()]"
```

Or:

```bash
python scripts/preload_all_models.py
```

(Already-ready models are skipped; missing ones download.)

---

## 2. What each model is for

### A. Document Context Chatbot (main product use)

| Piece | Role |
|---|---|
| **TF-IDF retrieval** (scikit-learn) | Finds the most relevant passages from *your* uploaded files |
| **Extractive mode** | Returns the best matching passages (no LLM needed) |
| **Qwen2.5-0.5B-Instruct** | Rewrites an answer in English **only from retrieved passages** |

Chatbot does **not** need BERT/NER/ALBERT. Those are for the Tutorial Lab.

### B. Tutorial Lab — NER (find people, places, orgs, dates)

| Model | Best for |
|---|---|
| **BERT NER** | Standard English named-entity demo |
| **DistilBERT NER** | Faster / smaller English NER |
| **RoBERTa NER** | Stronger English NER |
| **XLM-R NER** | Multilingual NER (other languages) |

### C. Tutorial Lab — BERT / ALBERT family (fill-mask / encoder demos)

| Model | Best for |
|---|---|
| **BERT uncased** | Classic masked word prediction (`[MASK]`) |
| **BERT cased** | Same, keeps capital letters |
| **DistilBERT** | Smaller/faster BERT-like demo |
| **RoBERTa** | Uses `<mask>` instead of `[MASK]` (app auto-retries) |
| **ALBERT** | Compact encoder / parameter-sharing demo |

### D. Local generative LLM

| Model | Best for |
|---|---|
| **Qwen2.5-0.5B-Instruct** | Local chat-style answers; document Q&A after retrieval |

---

## 3. How to use (step by step)

### Start the app

```text
Double-click run_app.bat
  or
streamlit run app.py
```

Install first if needed: see [INSTALL.md](INSTALL.md) (`setup.bat`).

### Sidebar — choose application

| Application | Use when |
|---|---|
| **Document Context Chatbot** | Ask questions about uploaded PDF/Word/Excel/CSV/TXT |
| **Tutorial Lab (BERT / NER / ALBERT / Qwen)** | Teach / experiment with catalogue models |

---

### 3.1 Document Context Chatbot — how to use

1. Sidebar → **Document Context Chatbot**
2. Choose **Answer mode**:
   - **Extractive** — safest “only from documents”, works everywhere
   - **Local Qwen LLM** — fluent answer from retrieved context (needs Qwen READY)
3. **Upload** one or more files (PDF, DOCX, XLSX, CSV, TXT, MD)
4. Click **Process documents**
5. Wait until the app shows chunk/document counts
6. Type an **English question** in the chat
7. Read the answer + **sources** (which file/page/sheet matched)

**Tips for better answers**

- Ask about content that is **written in the file** (names, dates, amounts, definitions)
- Prefer short, clear questions: *“What is the penalty mentioned in section 3?”*
- Upload the right files only (extra unrelated files can dilute retrieval)
- If Local LLM sounds vague, switch to **Extractive** to see the raw passages

**Clear data**

- **Clear chat** — removes messages only  
- **Clear documents** — removes knowledge base + chat  

---

### 3.2 Tutorial Lab — paper workflow (how to use)

Sidebar → **Tutorial Lab**. Follow the numbered steps (same shared text for all models).

**Default input = plain text** (a one-page generic digital forensics summary, no personal names).  
You can **paste** any content, or **upload** PDF / Word / Excel / CSV / TXT to fill the shared text box.

| Step | Module | What you do |
|---|---|---|
| **1** | Shared document text | Paste text (default) or upload a file → Load into shared text |
| **2** | Model library | Confirm READY; preload if needed |
| **3** | Encoder demos | Fill-mask on a sentence from the document (`[MASK]`) |
| **4** | NER explorer | Entities on shared / pasted text |
| **5** | Compare NER | Same passage, multiple NER models |
| **6** | Local LLM Q&A | Ask questions; context = shared document by default |

#### Step 1 tips
- Prefer **Paste / edit text** for classroom demos  
- **Upload file → fill text box** extracts text from PDF/Word/etc. into the same editor  
- **Reset to sample forensic page** restores the built-in one-page case summary  

#### Steps 3–6
- Each step can **preview** the shared document  
- NER/Compare: buttons to load full shared text or a short forensic snippet  
- Local LLM: checkbox **Use full shared document as context** (on by default)  

#### Encoder note
- Example mask sentence: `A bit-for-bit forensic image was created using a write [MASK].`  
- RoBERTa-style models may auto-retry with `<mask>`  

---

## 4. Best way to solve a **context-based chatbot** problem

### The problem

A chatbot must answer **from your documents only**, not from general internet knowledge, and must not invent facts.

### Best practical pattern: **RAG (Retrieval-Augmented Generation)**

This project already follows that pattern:

```text
Upload documents
        ↓
Split into chunks (passages)
        ↓
Retrieve top matching chunks for the question   ← TF-IDF here
        ↓
Answer ONLY using those chunks
   ├─ Extractive: show / quote the chunks
   └─ Generative: Local Qwen writes an answer from the chunks
```

### What works best in *this* app (no API key)

| Goal | Best choice | Why |
|---|---|---|
| **Most reliable “from my PDF only”** | **Extractive mode** | Returns real passages; hard to hallucinate |
| **More natural English answer** | **Local Qwen + good retrieval** | Fluent wording, still grounded in top chunks |
| **Cloud / low RAM** | **Extractive only** | No large LLM load |
| **Classroom offline PC** | Preload models + Local Qwen | Everything in `local_models/` |

### Recommended workflow for users

1. **Process documents** carefully (correct files, wait for success)  
2. Start with **Extractive** to verify the right passages are found  
3. If passages look correct, switch to **Local Qwen LLM** for a polished answer  
4. Always read the **sources** panel — if sources are wrong, the answer will be wrong  

### Why this is better than “put the whole PDF into the LLM”

| Approach | Result |
|---|---|
| Whole document into small local LLM | Often truncated, slow, mixed-up answers |
| **Retrieve few best chunks → then answer** | Focused, faster, more faithful to the file |
| LLM with no retrieval | Hallucinations / off-document answers |

### Quality rules (best practice)

1. **Grounding** — answer only from retrieved context (app prompt enforces this for Qwen)  
2. **Abstain** — if nothing matches, say *could not find that in the uploaded document(s)*  
3. **Citations** — show source file / page / sheet (app does this)  
4. **Chunking** — keep passages readable (not tiny, not huge)  
5. **Retrieval first** — fix bad answers by improving documents/questions before blaming the LLM  

### If answers are weak — what to try

| Symptom | Fix |
|---|---|
| Wrong topic / unrelated text | Re-upload only relevant files; ask a more specific question |
| “Could not find…” but text exists | Use words closer to the document wording; try Extractive |
| LLM invents details | Stay on **Extractive**, or trust sources over free text |
| Slow / heavy | Use Extractive; Local Qwen is optional |
| Cloud deploy | Extractive only (local_models usually not on Cloud) |

### Future upgrades (if you later want stronger RAG)

Not required for current lab use; listed for clarity:

1. Better embeddings (e.g. sentence-transformers) instead of only TF-IDF  
2. Hybrid search (keywords + vectors)  
3. Larger instruct model than 0.5B (needs more RAM/GPU)  
4. Re-ranker on top retrieved chunks  

For **this lab, no API key, minimal install**, the best solution remains:

> **TF-IDF retrieval + Extractive (default truth) + optional Local Qwen on top chunks**

---

## 5. Quick map: “I want to…” 

| I want to… | Do this |
|---|---|
| Install with least work | `setup.bat` then `run_app.bat` — [INSTALL.md](INSTALL.md) |
| Ask questions on my PDF | Document Context Chatbot → upload → Process → ask |
| Stay strictly on document text | Answer mode = **Extractive** |
| Get a written answer from context | Answer mode = **Local Qwen LLM** (Qwen READY) |
| Teach NER | Tutorial Lab → NER Explorer |
| Teach BERT/ALBERT masks | Tutorial Lab → BERT / ALBERT Explorer |
| See all model status | Tutorial Lab → Model Library |
| Install only chatbot LLM | `setup.bat` or **Download Local LLM only** |
| Install every teaching model | `setup_all_models.bat` |

---

## 6. File reference

| File | Purpose |
|---|---|
| [INSTALL.md](INSTALL.md) | Easy install for normal users |
| [USER_GUIDE.md](USER_GUIDE.md) | This guide — models + usage + RAG advice |
| [README.md](README.md) | Project overview |
| [models.json](models.json) | Official model catalogue |
| `local_models/` | Where installed weights live |
| `app.py` | Main app (chatbot + lab switch) |
| `tutorial_lab.py` | Tutorial modules |
| `model_manager.py` | Download / READY checks |

---

## 7. One-page cheat sheet

```text
INSTALL:   setup.bat  →  run_app.bat
MODELS:    4 NER + 5 encoders + 1 Qwen  (check Model Library)
CHATBOT:   Upload → Process → Ask (Extractive = safest, Qwen = fluent)
BEST RAG:  Retrieve top chunks first → answer only from those chunks
CHECK:     Tutorial Lab → Model Library → Refresh status
```
