# AI Document Lab (Chat · Learn · Install)

Rewritten classroom / research lab app for **document chat**, **learning Transformers**, and **one-click local model install**.
**No API key** required for core use.

## App flow (sidebar)

| Order | Mode | What you do |
|---|---|---|
| **1** | **Chat** | Start here. See available project models. Upload PDF/Word/Excel/TXT and ask English questions. |
| **2** | **Learn** | Lessons from AI basics → NLP → Transformers → NER/encoders → RAG/LLM → AGI honesty + hands-on practice. |
| **3** | **Install yourself** | Guided library: what is best to begin, shortcomings, **one-click** Chat / Beginner / Everything packs. |

```text
Chat (Extractive)  →  Learn  →  Install beginner pack  →  richer demos
```

## Quick start (Windows)

```text
setup.bat      → packages + Local LLM (optional but useful)
run_app.bat    → opens Streamlit
```

Or:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## One-click installs (inside the app)

Open **3 · Install yourself**:

| Button | Downloads |
|---|---|
| **Chat pack** | Qwen2.5-0.5B-Instruct only |
| **Beginner pack** (recommended) | DistilBERT NER + DistilBERT encoder + Qwen |
| **Download everything** | All 10 models in `models.json` |

Desktop equivalents: `setup.bat`, `setup_all_models.bat`.

## Chat engines

| Engine | Needs model? | Notes |
|---|---|---|
| **Extractive** | No | Best first step, low memory, grounded passages |
| **Local Qwen** | Yes (Install / setup.bat) | Fluent answer from retrieved chunks only |

## Model catalogue (10)

See **Install yourself** for level, best-for, shortcomings, and start advice.

- NER (4): DistilBERT NER *(begin here)*, BERT NER, XLM-R NER, RoBERTa-large NER *(last)*
- Encoders (5): DistilBERT *(begin here)*, BERT uncased/cased, RoBERTa, ALBERT
- Local LLM (1): Qwen2.5-0.5B-Instruct

## Project layout

| File | Role |
|---|---|
| `app.py` | Main shell — Chat / Learn / Install |
| `chat_module.py` | Document chatbot |
| `learn_module.py` | Lessons + hands-on |
| `install_module.py` | Guided one-click downloads |
| `tutorial_lab.py` | Shared-document paper demos |
| `model_manager.py` | Download / READY checks |
| `runtime_memory.py` | One-model-at-a-time memory helpers |
| `models.json` | Catalogue |
| `local_models/` | Installed weights |
| `INSTALL.md` / `USER_GUIDE.md` | Extra guides |

## Design principles

1. **Start with Chat** using what is already available
2. **Learn** concepts before stacking large models
3. **Install yourself** with clear begin-here vs install-last guidance
4. Prefer **RAG** (retrieve then answer) over stuffing whole files into a small LLM
5. Be honest: this lab is **not AGI** — narrow tools for teaching and document Q&A
