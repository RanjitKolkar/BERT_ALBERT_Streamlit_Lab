# Hugging Face BERT, NER & Local LLM Teaching Lab v3

## Easy install (normal users — minimise work)

**Windows (recommended): two double-clicks**

| Order | File | What it does |
|---|---|---|
| 1 | `setup.bat` | Installs packages + **one** Local LLM download (Qwen) |
| 2 | `run_app.bat` | Starts the app in your browser |

- Easy install: [INSTALL.md](INSTALL.md)
- **Full how-to (all models + chatbot + RAG advice):** [USER_GUIDE.md](USER_GUIDE.md)

```text
setup.bat  →  wait for SETUP COMPLETE  →  run_app.bat
```

- No API key
- No manual model search
- Default setup downloads **only the Local LLM** (not every tutorial model)
- Optional full pack later: `setup_all_models.bat`

One command (any OS):

```bash
python scripts/easy_setup.py
streamlit run app.py
```

### Models in this project (10)

| Group | Models | Used in |
|---|---|---|
| NER (4) | BERT NER, DistilBERT NER, RoBERTa NER, XLM-R NER | Tutorial Lab |
| Encoders (5) | BERT uncased/cased, DistilBERT, RoBERTa, ALBERT | Tutorial Lab |
| Local LLM (1) | Qwen2.5-0.5B-Instruct | Chatbot + Tutorial Lab |

**Check installed status:** Tutorial Lab → **Model Library**, or see [USER_GUIDE.md](USER_GUIDE.md#1-installed-models-status-this-pc).

### Context-based chatbot (best approach)

Use **RAG**: retrieve the best document chunks first, then answer **only** from those chunks.

| Mode | Best when |
|---|---|
| **Extractive** | You need answers strictly from the file (most reliable, no LLM) |
| **Local Qwen** | You want a fluent English answer grounded in retrieved chunks |

Details: [USER_GUIDE.md — Best way to solve context-based chatbot](USER_GUIDE.md#4-best-way-to-solve-a-context-based-chatbot-problem).

## Zero-command classroom experience (after setup)

The intended user workflow is simply:

```text
Open Streamlit app (run_app.bat)
        ↓
Select module
        ↓
Select model
        ↓
Enter text / use demonstration
        ↓
Run
```

Students do not need to run Python commands or download models manually after the one-time setup.

## Model catalogue

### NER models

| Display name | Hugging Face ID | Purpose |
|---|---|---|
| BERT NER | `dslim/bert-base-NER` | English NER |
| DistilBERT NER | `elastic/distilbert-base-uncased-finetuned-conll03-english` | Compact English NER |
| RoBERTa NER | `Jean-Baptiste/roberta-large-ner-english` | English NER |
| XLM-R NER | `Davlan/xlm-roberta-base-ner-hrl` | Multilingual NER |

### Encoder / BERT family

| Display name | Hugging Face ID | Demonstration |
|---|---|---|
| BERT base uncased | `google-bert/bert-base-uncased` | Masked language modeling |
| BERT base cased | `google-bert/bert-base-cased` | Masked language modeling |
| DistilBERT | `distilbert/distilbert-base-uncased` | Encoder demonstration |
| RoBERTa | `FacebookAI/roberta-base` | Encoder demonstration |
| ALBERT | `albert/albert-base-v2` | Encoder demonstration |

### Local generative model

| Display name | Hugging Face ID | Purpose |
|---|---|---|
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | Local text generation / document QA |

Qwen2.5-0.5B-Instruct is an Apache-2.0 licensed 0.49B-parameter causal language model. Its official model card documents Transformers and local-app usage. See the reference link in the application.

## Important deployment note

Model weights are large binary artifacts and should normally be stored with Git LFS, an object store, or a deployment artifact rather than ordinary Git blobs.

This repository therefore contains:
- the application
- model catalogue
- model reference information
- automatic model preparation logic
- local-model directory structure

If model files are already present in `local_models/`, the application uses them locally.

### Preload models

**Recommended for most users (one download — Local LLM only):**

```bash
python scripts/easy_setup.py
# or double-click setup.bat on Windows
```

**Full classroom bundle (all models in `models.json`):**

```bash
python scripts/easy_setup.py --all
# or: python scripts/preload_all_models.py
# or double-click setup_all_models.bat on Windows
```

This stores checkpoints under `local_models/`.

If a model is still absent at runtime, the application can prepare it automatically from the Hugging Face Hub (or use the in-app **Download Qwen** button). For a truly offline deployment, run setup first.

## Example-driven learning

The Learning module uses practical examples:

- NER: identifying a person, university, location and date in a forensic-training sentence.
- Tokenization: showing how one sentence becomes tokens.
- Attention: explaining how “Goa” relates to “visited”.
- Fine-tuning: comparing base BERT with BERT-NER.
- Local LLM: asking a question about a short forensic-report context.
- RAG: retrieving evidence before asking the local LLM to generate an answer.

## References

Hugging Face Transformers pipeline:
https://huggingface.co/docs/transformers/pipeline_tutorial

Hugging Face model loading:
https://huggingface.co/docs/transformers/main/models

Qwen2.5-0.5B-Instruct:
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
