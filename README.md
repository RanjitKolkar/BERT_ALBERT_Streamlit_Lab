# Hugging Face BERT, NER & Local LLM Teaching Lab v3

## Zero-command classroom experience

The intended user workflow is simply:

```text
Open Streamlit app
        ↓
Select module
        ↓
Select model
        ↓
Enter text / use demonstration
        ↓
Run
```

Students do not need to run Python commands or download models manually.

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

### Preload all models

To prepare the full classroom bundle offline, download every model from `models.json`:

```bash
python scripts/preload_all_models.py
```

This stores all NER, encoder, and local-LLM checkpoints under `local_models/`.

If a model is still absent at runtime, the application can prepare it automatically from the Hugging Face Hub. For a truly offline deployment, run the preload script first.

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
