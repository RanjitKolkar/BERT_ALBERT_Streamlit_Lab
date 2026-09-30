# Hugging Face BERT, NER & Local LLM Teaching Lab v2

## What changed

This version expands the learning material substantially and adds a **project-folder local LLM**.

### Learning module

15 detailed lessons cover:

1. Transformers
2. BERT
3. Encoder vs Decoder
4. Tokenization
5. Self-Attention
6. Pretraining and Fine-Tuning
7. NER
8. BERT NER internals
9. Base vs task-specific models
10. Hugging Face Hub/model loading
11. Confidence scores
12. Local LLMs
13. RAG + Local LLM
14. BERT QA vs RAG
15. Model selection/trade-offs

## Local LLM

Model:

`Qwen/Qwen2.5-0.5B-Instruct`

The model is downloaded into:

`local_models/Qwen2.5-0.5B-Instruct`

Run:

```bash
python scripts/download_local_llm.py
```

Then:

```bash
streamlit run app.py
```

The Streamlit application loads the model from the project folder with local-only loading.

## Important note about the ZIP

The model weights are not embedded in this ZIP because model weights are large and this build environment cannot directly download the Hugging Face snapshot. The supplied downloader performs the actual download into the project folder on the user's machine/server.

This is also better for GitHub: **do not commit large model weights to GitHub**. Keep the model folder out of version control and run the downloader during setup/deployment.

## Hugging Face model loading

Transformers supports downloading pretrained models from the Hub and loading from a local directory. Local/offline loading can use `local_files_only=True` after the files are available locally.

## Demonstration input

Dr. Ranjit Kolkar visited National Forensic Sciences University in Goa on 12 September 2026 for a Digital Forensics workshop. The team later travelled to New Delhi to meet officials from the Ministry of Home Affairs.

## Classroom workflow

Use the same text with:

BERT → tokenizer → contextual representation → NER classifier → entities

Then:

Question → retrieved context → Qwen local LLM → generated answer

This lets students compare **classification/extraction models** with **generative models**.
