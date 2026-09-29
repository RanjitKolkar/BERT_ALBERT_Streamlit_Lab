# Transformer AI Teaching Lab v3

Panel-style Streamlit teaching application covering Transformers, BERT QA, NER, embeddings, RAG, local LLMs and cloud LLMs.

## Main demonstration

The Comparison Lab uses one fixed forensic-document test case and one question:

**What information should a forensic report contain?**

It contrasts:
- BERT QA: extractive answer span
- Fast RAG: evidence retrieval
- RAG + Local LLM: Ollama generation from retrieved evidence
- RAG + Cloud LLM: API generation from retrieved evidence

## Performance

- Models are lazy-loaded.
- Models are cached with `st.cache_resource`.
- Fast RAG uses TF-IDF and avoids an embedding-model download.
- Semantic RAG loads MiniLM only when selected.
- Large documents are chunked.
- Spinners and progress bars show stages.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Local LLM

Install Ollama separately and pull a model, e.g. `ollama pull qwen2.5:3b`. Then use the Local LLM module.

## Cloud LLM

Enter the provider API key in the UI or use Streamlit Secrets. Never commit keys to GitHub.

## Streamlit Cloud

Deploy `app.py` and `requirements.txt`. Start with the fast modules; heavy models load only when explicitly used.
