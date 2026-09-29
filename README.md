# Transformer AI Teaching Lab v2 — PDF/QA Fix

This version fixes the PDF Question Answering error caused by variable-length
overflow tokenizer features being converted directly to PyTorch tensors.

## Main fix

QA tokenization now uses:

```python
padding="max_length"
return_tensors="pt"
return_overflowing_tokens=True
```

so all overflow windows have compatible tensor dimensions.

The application also:

- splits long PDF text into smaller overlapping passages;
- preserves PDF page markers where text extraction provides them;
- supports TXT/PDF/DOCX;
- supports BERT, ALBERT and RoBERTa QA;
- supports BERT/RoBERTa/DistilBERT NER;
- supports BERT/ALBERT/RoBERTa contextual representations;
- includes FLAN-T5 chat;
- contains Learning and Application modules.

## Streamlit Cloud

Upload `app.py` and `requirements.txt` to GitHub and deploy.

If the PDF is scanned/image-only, `pypdf` may extract little or no text.
OCR would be required for scanned PDFs.

## Important

The QA model is extractive: it selects an answer span from the supplied
document/context. The score displayed is a model-score-derived indicator,
not proof of factual correctness.
