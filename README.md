# Transformer AI Teaching Lab

A Streamlit teaching and experimentation application covering BERT, ALBERT,
RoBERTa, NER, extractive Question Answering, text representations and a
separate generative chat demonstration.

## Main structure

### Learning Module

- Transformers
- BERT
- ALBERT
- NER
- Question Answering
- Tokenization
- Self-Attention
- Embeddings
- BERT vs ALBERT
- Encoder vs Generative Models

### Application Module

1. Question Answering
2. Named Entity Recognition
3. Text Representation
4. Chat / General AI

## Inputs

Application tasks can accept:

- Typed/pasted text
- TXT
- PDF
- DOCX

## QA models

- BERT: deepset/bert-base-cased-squad2
- ALBERT: twmkn9/albert-base-v2-squad2
- RoBERTa: deepset/roberta-base-squad2

## NER models

- BERT NER: dslim/bert-base-NER
- RoBERTa NER: Jean-Baptiste/roberta-large-ner-english
- DistilBERT NER: elastic/distilbert-base-uncased-finetuned-conll03-english

## Text representation

- BERT: bert-base-uncased
- ALBERT: albert-base-v2
- RoBERTa: roberta-base

## Chat

The chat demonstration uses:

- google/flan-t5-small

This is intentional. BERT and ALBERT are encoder models and are not
general-purpose text generators.

## Streamlit deployment

Upload `app.py` and `requirements.txt` to GitHub and deploy `app.py`
using Streamlit Community Cloud.

No Windows-specific setup is required.

Models are downloaded automatically from Hugging Face on first use and
cached by Streamlit during the running application.

## Teaching methodology

```text
LEARN
  ↓
Understand architecture
  ↓
SELECT TASK
  ↓
Provide input
  ↓
SELECT MODEL
  ↓
RUN INFERENCE
  ↓
OBSERVE OUTPUT
  ↓
EXPLAIN HOW IT WORKED
  ↓
COMPARE MODELS
```

## Important

The model score/confidence shown in QA and NER is an indicator produced from
model outputs. It should not be treated as proof of factual correctness.

The chat module is a small generative demonstration and is not intended to
be equivalent to a large commercial conversational model.
