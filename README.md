# BERT & ALBERT Teaching Lab — Streamlit Deployment

This version is designed specifically for Streamlit deployment.

## Models

- BERT: bert-base-uncased
- ALBERT: albert-base-v2

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - app.py
   - requirements.txt
3. In Streamlit Community Cloud choose **Deploy an app**.
4. Select the repository and `app.py`.
5. Deploy.

No Windows virtual environment or BAT file is required.

## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How models are handled

The selected model is downloaded from Hugging Face the first time it is used.
`st.cache_resource` keeps the loaded model available during the application
session so it is not repeatedly loaded for every button click.

Only the selected model is loaded through the application function.

## Input

- Direct text
- Questions
- TXT
- PDF
- DOCX

## Output

- Tokenization
- Token IDs
- Transformer output tensor dimensions
- Example contextual representation
- Teaching explanation

## Important

The application demonstrates pretrained encoder models. It does not claim
that generic BERT or ALBERT can answer arbitrary questions or classify arbitrary
documents by themselves.

For true question answering, sentiment analysis, phishing classification,
forensic classification, etc., a task-specific model/head should be used.

## Streamlit deployment note

CPU inference is used. No CUDA/GPU setup is required.
