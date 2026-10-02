# Easy install guide (normal users)

You only need **two double-clicks** on a Windows PC.

```text
1. setup.bat     → installs packages + one Local LLM download
2. run_app.bat   → opens the app in your browser
```

No API key. No manual model hunting. No command-line skills required.

---

## Before you start (once)

1. Install **Python 3.10 or newer** from: https://www.python.org/downloads/
2. On the installer screen, tick **Add python.exe to PATH**
3. Finish install, then open the project folder:
   `BERT_ALBERT_Streamlit_Lab`

---

## Step 1 — One download setup (recommended)

**Double-click:** `setup.bat`

What it does automatically:

| Step | Action | Your work |
|---|---|---|
| 1 | Installs app packages from `requirements.txt` | Wait |
| 2 | Downloads **only** the Local LLM (`Qwen2.5-0.5B-Instruct`) into `local_models/` | Wait |

- Needs internet the first time
- Usually several minutes (depends on your connection)
- Safe to re-run; already-downloaded files are kept
- Size is much smaller than downloading every tutorial model

When you see **SETUP COMPLETE**, press any key to close the window.

---

## Step 2 — Start the app

**Double-click:** `run_app.bat`

- Your browser should open the app
- Keep the black window open while you use it
- To stop: close that window or press `Ctrl+C`

### In the app sidebar

| Choice | What it is |
|---|---|
| **Document Context Chatbot** | Upload PDF/Word/Excel and ask English questions |
| **Tutorial Lab** | BERT / NER / ALBERT / Qwen teaching demos |

### Chatbot answer modes

| Mode | When to use |
|---|---|
| **Extractive** | Works immediately, even without Local LLM |
| **Local Qwen LLM** | Uses the one model downloaded by `setup.bat` |

If Local LLM is missing, the app also has a **Download Qwen** button.

---

## Optional — full classroom models (only if needed)

`setup.bat` downloads **Local LLM only** (simplest).

If teachers/students also need all NER + BERT + ALBERT models offline:

**Double-click:** `setup_all_models.bat`

Or in a terminal from the project folder:

```bash
python scripts/easy_setup.py --all
```

This is larger and slower. Use only when you need the full Tutorial Lab offline.

---

## One-command options (advanced / Mac / Linux)

From the project folder:

```bash
# Recommended: packages + Local LLM only (one model download)
python scripts/easy_setup.py

# Packages + every catalogue model
python scripts/easy_setup.py --all

# Model download only (packages already installed)
python scripts/easy_setup.py --skip-pip
python scripts/easy_setup.py --all --skip-pip

# Start app
streamlit run app.py
# or
python -m streamlit run app.py
```

Local LLM only:

```bash
python scripts/download_local_llm.py
```

---

## What gets installed where

```text
project folder/
  setup.bat                 ← double-click install (Local LLM)
  setup_all_models.bat      ← optional full pack
  run_app.bat               ← double-click start
  requirements.txt          ← Python packages list
  local_models/             ← downloaded model files (auto-created)
    Qwen__Qwen2.5-0.5B-Instruct/
  scripts/
    easy_setup.py           ← the real one-step installer
    download_local_llm.py   ← Local LLM only
    preload_all_models.py   ← all models only
```

---

## Troubleshooting (plain language)

### “Python was not found”
- Reinstall Python and tick **Add python.exe to PATH**
- Close and reopen the folder, then run `setup.bat` again

### Setup is slow / stuck on download
- Keep the window open; first download can take time
- Check internet / firewall / VPN
- Re-run `setup.bat` — it continues from what already finished

### App opens but Local LLM says missing
1. Run `setup.bat` again, or  
2. In the app, click **Download Qwen into local_models/**  
3. Confirm folder exists: `local_models/Qwen__Qwen2.5-0.5B-Instruct/`

### Streamlit Cloud
- **Extractive** chatbot works without model files
- Large `local_models/` files are usually **not** uploaded to Cloud
- For Local LLM + Tutorial models, run on your **PC** after `setup.bat`

### Out of disk space
- Local LLM alone needs roughly **1 GB+** free
- Full model pack needs several more GB

---

## All models (catalogue)

| Group | Models | Needed for chatbot? |
|---|---|---|
| **Local LLM** | Qwen2.5-0.5B-Instruct | Yes (only if you choose Local Qwen mode) |
| **NER** | BERT NER, DistilBERT NER, RoBERTa NER, XLM-R NER | No — Tutorial Lab |
| **Encoders** | BERT uncased, BERT cased, DistilBERT, RoBERTa, ALBERT | No — Tutorial Lab |

- `setup.bat` → **Local LLM only** (enough for chatbot generative mode)
- `setup_all_models.bat` → **all 10 models** for Tutorial Lab offline

### Check what is installed

1. Start app → **Tutorial Lab** → **Model Library** → **Refresh status**
2. Or read the status section in [USER_GUIDE.md](USER_GUIDE.md)

### How to use + best context-chatbot method

See the full guide: **[USER_GUIDE.md](USER_GUIDE.md)**

Short version for document Q&A:

1. Upload files → **Process documents**
2. Prefer **Extractive** first (answers = real passages from your files)
3. Switch to **Local Qwen** only when you want a fluent rewrite of those passages
4. Always check **sources** under the answer

That’s the standard **RAG** approach: *retrieve context first, then answer only from context*.

---

## Minimal checklist

- [ ] Python installed (with PATH)
- [ ] Double-click `setup.bat` once and wait for **SETUP COMPLETE**
- [ ] Double-click `run_app.bat`
- [ ] Upload a document and ask a question
- [ ] Optional: open [USER_GUIDE.md](USER_GUIDE.md) for all models and Tutorial Lab steps

That’s the whole install for a normal user.
