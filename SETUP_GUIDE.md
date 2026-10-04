# AI Document Lab — Setup Guide (your machine)

This guide installs the lab **on your PC**. Models and packages stay local. **No API key.**

---

## What you need

| Item | Notes |
|---|---|
| Windows 10/11 (or any OS with Python) | Scripts below are Windows `.bat`; Linux/Mac can run the Python commands |
| Python 3.10+ | From https://www.python.org/downloads/ — tick **Add python.exe to PATH** |
| Internet (first time) | To install packages and (optionally) download models |
| Disk space | Beginner pack ~1–3 GB; full catalogue larger |

---

## Path A — Fastest (recommended)

### 1. Get the Setup Kit ZIP
In the app: **Step 2 · Install kit** → download **Lab Setup Kit (ZIP)**.  
Or copy this project folder if you already have the full repo.

### 2. Unzip on your machine
Example:

```text
C:\Users\<you>\AI_Document_Lab\
```

You should see at least:

- `setup.bat`
- `run_app.bat`
- `requirements.txt`
- `models.json`
- `model_manager.py`
- `scripts\easy_setup.py`
- `SETUP_GUIDE.md` (this file)

### 3. Double-click `setup.bat`
This will:

1. `pip install -r requirements.txt`
2. Download the **Local LLM (Qwen)** once into `local_models\`

Wait until it finishes. Leave the window open.

### 4. Start the app
Double-click **`run_app.bat`**.  
Browser opens → use the **top journey bar** (no left menu):

```text
Start → Install kit → Learn → Practice → Chat
```

---

## Path B — Classroom / full offline models

After Path A works:

1. Double-click **`setup_all_models.bat`**, **or**
2. In the app Install kit stage, use **Beginner pack** / **Everything** (needs internet once)

Files land in:

```text
local_models\
```

Copy the whole project folder (including `local_models`) to offline lab PCs if needed.

---

## Path C — Manual Python commands

```bash
cd AI_Document_Lab
python -m pip install -r requirements.txt
python scripts/easy_setup.py
python -m streamlit run app.py
```

---

## Recommended install order

| Order | What | Why |
|---|---|---|
| 1 | Packages only (`pip install -r requirements.txt`) | App can run |
| 2 | Nothing else | **Chat Extractive** works with zero neural models |
| 3 | Qwen (`setup.bat`) | Fluent local answers in Chat / Practice |
| 4 | DistilBERT + DistilBERT NER | Fast teaching demos |
| 5 | Full catalogue | Paper comparisons; large disk/RAM |

**Skip first:** RoBERTa-large NER on low-RAM laptops.

---

## App journey (after setup)

| Step | Name | You do |
|---|---|---|
| 1 | Start | Read the map |
| 2 | Install kit | ZIP + guide; check model status |
| 3 | Learn | Theory lessons in order |
| 4 | Practice | Shared forensic text → encoder → NER → LLM |
| 5 | Chat | Upload your own PDFs and ask questions |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python with PATH ticked; reopen terminal |
| pip errors | `python -m pip install --upgrade pip` then retry |
| App won't start | Run `setup.bat` once; then `run_app.bat` |
| Out of memory | Use Chat **Extractive**; free model memory; avoid large NER |
| Models missing | Re-run `setup.bat` or Beginner pack; check `local_models\` |
| Slow first download | Normal; leave the window open |

---

## Safety / teaching notes

- Answers should stay **grounded in your documents** when doing document Q&A.
- This lab is **not AGI** — models are narrow teaching tools.
- No cloud API key is required for the core flow.

---

## Files cheat sheet

| File | Role |
|---|---|
| `setup.bat` | Packages + Qwen |
| `setup_all_models.bat` | Packages + all models |
| `run_app.bat` | Launch Streamlit app |
| `requirements.txt` | Python packages |
| `models.json` | Model catalogue |
| `local_models/` | Downloaded weights |
| `SETUP_GUIDE.md` | This guide |
