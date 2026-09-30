
from pathlib import Path
import json
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CATALOGUE = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
MODEL_ROOT = ROOT / "local_models"

def all_models():
    out=[]
    for group in CATALOGUE.values():
        out.extend(group)
    return out

def local_path(model_id):
    return MODEL_ROOT / model_id.replace("/", "__")

def is_ready(model_id):
    p=local_path(model_id)
    return p.exists() and any(p.iterdir())

def ensure_model(model_id):
    p=local_path(model_id)
    p.mkdir(parents=True, exist_ok=True)
    if not is_ready(model_id):
        snapshot_download(repo_id=model_id, local_dir=str(p))
    return p
