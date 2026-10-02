from pathlib import Path
import json
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CATALOGUE = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
MODEL_ROOT = ROOT / "local_models"

CONFIG_PATTERNS = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "special_tokens*",
    "added_tokens*",
    "spm*",
    "sentencepiece*",
    "*.model",
    "chat_template*",
    "preprocessor_config.json",
]


def all_models():
    out = []
    for group in CATALOGUE.values():
        out.extend(group)
    return out


def local_path(model_id: str) -> Path:
    return MODEL_ROOT / model_id.replace("/", "__")


def is_ready(model_id: str) -> bool:
    path = local_path(model_id)
    if not path.exists() or not (path / "config.json").exists():
        return False
    has_weight = any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    # also accept sharded index + shards
    has_weight = has_weight or (path / "model.safetensors.index.json").exists() or (path / "pytorch_model.bin.index.json").exists()
    return has_weight


def ensure_model(model_id: str, progress_callback=None) -> Path:
    """Download one practical weight format into local_models/ if missing."""
    path = local_path(model_id)
    path.mkdir(parents=True, exist_ok=True)
    if is_ready(model_id):
        if progress_callback:
            progress_callback(f"Already ready: {model_id}")
        return path

    if progress_callback:
        progress_callback(f"Downloading: {model_id}")

    # Prefer safetensors only first (smaller, one format). Fall back to pytorch bin.
    attempts = [
        CONFIG_PATTERNS + ["*.safetensors", "*.safetensors.index.json"],
        CONFIG_PATTERNS + ["pytorch_model*.bin", "pytorch_model.bin.index.json"],
    ]
    last_error = None
    for patterns in attempts:
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(path),
                allow_patterns=patterns,
                ignore_patterns=["onnx/*", "openvino/*", "coreml/*", "*.h5", "*.ot", "*.msgpack"],
                max_workers=4,
            )
            if is_ready(model_id):
                if progress_callback:
                    progress_callback(f"Ready: {model_id}")
                return path
        except Exception as exc:
            last_error = exc

    if not is_ready(model_id):
        raise RuntimeError(f"Download finished but model is not ready: {model_id}. Last error: {last_error}")
    return path


def ensure_all_models(progress_callback=None):
    """Preload every model listed in models.json into local_models/."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    models = all_models()
    total = len(models)
    for index, item in enumerate(models, start=1):
        model_id = item["id"]
        name = item.get("name", model_id)
        if progress_callback:
            progress_callback(f"[{index}/{total}] {name} ({model_id})")
        try:
            path = ensure_model(model_id, progress_callback=progress_callback)
            results.append({"id": model_id, "name": name, "path": str(path), "ok": True, "error": None})
        except Exception as exc:
            results.append({"id": model_id, "name": name, "path": str(local_path(model_id)), "ok": False, "error": str(exc)})
            if progress_callback:
                progress_callback(f"FAILED: {model_id} -> {exc}")
    return results


def status_report():
    rows = []
    for item in all_models():
        model_id = item["id"]
        path = local_path(model_id)
        rows.append({
            "name": item.get("name", model_id),
            "id": model_id,
            "ready": is_ready(model_id),
            "path": str(path),
        })
    return rows
