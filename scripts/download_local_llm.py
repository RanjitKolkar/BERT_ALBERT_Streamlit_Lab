
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "local_models" / "Qwen2.5-0.5B-Instruct"
DEST.mkdir(parents=True, exist_ok=True)

print("Downloading Qwen2.5-0.5B-Instruct...")
print("Destination:", DEST)

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir=str(DEST),
    ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
)

print("\nDownload complete.")
print("The Streamlit application can now load the model from the project folder.")
