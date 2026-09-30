from pathlib import Path
from model_manager import ensure_model

print("Preparing local LLM...")
print(ensure_model("Qwen/Qwen2.5-0.5B-Instruct"))
print("Done.")
