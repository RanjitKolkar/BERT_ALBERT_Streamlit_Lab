"""Light-touch memory helpers for local Transformer demos.

Problem: Streamlit @st.cache_resource keeps every loaded model forever.
Compare-NER + Qwen can stack several GB in RAM.

Strategy:
  - keep at most one pipeline / one LLM in cache (max_entries=1)
  - load with low_cpu_mem_usage
  - explicit free_model_memory() for users
  - limit torch CPU threads
"""

from __future__ import annotations

import gc
import os
from typing import Any, Callable, Iterable, Optional

# Limit tokenizer/BLAS thread fan-out (set before heavy imports when possible)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_TORCH_CONFIGURED = False


def configure_torch_runtime() -> None:
    """Call once per process; safe to call repeatedly."""
    global _TORCH_CONFIGURED
    if _TORCH_CONFIGURED:
        return
    try:
        import torch

        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2 or 1)))
        torch.set_grad_enabled(False)
    except Exception:
        pass
    _TORCH_CONFIGURED = True


def preferred_dtype():
    """float16 on CUDA saves VRAM; float32 on CPU is the portable default."""
    import torch

    try:
        if torch.cuda.is_available():
            return torch.float16
    except Exception:
        pass
    return torch.float32


def model_load_kwargs() -> dict:
    configure_torch_runtime()
    return {
        "low_cpu_mem_usage": True,
        "torch_dtype": preferred_dtype(),
    }


def free_python_and_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_cached_resources(*cache_funcs: Callable) -> str:
    """Clear Streamlit cache_resource functions and run GC."""
    cleared = 0
    for fn in cache_funcs:
        try:
            fn.clear()
            cleared += 1
        except Exception:
            pass
    free_python_and_torch()
    return f"Cleared {cleared} model cache(s) and ran garbage collection."


def rss_mb() -> Optional[float]:
    """Best-effort process RSS in MiB (Windows/Unix)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        # Linux: ru_maxrss is KB; macOS is bytes
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix" and sys_platform_is_darwin():
            return usage / (1024 * 1024)
        return usage / 1024.0
    except Exception:
        return None


def sys_platform_is_darwin() -> bool:
    import sys

    return sys.platform == "darwin"


def memory_caption() -> str:
    mb = rss_mb()
    if mb is None:
        return (
            "Memory tip: only one model stays loaded. "
            "Use **Free model memory** after demos. Prefer Extractive chatbot mode."
        )
    return (
        f"Approx. process RAM: **{mb:.0f} MiB**. "
        "Only one heavy model is kept loaded. Use **Free model memory** after demos."
    )
