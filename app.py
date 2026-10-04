"""
AI Document Lab — rewritten main flow
-------------------------------------
1 · Chat              Start here — document Q&A with available project models
2 · Learn             Lessons through Transformers, RAG, LLMs, AGI context + hands-on
3 · Install yourself  Guided one-click model downloads

No API key required for core use.
"""

from __future__ import annotations

import streamlit as st

from runtime_memory import configure_torch_runtime, memory_caption

st.set_page_config(
    page_title="AI Document Lab — Chat · Learn · Install",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_MODES = [
    "1 · Chat",
    "2 · Learn",
    "3 · Install yourself",
]


def render_home_strip(mode: str) -> None:
    st.sidebar.markdown("### Flow")
    st.sidebar.markdown(
        """
1. **Chat** — use what is ready  
2. **Learn** — understand modules  
3. **Install** — one-click downloads  
"""
    )
    st.sidebar.caption(memory_caption())
    st.sidebar.markdown("---")
    st.sidebar.caption("Project files → local_models/ · catalogue → models.json")
    st.sidebar.caption("Desktop: run_app.bat · setup.bat")


def main() -> None:
    configure_torch_runtime()

    st.sidebar.title("AI Document Lab")
    st.sidebar.success("No API key required")
    mode = st.sidebar.radio("Go to", APP_MODES, key="app_main_mode")
    render_home_strip(mode)

    if mode == "1 · Chat":
        from chat_module import render_chat

        render_chat()
    elif mode == "2 · Learn":
        from learn_module import render_learn

        render_learn()
    else:
        from install_module import render_install

        render_install()


if __name__ == "__main__":
    main()
