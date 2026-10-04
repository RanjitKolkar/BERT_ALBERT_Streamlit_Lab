"""
AI Document Lab — single-page guided flow (no left drawer)

Journey: Start → Install kit → Learn → Practice → Chat
"""

from __future__ import annotations

import streamlit as st

from runtime_memory import configure_torch_runtime
from ui_flow import FLOW_STEPS, callout, render_flow_header, render_nav_buttons

st.set_page_config(
    page_title="AI Document Lab",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_start() -> None:
    st.markdown("## Step 1 · Welcome")
    st.caption("One page. Clear order. Everything stays on your machine when you install.")

    callout(
        "<b>How to use this lab</b><br>"
        "Follow the numbered stages at the top from left to right the first time. "
        "There is <b>no left menu</b>. Use <b>Back / Continue</b> at the bottom of each stage."
    )

    st.markdown("### Your journey")
    for step in FLOW_STEPS:
        st.markdown(
            f"**{step['num']}. {step['label']} — {step['title']}**  \n"
            f"{step['blurb']}"
        )

    st.markdown("### First-time checklist")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Install kit**\n\nDownload ZIP + setup guide → unzip on your PC → `setup.bat`")
    with c2:
        st.info("**Learn + Practice**\n\nTheory lessons, then hands-on on shared forensic text")
    with c3:
        st.info("**Chat**\n\nUpload your documents and ask English questions (Extractive works immediately)")

    st.markdown("### Design rules")
    st.markdown(
        """
- **No API key** for the core lab  
- **Install = files on your machine** (ZIP kit + local models folder)  
- **Chat** can start with Extractive search even before neural models exist  
- **Practice** uses one shared document text across encoder / NER / LLM demos  
- This is a **teaching lab**, not AGI  
"""
    )

    from ui_flow import make_goto_callback

    b1, b2 = st.columns(2)
    with b1:
        st.button(
            "Start journey → Install kit",
            type="primary",
            use_container_width=True,
            key="start_to_install",
            on_click=make_goto_callback("install", mark_done_key="start"),
        )
    with b2:
        st.button(
            "Skip to Chat (Extractive)",
            use_container_width=True,
            key="start_to_chat",
            on_click=make_goto_callback("chat", mark_done_key="start"),
        )

    st.markdown("---")
    render_nav_buttons(
        prev_key=None,
        next_key="install",
        next_label="Continue to Install kit →",
        mark_done_key="start",
    )


def main() -> None:
    configure_torch_runtime()
    step = render_flow_header()

    if step == "start":
        render_start()
    elif step == "install":
        from install_module import render_install

        render_install()
    elif step == "learn":
        from learn_module import render_learn

        render_learn()
    elif step == "practice":
        from tutorial_lab import render_tutorial_lab

        render_tutorial_lab()
    else:
        from chat_module import render_chat

        render_chat()


if __name__ == "__main__":
    main()
