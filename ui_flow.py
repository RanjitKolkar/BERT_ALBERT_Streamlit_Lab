"""Shared single-page flow UI — no left drawer."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

# Ordered product journey (main page only)
FLOW_STEPS = [
    {
        "key": "start",
        "num": 1,
        "label": "Start",
        "title": "Welcome & journey",
        "blurb": "See the path and what each stage does",
    },
    {
        "key": "install",
        "num": 2,
        "label": "Install kit",
        "title": "Get files on your machine",
        "blurb": "Download ZIP + setup guide, then optional models",
    },
    {
        "key": "learn",
        "num": 3,
        "label": "Learn",
        "title": "Concepts step-by-step",
        "blurb": "AI → Transformers → RAG → LLM (theory)",
    },
    {
        "key": "practice",
        "num": 4,
        "label": "Practice",
        "title": "Hands-on tutorial",
        "blurb": "Shared text → encoders → NER → local LLM",
    },
    {
        "key": "chat",
        "num": 5,
        "label": "Chat",
        "title": "Document Q&A",
        "blurb": "Upload files and ask grounded questions",
    },
]


def inject_app_css() -> None:
    """Hide Streamlit left drawer and style the on-page flow."""
    st.markdown(
        """
<style>
  /* Hide left drawer completely */
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }
  [data-testid="stSidebarNav"] { display: none !important; }
  section.main > div { max-width: 1100px; margin: 0 auto; }

  .flow-wrap {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    border-radius: 16px;
    padding: 1.1rem 1.25rem 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid #334155;
  }
  .flow-wrap h1 {
    margin: 0 0 0.25rem 0;
    font-size: 1.55rem;
    color: #f8fafc;
    font-weight: 700;
  }
  .flow-wrap .sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 0.9rem;
  }
  .step-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: stretch;
  }
  .step-pill {
    flex: 1 1 120px;
    min-width: 110px;
    background: #0b1220;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.55rem 0.65rem;
    text-align: center;
  }
  .step-pill.active {
    background: #1d4ed8;
    border-color: #60a5fa;
    box-shadow: 0 0 0 2px rgba(96,165,250,0.35);
  }
  .step-pill.done {
    border-color: #22c55e;
    background: #052e1b;
  }
  .step-pill .n {
    font-weight: 800;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    color: #93c5fd;
  }
  .step-pill.active .n { color: #fff; }
  .step-pill.done .n { color: #86efac; }
  .step-pill .t {
    display: block;
    font-weight: 650;
    font-size: 0.92rem;
    margin-top: 0.15rem;
    color: #f1f5f9;
  }
  .step-pill .b {
    display: block;
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.2rem;
    line-height: 1.25;
  }
  .step-pill.active .b { color: #dbeafe; }

  .panel-card {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: #ffffff;
    margin-bottom: 0.85rem;
  }
  .panel-card h3 {
    margin-top: 0;
    margin-bottom: 0.35rem;
  }
  .callout {
    border-left: 4px solid #2563eb;
    background: #eff6ff;
    padding: 0.75rem 1rem;
    border-radius: 0 10px 10px 0;
    margin: 0.6rem 0 0.9rem 0;
  }
  .callout.warn {
    border-left-color: #d97706;
    background: #fffbeb;
  }
  .callout.ok {
    border-left-color: #16a34a;
    background: #f0fdf4;
  }
  .muted { color: #64748b; font-size: 0.9rem; }
  .progress-label {
    font-size: 0.85rem;
    color: #475569;
    margin-bottom: 0.25rem;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def init_flow_state() -> None:
    if "flow_step" not in st.session_state:
        st.session_state.flow_step = "start"
    if "flow_completed" not in st.session_state:
        st.session_state.flow_completed = set()
    # ensure set type after Streamlit serialisation edge cases
    if not isinstance(st.session_state.flow_completed, set):
        st.session_state.flow_completed = set(st.session_state.flow_completed or [])


def mark_step_done(key: str) -> None:
    init_flow_state()
    done = set(st.session_state.flow_completed)
    done.add(key)
    st.session_state.flow_completed = done


def goto_step(key: str) -> None:
    init_flow_state()
    st.session_state.flow_step = key
    # Keep top radio in sync with programmatic navigation
    labels = [f"{s['num']}. {s['label']} — {s['title']}" for s in FLOW_STEPS]
    keys = [s["key"] for s in FLOW_STEPS]
    if key in keys:
        st.session_state.flow_nav_radio = labels[keys.index(key)]
    st.rerun()


def current_step_index() -> int:
    init_flow_state()
    keys = [s["key"] for s in FLOW_STEPS]
    key = st.session_state.flow_step
    return keys.index(key) if key in keys else 0


def render_flow_header() -> str:
    """Top journey map + step chooser. Returns current step key."""
    init_flow_state()
    inject_app_css()

    idx = current_step_index()
    done = set(st.session_state.flow_completed)
    pills_html = []
    for i, step in enumerate(FLOW_STEPS):
        classes = ["step-pill"]
        if step["key"] == st.session_state.flow_step:
            classes.append("active")
        elif step["key"] in done or i < idx:
            classes.append("done")
        pills_html.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="n">STEP {step["num"]}</span>'
            f'<span class="t">{step["label"]}</span>'
            f'<span class="b">{step["blurb"]}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
<div class="flow-wrap">
  <h1>AI Document Lab</h1>
  <div class="sub">One-page guided flow · No left menu · No API key · Models stay on your PC</div>
  <div class="step-row">
    {''.join(pills_html)}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar
    pct = (idx) / max(len(FLOW_STEPS) - 1, 1)
    st.markdown(
        f'<div class="progress-label">Journey progress · Step {idx + 1} of {len(FLOW_STEPS)} · '
        f'{FLOW_STEPS[idx]["title"]}</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, pct if idx < len(FLOW_STEPS) - 1 else 1.0))

    labels = [f"{s['num']}. {s['label']} — {s['title']}" for s in FLOW_STEPS]
    choice = st.radio(
        "Jump to stage",
        labels,
        index=idx,
        horizontal=True,
        label_visibility="collapsed",
        key="flow_nav_radio",
    )
    chosen_key = FLOW_STEPS[labels.index(choice)]["key"]
    if chosen_key != st.session_state.flow_step:
        st.session_state.flow_step = chosen_key
        st.rerun()

    return st.session_state.flow_step


def render_nav_buttons(
    *,
    prev_key: Optional[str] = None,
    next_key: Optional[str] = None,
    next_label: str = "Continue →",
    mark_done_key: Optional[str] = None,
) -> None:
    """Bottom previous / next for linear flow."""
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if prev_key:
            if st.button("← Back", use_container_width=True, key=f"nav_back_{prev_key}"):
                goto_step(prev_key)
        else:
            st.write("")
    with c2:
        st.caption("Follow steps in order the first time. You can jump anytime using the top bar.")
    with c3:
        if next_key:
            if st.button(next_label, type="primary", use_container_width=True, key=f"nav_next_{next_key}"):
                if mark_done_key:
                    mark_step_done(mark_done_key)
                goto_step(next_key)


def panel_start(title: str, body_html: str = "") -> None:
    st.markdown(
        f'<div class="panel-card"><h3>{title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    cls = "callout"
    if kind == "warn":
        cls += " warn"
    elif kind == "ok":
        cls += " ok"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def lesson_stepper(
    titles: List[str],
    index_key: str,
    *,
    total_label: str = "Lesson",
) -> int:
    """Previous / progress / next for lessons or practice steps."""
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    idx = int(st.session_state[index_key])
    idx = max(0, min(idx, len(titles) - 1))
    st.session_state[index_key] = idx

    st.progress((idx + 1) / len(titles))
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Previous", disabled=idx <= 0, use_container_width=True, key=f"{index_key}_prev"):
            st.session_state[index_key] = idx - 1
            st.rerun()
    with c2:
        st.markdown(
            f"**{total_label} {idx + 1} / {len(titles)}**  \n"
            f"<span class='muted'>{titles[idx]}</span>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button(
            "Next →",
            disabled=idx >= len(titles) - 1,
            type="primary",
            use_container_width=True,
            key=f"{index_key}_next",
        ):
            st.session_state[index_key] = idx + 1
            st.rerun()
    return idx
