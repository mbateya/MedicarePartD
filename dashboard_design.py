from __future__ import annotations

import streamlit as st


def install_design_system() -> None:
    st.markdown(
        """
<style>
:root {
    --app-bg: #f5f8fc;
    --surface: #ffffff;
    --surface-soft: #f8fbff;
    --text: #1f2f46;
    --muted: #667085;
    --line: #e4eaf2;
    --blue: #377bd7;
    --teal: #2f9c8f;
    --rose: #cf5fa9;
    --amber: #e5a44b;
    --nav: #08163a;
}
.stApp {
    background: var(--app-bg);
}
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px;
}
[data-testid="stSidebar"] {
    display: none;
}
.page-shell {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 420px);
    gap: 22px;
    align-items: start;
    margin: 6px 0 20px;
}
.page-crumb {
    color: #7a8494;
    font-size: 0.95rem;
    margin-bottom: 14px;
}
.page-title {
    color: var(--text);
    font-size: clamp(2rem, 3vw, 3rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: 0;
    margin: 0 0 8px;
}
.page-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 860px;
}
.ai-command {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 14px 34px rgba(32, 50, 75, 0.08);
}
.ai-command-label {
    font-size: 0.77rem;
    font-weight: 700;
    color: #667085;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 8px;
}
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 14px;
    box-shadow: 0 18px 42px rgba(31, 47, 70, 0.08);
}
.section-heading {
    margin: 34px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-heading-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #e8f2ff;
    color: var(--blue);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.section-heading-title {
    color: var(--text);
    font-size: 1.18rem;
    font-weight: 800;
}
.insight-strip {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-left: 4px solid var(--blue);
    border-radius: 8px;
    padding: 11px 14px;
    font-size: 0.88rem;
    color: #2c3b51;
    margin: 10px 0 16px;
    line-height: 1.6;
    box-shadow: 0 8px 22px rgba(31, 47, 70, 0.05);
}
.scope-note {
    background: #fff8ea;
    border: 1px solid #f2ddb6;
    border-left: 4px solid var(--amber);
    border-radius: 8px;
    padding: 11px 14px;
    font-size: 0.88rem;
    color: #745321;
    margin: 10px 0 18px;
    line-height: 1.6;
}
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px;
}
div[style*="min-height:82px"] {
    border-color: #e4eaf2 !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 28px rgba(31, 47, 70, 0.07);
}
button[kind="primary"], div.stButton > button {
    border-radius: 8px !important;
    font-weight: 700 !important;
}
@media (max-width: 900px) {
    .page-shell {
        grid-template-columns: 1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_header(
    *,
    title: str,
    subtitle: str,
    section: str,
    icon: str,
) -> None:
    from dashboard_ai import render_chatbot_button

    install_design_system()
    left, right = st.columns([4.8, 1.35], vertical_alignment="top")
    with left:
        st.markdown(
            f"""
<div class="page-crumb">{icon} / {section}</div>
<h1 class="page-title">{title}</h1>
<div class="page-subtitle">{subtitle}</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        with st.container(border=True):
            st.markdown('<div class="ai-command-label">Ask across rollups</div>', unsafe_allow_html=True)
            render_chatbot_button()


def section_heading(text: str, icon: str = "▦") -> None:
    st.markdown(
        f"""
<div class="section-heading">
  <span class="section-heading-icon">{icon}</span>
  <span class="section-heading-title">{text}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def insight_strip(text: str) -> None:
    st.markdown(f'<div class="insight-strip">{text}</div>', unsafe_allow_html=True)


def scope_note(text: str) -> None:
    st.markdown(f'<div class="scope-note">{text}</div>', unsafe_allow_html=True)


def chart_card(fig) -> None:
    st.plotly_chart(fig, use_container_width=True)
