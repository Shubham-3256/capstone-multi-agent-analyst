"""Reusable corporate metric-card renderer."""


import streamlit as st


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Render one compact enterprise metric card using Streamlit native components."""
    st.metric(label, value, delta)


def info_card(title: str, value: str, subtitle: str | None = None, icon: str | None = "📊", bg_gradient: str | None = None) -> None:
    """Render a premium enterprise card with custom HTML/CSS styling."""
    style_str = f"background: {bg_gradient};" if bg_gradient else ""
    html = f"""
    <div class="custom-card" style="{style_str}">
        <div class="custom-card-header">
            <span class="custom-card-icon">{icon}</span>
            <span class="custom-card-title">{title}</span>
        </div>
        <div class="custom-card-body">
            <div class="custom-card-value">{value}</div>
            {f'<div class="custom-card-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

