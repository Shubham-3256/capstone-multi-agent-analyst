"""Reusable chart renderers for Plotly figures and local images."""

from pathlib import Path
from typing import Any

import streamlit as st


def render_chart(chart_meta: Any) -> None:
    """Render a single chart metadata component, showing the image/HTML, caption, and download buttons."""

    from app.services.config import PROJECT_ROOT

    # 1. Resolve path robustly
    def resolve_path(p: str) -> Path | None:
        if not p:
            return None
        # Try direct absolute path
        path_obj = Path(p)
        if path_obj.exists():
            return path_obj
        # Try relative to project root
        proj_rel = PROJECT_ROOT / p
        if proj_rel.exists():
            return proj_rel
        # Try stripping leading workspace
        parts = Path(p).parts
        if parts and parts[0] == "workspace":
            trimmed = PROJECT_ROOT / "/".join(parts[1:])
            if trimmed.exists():
                return trimmed
        return None

    title = getattr(chart_meta, "title", "Chart")
    chart_id = getattr(chart_meta, "chart_id", "chart")
    caption_obj = getattr(chart_meta, "caption", None)
    summary = getattr(caption_obj, "summary", "") if caption_obj else ""
    details = getattr(caption_obj, "details", "") if caption_obj else ""

    st.markdown(f"### {title}")

    if chart_id == "missing_heatmap" and not getattr(chart_meta, "file_path", ""):
        st.info("No missing values detected.")
        return

    # Render HTML if available
    html_path_str = getattr(chart_meta, "html_path", None)
    html_path = resolve_path(html_path_str) if html_path_str else None
    file_path_str = getattr(chart_meta, "file_path", None)
    file_path = resolve_path(file_path_str) if file_path_str else None

    rendered = False
    if html_path and html_path.exists():
        try:
            with open(html_path, encoding="utf-8") as f:
                html_content = f.read()
            import urllib.parse

            data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(
                html_content
            )
            st.iframe(data_url, height=450)
            rendered = True
        except Exception as e:
            st.warning(f"Could not render interactive HTML chart: {e}")

    if not rendered and file_path and file_path.exists():
        st.image(str(file_path), width="stretch")
        rendered = True

    if not rendered:
        st.info("Chart visual asset is not available.")
        return

    # Render Captions
    if summary or details:
        with st.expander("📋 Chart Analysis & Explanations", expanded=True):
            if summary:
                st.markdown(f"**Key Summary:** {summary}")
            if details:
                st.markdown(f"**Analytical Details:** {details}")

    # Render Download Buttons
    col1, col2 = st.columns(2)
    if file_path and file_path.exists():
        with open(file_path, "rb") as f:
            btn_bytes = f.read()
        with col1:
            st.download_button(
                label="💾 Download PNG",
                data=btn_bytes,
                file_name=f"{chart_id}.png",
                mime="image/png",
                key=f"dl_png_{chart_id}",
            )

    if html_path and html_path.exists():
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        with col2:
            st.download_button(
                label="🌐 Download Interactive HTML",
                data=html_bytes,
                file_name=f"{chart_id}.html",
                mime="text/html",
                key=f"dl_html_{chart_id}",
            )
