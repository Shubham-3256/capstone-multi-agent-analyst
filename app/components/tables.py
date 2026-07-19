"""Reusable dataframe rendering helpers."""

from typing import Any

import streamlit as st


def render_table(data: Any, empty_message: str = "No data available yet.") -> None:
    """Render tabular data or a consistent empty state."""
    import pandas as pd

    if data is None:
        st.info(empty_message)
        return
    if isinstance(data, pd.DataFrame):
        if data.empty:
            st.info(empty_message)
            return
        st.dataframe(data, width="stretch", hide_index=True)
    elif isinstance(data, list):
        if not data:
            st.info(empty_message)
            return
        df = pd.DataFrame(data)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.write(data)


def render_html_table(
    headers: list[str],
    rows: list[list[Any]],
    empty_message: str = "No data available yet.",
) -> None:
    """Render a premium styled HTML table for clean data presentation."""
    if not headers or not rows:
        st.info(empty_message)
        return

    thead = "".join(f"<th>{h}</th>" for h in headers)
    tbody = ""
    for row in rows:
        tbody += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"

    html = f"""
    <div class="custom-table-container">
        <table class="custom-table">
            <thead>
                <tr>{thead}</tr>
            </thead>
            <tbody>
                {tbody}
            </tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
