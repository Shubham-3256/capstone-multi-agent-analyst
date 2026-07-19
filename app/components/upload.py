"""Dataset upload widget with presentation-only preview."""

from typing import Any

import pandas as pd
import streamlit as st


def upload_dataset() -> Any | None:
    """Render a drag-and-drop input and show a small pre-execution preview."""
    uploaded = st.file_uploader(
        "Upload Dataset File",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supported formats: CSV, Excel (XLSX, XLS), Parquet"
    )
    if uploaded:
        st.markdown(f"📊 **File Uploaded:** `{uploaded.name}` ({uploaded.size / 1024:.2f} KB)")
        try:
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            elif name.endswith(".parquet"):
                df = pd.read_parquet(uploaded)
            else:
                st.error("Unsupported file format uploaded.")
                return None

            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(10), width="stretch")

            # Reset file pointer for downstream operations
            uploaded.seek(0)
            return uploaded
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
            return None
    return None

