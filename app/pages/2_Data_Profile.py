"""Data Profiling dashboard screen."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.cards import info_card
from app.components.sidebar import setup_page
from app.components.tables import render_table
from app.services.session import get_workflow_result, initialize_session


def main() -> None:
    """Render the data profile page."""
    initialize_session()
    setup_page("Data Profile")

    st.title("📊 Dataset Profile & Data Quality")

    result = get_workflow_result()
    if not result:
        st.info("No active workflow result. Please upload a dataset and run the pipeline on the Upload page.")
        return

    di_result = getattr(result.state, "data_intelligence_result", None)
    profile = getattr(result.state, "dataset_profile", None)

    if not di_result or not profile:
        st.warning("No data profile information was produced in this workflow execution.")
        return

    # 1. High-level Profile Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        info_card("Total Rows", f"{profile.row_count:,}", "Cleaned dataset size", "📋")
    with col2:
        info_card("Total Columns", str(profile.column_count), "Feature dimensions", "📐")
    with col3:
        mem = profile.memory_usage_bytes
        mem_str = f"{mem / 1024:.2f} KB" if mem < 1024 * 1024 else f"{mem / (1024*1024):.2f} MB"
        info_card("Memory Footprint", mem_str, "Allocated RAM footprint", "💾")
    with col4:
        info_card("Suggested ML Task", profile.recommended_ml_task.title(), "Target based detection", "🤖")

    st.divider()

    # 2. Recommendations & Validation Issues
    st.subheader("Data Quality Warnings & Pipeline Recommendations")

    # Validation report
    val_report = di_result.validation_report
    if val_report and val_report.issues:
        with st.expander(f"⚠️ Validation Warnings ({len(val_report.issues)})", expanded=True):
            for issue in val_report.issues:
                sev = issue.severity.upper()
                col_txt = f" Column: `{issue.column}` |" if issue.column else ""
                st.markdown(f"**[{sev}]**{col_txt} {issue.message}")
    else:
        st.success("✅ **Dataset passed all structural validation checks!**")

    # Recommendations
    if profile.recommendations:
        with st.expander("💡 Modeling Recommendations", expanded=True):
            for idx, rec in enumerate(profile.recommendations, 1):
                col_txt = f" (`{rec.column}`)" if rec.column else ""
                st.markdown(f"**{idx}. {rec.title}**{col_txt} — {rec.description}")

    st.divider()

    # 3. Column Profiles
    st.subheader("Detailed Column Summary Profiles")

    # Compile a dataframe of columns
    col_rows = []
    for col_name, col_prof in profile.columns.items():
        # Compile summaries
        summary_txt = ""
        if col_prof.numeric_summary:
            s = col_prof.numeric_summary
            summary_txt = f"Mean: {s.get('mean', 0.0):.2f} | Min: {s.get('min', 0.0):.2f} | Max: {s.get('max', 0.0):.2f}"
        elif col_prof.categorical_summary:
            s = col_prof.categorical_summary
            summary_txt = f"Top Category: '{s.get('top', 'N/A')}' (Freq: {s.get('freq', 0)})"
        elif col_prof.date_summary:
            s = col_prof.date_summary
            summary_txt = f"Min Date: {s.get('min', 'N/A')} | Max Date: {s.get('max', 'N/A')}"

        col_rows.append({
            "Column Name": col_name,
            "Cleaned Type": col_prof.dtype,
            "Unique Count": col_prof.unique_count,
            "Null Count": col_prof.null_count,
            "Null %": f"{col_prof.null_percentage:.2f}%",
            "Summary Details": summary_txt
        })

    render_table(pd.DataFrame(col_rows))

    # 4. Target distribution
    target_dist = profile.target_distribution
    if target_dist:
        st.divider()
        st.subheader("Target Column Class Distribution")
        df_target = pd.DataFrame([
            {"Class / Value": str(k), "Count": v}
            for k, v in target_dist.items()
        ])
        fig = px.bar(
            df_target,
            x="Class / Value",
            y="Count",
            color="Class / Value",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title=f"Distribution of '{result.state.metadata.target_column}' target values"
        )
        st.plotly_chart(fig, width="stretch")

    # 5. Correlation matrix
    corr_matrix = profile.correlation_matrix
    if corr_matrix:
        st.divider()
        st.subheader("Numeric Features Correlation Matrix")
        df_corr = pd.DataFrame(corr_matrix)

        # Render a correlation heat map
        fig_corr = px.imshow(
            df_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            title="Pearson Correlation Coefficient Grid"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, width="stretch")


if __name__ == "__main__":
    main()
