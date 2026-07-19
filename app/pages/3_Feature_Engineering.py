"""Feature Engineering dashboard page."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.cards import info_card
from app.components.sidebar import setup_page
from app.components.tables import render_html_table, render_table
from app.services.session import get_workflow_result, initialize_session


def main() -> None:
    """Render the feature engineering page."""
    initialize_session()
    setup_page("Feature Engineering")

    st.title("🛠️ Feature Engineering & Data Splits")

    result = get_workflow_result()
    if not result:
        st.info("No active workflow result. Please upload a dataset and run the pipeline on the Upload page.")
        return

    fe_result = getattr(result.state, "feature_result", None)
    if not fe_result:
        # Check if skipped
        target_col = getattr(getattr(result.state, "metadata", None), "target_column", None)
        if not target_col:
            st.warning("Feature engineering was skipped because no target column was selected.")
        else:
            st.warning("No feature engineering result was found in the execution state.")
        return

    # 1. Split Dimensions Stats
    st.subheader("Data Partition Shapes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tr = fe_result.split_report.train_shape
        info_card("Training Split", f"{tr[0]:,} × {tr[1]}", "Used for model training", "🏋️")
    with col2:
        val = fe_result.split_report.val_shape
        info_card("Validation Split", f"{val[0]:,} × {val[1]}", "Used for hyperparameter tuning", "🧪")
    with col3:
        ts = fe_result.split_report.test_shape
        info_card("Test Split", f"{ts[0]:,} × {ts[1]}", "Used for final testing", "🏁")
    with col4:
        info_card("Splitting Strategy", fe_result.split_report.strategy.title(), "Target stratification choice", "⚖️")

    st.divider()

    # 2. Leakage Check
    st.subheader("Data Leakage Audit Checks")
    leak_rep = fe_result.leakage_report
    if leak_rep.has_leakage:
        st.error("⚠️ **Data Leakage Detected!** Please review the details below.")
        for issue in leak_rep.leakage_issues:
            col_txt = f"`{issue.get('column')}`:" if issue.get("column") else ""
            st.markdown(f"* {col_txt} {issue.get('issue', 'Potential data leak')}")
    else:
        st.success("✅ **No target variable or identifier column leakage issues detected.**")

    st.divider()

    # 3. Interactive Tabs for Detailed reports
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏷️ Feature Types",
        "🔤 Encodings",
        "⚖️ Scalings",
        "🎯 Feature Selection",
        "📦 Serialization Pipeline"
    ])

    with tab1:
        st.subheader("Detected Feature Type Classifications")
        st.markdown("Features were automatically audited and grouped by datatype class:")
        types_df = pd.DataFrame([
            {"Feature Name": col, "Inferred Type": val.title()}
            for col, val in fe_result.feature_types.items()
        ])
        render_table(types_df)

    with tab2:
        st.subheader("Categorical Feature Encoding Mappings")
        enc_rep = fe_result.encoding_report
        if enc_rep.strategy_used:
            st.markdown("The following categories were encoded for model ingestion:")
            headers = ["Feature", "Strategy", "Category Mappings"]
            rows = []
            for col, strat in enc_rep.strategy_used.items():
                mapping_str = " -> ".join(f"'{k}': {v}" for k, v in enc_rep.mappings.get(col, {}).items())
                rows.append([col, strat.upper(), mapping_str or "None"])
            render_html_table(headers, rows)
        else:
            st.info("No categorical features required encoding.")

    with tab3:
        st.subheader("Numerical Scaling Coefficients")
        scale_rep = fe_result.scaling_report
        if scale_rep.scaler_type:
            st.markdown("Fitted preprocessing coefficients scaling bounds:")
            headers = ["Feature", "Scaler Type", "Coefficient Parameters"]
            rows = []
            for col, s_type in scale_rep.scaler_type.items():
                params = scale_rep.scaling_parameters.get(col, {})
                params_str = ", ".join(f"{k}: {v:.4f}" for k, v in params.items())
                rows.append([col, s_type.title(), params_str])
            render_html_table(headers, rows)
        else:
            st.info("No numerical features scaling was executed.")

    with tab4:
        st.subheader("Feature Dimensions Selection")
        sel_rep = fe_result.selection_report
        st.markdown(
            f"""
            **Method:** `{sel_rep.method.upper()}`  
            **Features Input Count:** {sel_rep.original_count}  
            **Features Selected Count:** {sel_rep.selected_count} (Pruned {sel_rep.original_count - sel_rep.selected_count} features)
            """
        )

        if sel_rep.feature_importances:
            st.markdown("### Selection Feature Importance Scores")
            df_imp = pd.DataFrame([
                {"Feature": col, "Importance Score": val}
                for col, val in sel_rep.feature_importances.items()
            ]).sort_values("Importance Score", ascending=True)

            fig = px.bar(
                df_imp,
                y="Feature",
                x="Importance Score",
                orientation="h",
                color="Importance Score",
                color_continuous_scale="Teal",
                title=f"Feature Relevance Scores via {sel_rep.method}"
            )
            st.plotly_chart(fig, width="stretch")

    with tab5:
        st.subheader("Fitted Preprocessing Pipeline Summary")
        pipe_rep = fe_result.pipeline_report
        st.markdown(
            f"""
            **Saved Pipeline File Path:**  
            `{pipe_rep.pipeline_filepath}`
            
            **Components Pipeline Execution Order:**
            """
        )
        for idx, comp in enumerate(pipe_rep.components, 1):
            st.markdown(f"**Step {idx}:** `{comp.replace('_', ' ').title()}`")


if __name__ == "__main__":
    main()
