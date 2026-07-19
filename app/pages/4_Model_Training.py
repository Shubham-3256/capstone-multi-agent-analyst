"""Model Training and AutoML Leaderboard page."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.metrics import render_ml_metrics
from app.components.sidebar import setup_page
from app.services.session import get_workflow_result, initialize_session


def main() -> None:
    """Render the model page."""
    initialize_session()
    setup_page("Model Tuning & Results")

    st.title("🤖 AutoML Model Tuning & Evaluation")

    result = get_workflow_result()
    if not result:
        st.info("No active workflow result. Please upload a dataset and run the pipeline on the Upload page.")
        return

    ml_result = getattr(result.state, "ml_result", None)
    if not ml_result:
        # Check if skipped
        target_col = getattr(getattr(result.state, "metadata", None), "target_column", None)
        if not target_col:
            st.warning("AutoML model training was skipped because no target column was selected.")
        else:
            st.warning("No model training results found in the execution state.")
        return

    # 1. Render AutoML Metrics and Leaderboard
    split_rep = getattr(getattr(result.state, "feature_result", None), "split_report", None)
    train_shape = getattr(split_rep, "train_shape", None)
    if train_shape and train_shape[0] < 10:
        st.warning("⚠️ Results for demonstration only: The dataset is too small (fewer than 10 training samples) for robust model training.")
        
    render_ml_metrics(ml_result)

    st.divider()

    # 2. Best Model Hyperparameters & Details
    st.subheader("Best Model Configuration Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            **Estimator Model:** `{ml_result.best_model_name}`  
            **Saved Artifact Path:** `{ml_result.best_model_path}`  
            **Tuning Time:** {ml_result.duration_seconds:.2f} seconds
            """
        )
    with col2:
        # Check target classification or regression task
        task_type = ml_result.task_report.task_type.title()
        classes_str = ", ".join(map(str, ml_result.task_report.classes or []))
        st.markdown(
            f"""
            **ML Task Category:** `{task_type}`  
            **Unique Labels:** `{classes_str or "None (Continuous)"}`  
            **Binary Classification:** `{ml_result.task_report.is_binary}`
            """
        )

    st.markdown("### Best Tuned Hyperparameters")
    best_params = ml_result.leaderboard.entries[0] if ml_result.leaderboard.entries else None

    # Let's search for hyperparameter details
    # We can write it in an expander
    with st.expander("🛠️ View Hyperparameter Details", expanded=True):
        # We can extract details from the best model entry or search for them
        # Let's see: MachineLearningResult has some details
        # Let's write them as code block
        st.code(str(getattr(ml_result, "best_params", {})) or "{'tuned': True}")

    # 3. Best Model Feature Importance Chart
    if ml_result.feature_importances:
        st.divider()
        st.subheader("Global Feature Importance Chart")
        st.markdown("Feature importances calculated from the best fitted AutoML estimator:")

        df_imp = pd.DataFrame([
            {"Feature": item.column, "Importance": item.importance}
            for item in ml_result.feature_importances
        ]).sort_values("Importance", ascending=True)

        fig = px.bar(
            df_imp,
            y="Feature",
            x="Importance",
            orientation="h",
            color="Importance",
            color_continuous_scale="Reds",
            title=f"Feature Relevance Scores - {ml_result.best_model_name}"
        )
        st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
