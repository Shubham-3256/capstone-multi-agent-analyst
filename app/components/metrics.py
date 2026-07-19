"""Workflow metric helpers."""

from typing import Any


def workflow_metrics(result: Any) -> dict[str, float]:
    """Extract compact metrics from a WorkflowResult for UI rendering."""
    state = getattr(result, "state", None)
    timing = getattr(state, "timing", {}) if state else {}
    history = getattr(state, "execution_history", []) if state else []
    return {
        "nodes": float(len(history)),
        "seconds": round(sum(timing.values()), 3)
    }


def render_ml_metrics(ml_result: Any) -> None:
    """Render AutoML leaderboard and best model metrics."""
    import streamlit as st

    from app.components.cards import info_card
    from app.components.tables import render_html_table

    if not ml_result:
        st.info("No machine learning results available.")
        return

    st.subheader("Model Selection & Leaderboard")

    col1, col2 = st.columns(2)
    with col1:
        info_card(
            title="Best Performing Model",
            value=ml_result.best_model_name,
            subtitle="Based on validation metric score",
            icon="🏆",
            bg_gradient="linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
        )
    with col2:
        best_score = ml_result.best_metrics.get("f1") or ml_result.best_metrics.get("rmse") or 0.0
        metric_name = "macro-F1" if "f1" in ml_result.best_metrics else "RMSE"
        info_card(
            title=f"Best {metric_name} Score",
            value=f"{best_score:.4f}",
            subtitle="Validation set score",
            icon="📈",
            bg_gradient="linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
        )

    st.markdown("### Candidate Leaderboard")
    leaderboard = ml_result.leaderboard
    headers = ["Rank", "Model Name", "Score", "Secondary Metrics"]
    rows = []
    for entry in leaderboard.entries:
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in entry.metrics.items() if k != "f1" and k != "rmse")
        rows.append([
            str(entry.rank),
            entry.model_name,
            f"{entry.score:.4f}",
            metrics_str
        ])
    render_html_table(headers, rows)


def render_timing_metrics(result: Any) -> None:
    """Render horizontal bar chart for step execution times."""
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    state = getattr(result, "state", None)
    if not state or not state.timing:
        st.info("No timing records available.")
        return

    st.subheader("Pipeline Execution Timing")
    df = pd.DataFrame([
        {"Step": step.replace("_", " ").title(), "Duration (s)": duration}
        for step, duration in state.timing.items()
    ])
    fig = px.bar(
        df,
        y="Step",
        x="Duration (s)",
        orientation="h",
        color="Duration (s)",
        color_continuous_scale="Viridis",
        title="Time Spent per Pipeline Step"
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")

