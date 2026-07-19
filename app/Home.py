"""Home page for the Multi-Agent AI Data Analyst Dashboard."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.cards import info_card
from app.components.sidebar import setup_page
from app.components.tables import render_html_table
from app.services.history_service import HistoryService
from app.services.session import initialize_session


def render_mermaid() -> None:
    """Render the orchestration flow chart dynamically using Mermaid.js."""
    st.subheader("Orchestration Pipeline Flow")
    mermaid_html = """
    <div style="background: rgba(255,255,255,0.02); border-radius: 8px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({ startOnLoad: true, theme: 'dark' });
        </script>
        <div class="mermaid" style="display: flex; justify-content: center;">
            graph TD
                A[User Uploads Dataset] --> B[load_dataset Node]
                B --> C[data_intelligence Node]
                C -->|Checks Schema & Cleans| D{Has target column?}
                D -->|No target column| G[visualization Node]
                D -->|Yes target column| E[feature_engineering Node]
                E --> F[machine_learning Node]
                F --> G
                G --> H[business_insights Node]
                H --> I[report_generation Node]
                I --> J[finalize Node]
                J --> K((Workflow End))
                
                style A fill:#1e3c72,stroke:#fff,stroke-width:1px
                style C fill:#2a5298,stroke:#fff,stroke-width:1px
                style F fill:#11998e,stroke:#fff,stroke-width:1px
                style H fill:#38ef7d,stroke:#fff,stroke-width:1px,color:#000
                style I fill:#f39c12,stroke:#fff,stroke-width:1px
                style K fill:#e74c3c,stroke:#fff,stroke-width:1px
        </div>
    </div>
    """
    import urllib.parse

    data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(mermaid_html)
    st.iframe(data_url, height=1200)


def main() -> None:
    """Render the dashboard Home page."""
    initialize_session()
    setup_page("Home")

    st.title("🤖 Multi-Agent AI Data Analyst")
    st.markdown("""
        Welcome to the **Multi-Agent AI Data Analyst** platform. This system orchestrates specialized 
        autonomous AI agents to ingest, clean, explore, model, and compile PDF reports for your datasets automatically.
        """)

    st.divider()

    # 1. Quick Statistics
    st.subheader("System Telemetry Summary")
    workflows = HistoryService.workflows()
    reports = HistoryService.reports()
    datasets = HistoryService.datasets()

    total_runs = len(workflows)
    total_reps = len(reports)
    total_data = len(datasets)

    # Calculate average duration
    durations = []
    for w in workflows:
        import json

        try:
            timing = json.loads(w.timing_json)
            durations.append(sum(timing.values()))
        except Exception:
            pass
    avg_duration = f"{sum(durations) / len(durations):.1f}s" if durations else "0.0s"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        info_card("Datasets Registered", str(total_data), "Uploaded in workspace", "📂")
    with col2:
        info_card("Pipeline Runs", str(total_runs), "Total graph executions", "⚙️")
    with col3:
        info_card("Average Run Time", avg_duration, "Across all agent phases", "⏱️")
    with col4:
        info_card("Reports Generated", str(total_reps), "PDF/DOCX/HTML formats", "📄")

    st.divider()

    # 2. Workflow flow diagram
    render_mermaid()

    st.divider()

    # 3. Recent Workflows and Reports
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recent Workflow Executions")
        if workflows:
            headers = ["Workflow ID", "Status", "Executed At"]
            rows = [
                [
                    w.workflow_id[:8] + "...",
                    w.status.upper(),
                    w.created_at.strftime("%Y-%m-%d %H:%M"),
                ]
                for w in workflows[:5]
            ]
            render_html_table(headers, rows)
        else:
            st.info("No workflows executed yet.")

    with c2:
        st.subheader("Recent Reports")
        if reports:
            headers = ["Report ID", "Template Type", "Created At"]
            rows = [
                [
                    r.report_id[:8] + "...",
                    r.template_type.title(),
                    r.created_at.strftime("%Y-%m-%d %H:%M"),
                ]
                for r in reports[:5]
            ]
            render_html_table(headers, rows)
        else:
            st.info("No reports compiled yet.")


if __name__ == "__main__":
    main()
