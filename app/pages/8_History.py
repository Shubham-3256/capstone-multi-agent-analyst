"""Workflow History and Execution Logs screen."""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.sidebar import setup_page
from app.components.tables import render_table, render_html_table
from app.services.session import initialize_session, set_workflow_result, set_uploaded_dataset_path
from app.services.history_service import HistoryService
from src.core.paths import Paths
from src.orchestration.checkpoint import FileCheckpointStore
from src.orchestration.state import WorkflowResult


def load_past_workflow_result(workflow_id: str) -> bool:
    """Load a prior file-based checkpoint into the active session state."""
    try:
        store = FileCheckpointStore(Paths.WORKSPACE_DIR / "checkpoints")
        state = store.load(workflow_id)
        if state:
            result = WorkflowResult(
                is_success=not state.errors,
                state=state,
                output_paths=dict(getattr(state.report_result, "output_paths", {}) or {})
            )
            set_workflow_result(result)
            set_uploaded_dataset_path(state.dataset_path)
            return True
    except Exception as e:
        st.error(f"Error loading checkpoint for workflow ID {workflow_id}: {str(e)}")
    return False


def main() -> None:
    """Render the history and logs page."""
    initialize_session()
    setup_page("History & Logs")

    st.title("📜 Execution History & Audit Logs")
    st.markdown(
        """
        Audit prior analytics runs, inspect error details, compare execution durations, 
        and reload prior workflows back into the active presentation workspace to browse their charts and findings.
        """
    )
    st.divider()

    # Search & Filter controls
    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "Completed", "Degraded"],
            index=0
        )
    with col_c2:
        search_query = st.text_input(
            "Search by ID or keywords",
            value="",
            placeholder="Search workflow ID..."
        )

    # Fetch history
    workflows = HistoryService.workflows(status_filter=status_filter, search_query=search_query)

    if not workflows:
        st.info("No workflow execution logs found matching the filters.")
        return

    # Render interactive table of executions with "Inspect" select box
    st.subheader(f"Workflow Executions ({len(workflows)} matches)")
    
    # We will build a styled DataFrame and display it
    rows = []
    for idx, w in enumerate(workflows, 1):
        try:
            errors = json.loads(w.errors_json)
            timing = json.loads(w.timing_json)
            duration = f"{sum(timing.values()):.2f}s"
            err_count = str(len(errors))
        except Exception:
            duration = "N/A"
            err_count = "N/A"
            
        rows.append({
            "Index": idx,
            "Workflow ID": w.workflow_id,
            "Status": w.status.upper(),
            "Duration": duration,
            "Errors Count": err_count,
            "Executed At": w.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df_workflows = pd.DataFrame(rows)
    render_table(df_workflows)

    st.divider()

    # 2. Inspect / Reload Past Workflow
    st.subheader("🕵️ Reload Past Run into Workspace")
    st.markdown("Select a workflow ID from the dropdown below to load its full state back into the active session.")
    
    workflow_ids_list = [w.workflow_id for w in workflows]
    selected_id = st.selectbox("Select Workflow ID to Inspect", options=workflow_ids_list)
    
    if st.button("🔄 Inspect and Reload Selected Run", width="stretch"):
        if load_past_workflow_result(selected_id):
            st.success(f"🎉 **Workflow ID `{selected_id}` successfully loaded!** Navigate to pages 2-7 to inspect its data.")
            st.toast("Past workflow loaded successfully!")
        else:
            st.error("Could not load the state checkpoint for this workflow ID (checkpoint file might be missing or corrupted).")


if __name__ == "__main__":
    main()
