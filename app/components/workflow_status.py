"""Workflow status and execution timeline renderer."""

from typing import Any
import streamlit as st


class LiveProgressViewer:
    """Callback listener that updates a single Streamlit container in real-time."""

    def __init__(self) -> None:
        self.placeholder = st.empty()
        self.events = []

    def __call__(self, event: Any) -> None:
        """Process a workflow event and re-render the progress timeline."""
        self.events.append(event)
        node_title = event.node_name.replace("_", " ").title() if event.node_name else ""
        
        with self.placeholder.container():
            status_text = "Processing..."
            if event.event_type == "workflow_started":
                status_text = "Workflow Initializing..."
            elif event.event_type == "node_started":
                status_text = f"Executing Stage: {node_title}"
            elif event.event_type == "node_completed":
                status_text = f"Finished Stage: {node_title}"
            elif event.event_type == "workflow_completed":
                status_text = "Workflow Execution Finished"

            state_str = "running"
            if event.event_type == "workflow_completed":
                state_str = "complete" if event.payload.get("success") else "error"

            with st.status(f"⚙️ **{status_text}**", expanded=True, state=state_str) as status_box:
                for ev in self.events:
                    ev_node = ev.node_name.replace("_", " ").title() if ev.node_name else ""
                    if ev.event_type == "node_started":
                        st.markdown(f"⏳ **Running** `{ev_node}`...")
                    elif ev.event_type == "node_completed":
                        st.markdown(f"✅ **Completed** `{ev_node}`")
                    elif ev.event_type == "node_failed":
                        st.markdown(f"❌ **Failed** `{ev_node}`: {ev.payload.get('error', 'Unknown Error')}")
                
                if event.event_type == "workflow_completed":
                    if event.payload.get("success"):
                        status_box.update(label="🎉 **Workflow Completed Successfully!**", state="complete")
                    else:
                        status_box.update(label="⚠️ **Workflow Completed with Errors.**", state="error")


def render_workflow_status(result: Any) -> None:
    """Render completed, skipped, and failed nodes from WorkflowResult history."""
    if result is None:
        st.info("Upload a dataset to start a workflow.")
        return

    st.subheader("Workflow Execution Audit Timeline")
    
    all_possible_nodes = [
        "load_dataset",
        "data_intelligence",
        "feature_engineering",
        "machine_learning",
        "visualization",
        "business_insights",
        "report_generation"
    ]
    
    state = getattr(result, "state", None)
    history = getattr(state, "execution_history", []) if state else []
    executed_nodes = {item.node_name: item for item in history}
    
    # Render nodes in order
    for idx, node in enumerate(all_possible_nodes, 1):
        node_title = f"{idx}. {node.replace('_', ' ').title()}"
        if node in executed_nodes:
            item = executed_nodes[node]
            if item.status == "completed":
                st.success(f"✅ **{node_title}** — Completed in {item.duration_seconds:.2f}s (Attempts: {item.retries + 1})")
            else:
                st.error(f"❌ **{node_title}** — Failed: `{item.error}`")
        else:
            # Check if this node was skipped
            # A node is skipped if data_intelligence succeeded, but target_column is not set (skipping feature_engineering and machine_learning)
            target_col = getattr(getattr(state, "metadata", None), "target_column", None)
            is_ml_or_fe = node in ["feature_engineering", "machine_learning"]
            
            if is_ml_or_fe and not target_col:
                st.warning(f"⏭️ **{node_title}** — Skipped (No target column selected)")
            else:
                st.markdown(f"⬜ **{node_title}** — Not run")

    if state and state.warnings:
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for warning in state.warnings:
                st.warning(warning)
                
    if state and state.errors:
        with st.expander("❌ Pipeline Errors", expanded=True):
            for error in state.errors:
                st.error(error)

