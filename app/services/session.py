"""Streamlit session-state helpers that do not duplicate WorkflowState."""

from typing import Any

import streamlit as st


def initialize_session() -> None:
    """Set dashboard-local state defaults once per browser session."""
    st.session_state.setdefault("workflow_result", None)
    st.session_state.setdefault("uploaded_dataset_path", None)
    st.session_state.setdefault("workflow_events", [])
    st.session_state.setdefault("target_column", None)


def set_workflow_result(result: Any) -> None:
    """Store only the workflow result returned by the orchestration layer."""
    st.session_state.workflow_result = result


def get_workflow_result() -> Any | None:
    """Retrieve the latest workflow result, ensuring it is a fully validated Pydantic model."""
    raw = st.session_state.get("workflow_result")
    if raw is None:
        return None
    from src.orchestration.state import WorkflowResult

    try:
        if isinstance(raw, dict):
            return WorkflowResult.model_validate(raw)
        elif hasattr(raw, "model_dump"):
            # Ensure nested fields are fully validated and reconstructed
            return WorkflowResult.model_validate(raw.model_dump(mode="json"))
    except Exception:
        pass
    return raw


def set_uploaded_dataset_path(path: str | None) -> None:
    """Set the active dataset filepath in the session state."""
    st.session_state.uploaded_dataset_path = path


def get_uploaded_dataset_path() -> str | None:
    """Get the active dataset filepath from the session state."""
    return st.session_state.get("uploaded_dataset_path")


def set_workflow_events(events: list) -> None:
    """Set the list of workflow events."""
    st.session_state.workflow_events = events


def get_workflow_events() -> list:
    """Get the active workflow events list."""
    return st.session_state.get("workflow_events", [])


def clear_session() -> None:
    """Reset the session state variables to their defaults."""
    st.session_state.workflow_result = None
    st.session_state.uploaded_dataset_path = None
    st.session_state.workflow_events = []
    st.session_state.target_column = None
