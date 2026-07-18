"""Consistent dashboard notification helpers."""

import streamlit as st


def notify_result(result: object) -> None:
    """Display a concise success or degraded-workflow notification."""
    if getattr(result, "is_success", False):
        st.success("✅ **Workflow completed successfully!** All pipeline stages finished.")
    else:
        errors = getattr(getattr(result, "state", None), "errors", [])
        error_msg = f" ({errors[0]})" if errors else ""
        st.error(f"⚠️ **Workflow completed with errors.**{error_msg} Review status details.")


def show_success(message: str) -> None:
    """Render a premium success alert banner."""
    st.success(f"✅ {message}")


def show_error(message: str) -> None:
    """Render a premium error alert banner."""
    st.error(f"❌ {message}")


def show_warning(message: str) -> None:
    """Render a premium warning alert banner."""
    st.warning(f"⚠️ {message}")


def show_info(message: str) -> None:
    """Render a premium info alert banner."""
    st.info(f"ℹ️ {message}")

