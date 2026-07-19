"""Unit tests for reusable dashboard UI components."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.cards import info_card, metric_card
from app.components.metrics import workflow_metrics
from app.components.notifications import notify_result
from app.components.tables import render_html_table, render_table


@patch("streamlit.metric")
def test_metric_card_renders(mock_st_metric):
    """Test that metric_card invokes Streamlit's native st.metric."""
    metric_card("Test Label", "100", "+5")
    mock_st_metric.assert_called_once_with("Test Label", "100", "+5")


@patch("streamlit.markdown")
def test_info_card_renders(mock_st_markdown):
    """Test that info_card renders custom HTML content containing labels."""
    info_card("Total Revenue", "$5,000", "Updated today", icon="💰")
    mock_st_markdown.assert_called_once()
    html_arg = mock_st_markdown.call_args[0][0]
    assert "Total Revenue" in html_arg
    assert "$5,000" in html_arg
    assert "💰" in html_arg


@patch("streamlit.success")
def test_notifications_success(mock_st_success):
    """Test notify_result success alerts."""
    mock_result = SimpleNamespace(is_success=True, state=None)
    notify_result(mock_result)
    mock_st_success.assert_called_once()


@patch("streamlit.error")
def test_notifications_error(mock_st_error):
    """Test notify_result error alerts."""
    mock_result = SimpleNamespace(
        is_success=False, state=SimpleNamespace(errors=["Ingestion failed"])
    )
    notify_result(mock_result)
    mock_st_error.assert_called_once()
    assert "Ingestion failed" in mock_st_error.call_args[0][0]


@patch("streamlit.dataframe")
def test_render_table_dataframe(mock_st_dataframe):
    """Test render_table with a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    render_table(df)
    mock_st_dataframe.assert_called_once_with(df, width="stretch", hide_index=True)


@patch("streamlit.markdown")
def test_render_html_table(mock_st_markdown):
    """Test render_html_table renders table rows and headers."""
    render_html_table(["A", "B"], [[1, 2], [3, 4]])
    mock_st_markdown.assert_called_once()
    html_arg = mock_st_markdown.call_args[0][0]
    assert "<th>A</th>" in html_arg
    assert "<td>1</td>" in html_arg
    assert "<td>3</td>" in html_arg


def test_workflow_metrics_calculation():
    """Test extraction of node execution count and timing sum."""
    mock_state = SimpleNamespace(
        execution_history=[
            SimpleNamespace(node_name="load_dataset", status="completed")
        ],
        timing={"load_dataset": 1.25},
    )
    mock_result = SimpleNamespace(state=mock_state)
    metrics = workflow_metrics(mock_result)
    assert metrics["nodes"] == 1.0
    assert metrics["seconds"] == 1.25
