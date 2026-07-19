"""Unit tests for ContextBuilder consolidations."""

import pandas as pd

from src.agents.report_generation.context_builder import ContextBuilder


def test_context_builder_builds():
    """Test consolidating dataframes and strings outputs into ReportContext."""
    df = pd.DataFrame({"col": [1, 2]})
    context = ContextBuilder.build_context(
        dataset_profile=df,
        feature_result=None,
        ml_result=None,
        visualization_result=None,
        business_result=None,
    )

    assert "DataFrame" in context.dataset_profile_str
    assert "was not executed" in context.ml_summary_str
