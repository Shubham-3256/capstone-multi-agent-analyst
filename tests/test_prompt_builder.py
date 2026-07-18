"""Unit tests for prompt builder templates formatting."""

from src.agents.business_insights.executive_summary import ExecutiveSummaryBuilder
from src.agents.business_insights.data_quality import DataQualityBuilder


def test_prompt_builder_executive():
    """Test compiling executive summary prompts."""
    prompt = ExecutiveSummaryBuilder.build_prompt("mock_dataset", "mock_leaderboard")
    assert "mock_dataset" in prompt
    assert "mock_leaderboard" in prompt


def test_prompt_builder_quality():
    """Test compiling data quality prompts."""
    prompt = DataQualityBuilder.build_prompt("missingness heatmap info", "correlations matrix info")
    assert "missingness heatmap" in prompt
    assert "correlations matrix" in prompt
