"""Integration tests verifying VisualizationAgent robustness, column filters, and missingness heatmap bypass."""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.visualization.agent import VisualizationAgent
from src.agents.visualization.dataset_visualizer import DatasetVisualizer
from src.database.base import Base


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_visualizations_filtering_and_bypass(clean_db):
    """Verify that identifiers/constants are filtered out and empty missingness heatmap is bypassed."""
    agent = VisualizationAgent(clean_db)

    # Dataset with 0 missing values, constant column, empty column, and id column
    df = pd.DataFrame(
        {
            "customer_id": [f"ID{i}" for i in range(5)],  # Identifier
            "constant_feat": [42] * 5,  # Constant
            "empty_feat": [0.0] * 5,  # Non-empty
            "age": [20, 30, 40, 50, 60],  # Valid Numeric
            "income": [50000, 60000, 70000, 80000, 90000],  # Valid Numeric
        }
    )

    result = agent.run(dataset_profile=df)

    assert result.is_success is True
    # The missingness heatmap should have file_path == "" and html_path == None since there are 0 missing cells
    missing_chart = next(
        c for c in result.report.charts if c.chart_id == "missing_heatmap"
    )
    assert missing_chart.file_path == ""
    assert missing_chart.html_path is None
    assert "No missing values detected" in missing_chart.caption.summary

    # Valid columns for correlation heatmap should only include valid numeric features (age, income)
    valid_cols = DatasetVisualizer.get_valid_columns(df)
    assert "age" in valid_cols
    assert "income" in valid_cols
    assert "customer_id" not in valid_cols
    assert "constant_feat" not in valid_cols
    assert "empty_feat" not in valid_cols

    # Verify that the correlation heatmap was successfully generated and saved
    corr_chart = next(
        c for c in result.report.charts if c.chart_id == "correlation_heatmap"
    )
    assert corr_chart.file_path != ""
    assert Path(corr_chart.file_path).exists()
