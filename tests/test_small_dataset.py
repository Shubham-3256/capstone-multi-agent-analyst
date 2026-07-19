"""Unit and integration tests verifying modeling resiliency on extremely small datasets."""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.machine_learning.cross_validator import CrossValidator
from src.database.base import Base
from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph


def test_small_dataset_cross_validation_classification():
    """Verify that CrossValidator switches to LeaveOneOut CV when dataset size is less than folds."""
    # 4 rows, target classification task, cv_folds=5
    df = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0]}
    )
    y = pd.Series([0, 1, 0, 1])

    model = LogisticRegression()
    result = CrossValidator.evaluate(model, df, y, "classification", cv_folds=5)

    # Folds count should be switched to number of samples (4) due to Leave-One-Out
    assert len(result.fold_scores) == 4
    assert result.mean_score >= 0.0
    assert result.std_score >= 0.0


def test_small_dataset_cross_validation_regression():
    """Verify that CrossValidator switches to LeaveOneOut CV for regression on small inputs."""
    # 3 rows, target continuous task, cv_folds=5
    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
    y = pd.Series([1.5, 2.5, 3.5])

    model = LinearRegression()
    result = CrossValidator.evaluate(model, df, y, "regression", cv_folds=5)

    # Folds count should be switched to number of samples (3)
    assert len(result.fold_scores) == 3
    assert result.mean_score >= 0.0
    assert result.std_score >= 0.0


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_e2e_small_dataset_pipeline(clean_db):
    """Verify that the full multi-agent pipeline completes successfully on a small dataset (4 rows)."""
    # Create a small dataset CSV file inside workspace/temp
    from src.core.paths import Paths

    temp_dir = Paths.WORKSPACE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = temp_dir / "tiny_dataset.csv"
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "monthly_charges": [50.0, 100.0, 80.0, 120.0],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_csv(dataset_file, index=False)

    # Instantiate graph with local mock configurations (or sqlite DB in memory)
    # We will inject clean_db session directly by overriding _default_callbacks
    graph = WorkflowGraph(
        config=WorkflowConfig(checkpoint_mode="none", persist_execution=False)
    )
    graph.db = clean_db

    # Overwrite DB session in graph nodes
    graph.nodes.callbacks = graph._default_callbacks()

    # Execute graph run
    result = graph.run(str(dataset_file), target_column="target")

    assert result.is_success is True
    # Verify AutoML finished and selected a model
    assert result.state.ml_result is not None
    assert result.state.ml_result.best_model_name != ""

    # Verify report generated successfully
    assert result.state.report_result is not None
    assert result.state.report_result.is_success is True

    # Verify no missing heatmap was generated (since the dataset is complete)
    assert result.state.visualization_result is not None
    missing_chart = next(
        c
        for c in result.state.visualization_result.report.charts
        if c.chart_id == "missing_heatmap"
    )
    assert missing_chart.file_path == ""
    assert "No missing values detected" in missing_chart.caption.summary
