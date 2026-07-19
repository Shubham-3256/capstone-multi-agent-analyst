"""Performance tests profiling workflow scalability on larger dataset inputs."""

import time
import pytest
import pandas as pd
import numpy as np

from src.agents.feature_engineering.agent import FeatureEngineeringAgent
from src.agents.visualization.agent import VisualizationAgent
from src.database.base import Base
from sqlalchemy import create_engine


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_large_dataset_scalability_limit(clean_db):
    """Profile processing times of feature engineering and visualizations on a moderately large dataset."""
    # Generate ~5 MB synthetic dataset (50,000 rows, 10 columns)
    n_rows = 50000
    df = pd.DataFrame({
        "id": range(n_rows),
        "feat_num1": np.random.randn(n_rows),
        "feat_num2": np.random.randn(n_rows),
        "feat_num3": np.random.randn(n_rows),
        "feat_cat1": ["A", "B", "C", "D", "E"] * (n_rows // 5),
        "feat_cat2": (["High", "Medium", "Low"] * (n_rows // 3 + 1))[:n_rows],
        "constant_col": [9.9] * n_rows,
        "empty_col": [None] * n_rows,
        "satisfaction": np.random.uniform(0, 10, n_rows),
        "target": np.random.randint(0, 2, n_rows)
    })

    # 1. Measure Preprocessing & Feature Engineering time
    fe_agent = FeatureEngineeringAgent(clean_db)
    
    start_time = time.time()
    fe_res = fe_agent.run(df, target_column="target")
    fe_duration = time.time() - start_time
    
    # Assert feature engineering executes fast (under 10 seconds for 50k rows)
    assert fe_duration < 10.0
    assert fe_res.train_filepath != ""

    # 2. Measure Visualization Agent sweep time
    viz_agent = VisualizationAgent(clean_db)
    
    start_time = time.time()
    viz_res = viz_agent.run(df)
    viz_duration = time.time() - start_time

    # Assert visualization executes fast (under 8 seconds)
    assert viz_duration < 8.0
    assert viz_res.is_success is True
