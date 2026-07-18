"""Unit tests for Persistence artifact saves."""

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.agents.machine_learning.persistence import Persistence
from src.agents.machine_learning.models import Leaderboard, FeatureImportance


def test_persistence_save_and_load(tmp_path):
    """Test serializing model, metrics JSON, leaderboard entries, and feature CSV."""
    model = LogisticRegression()
    # Mock data to fit to enable loading back successfully
    import numpy as np
    model.fit(np.array([[1], [2]]), np.array([0, 1]))
    
    leaderboard = Leaderboard(entries=[])
    metrics = {"f1": 0.85, "accuracy": 0.88}
    importances = [FeatureImportance(column="age", importance=1.0)]
    metadata = {"info": "test"}
    
    paths = Persistence.save_artifacts(
        best_model=model,
        best_model_name="LogisticRegression",
        leaderboard=leaderboard,
        metrics=metrics,
        feature_importances=importances,
        metadata=metadata,
        base_dir=tmp_path
    )
    
    assert Path(paths["best_model_path"]).exists()
    assert Path(paths["leaderboard_path"]).exists()
    assert Path(paths["metrics_path"]).exists()
    assert Path(paths["feature_importance_path"]).exists()
    assert Path(paths["metadata_path"]).exists()
    
    # Assert loading works
    loaded_model = Persistence.load_model(Path(paths["best_model_path"]))
    assert isinstance(loaded_model, LogisticRegression)
