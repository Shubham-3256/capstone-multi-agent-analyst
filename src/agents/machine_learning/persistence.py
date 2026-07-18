"""Persistence manager to save fitted joblib models, metrics profiles, and leaderboards."""

import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd

from src.core.logger import get_logger
from src.core.paths import Paths
from src.agents.machine_learning.models import Leaderboard, FeatureImportance

logger = get_logger(__name__)


class Persistence:
    """Manages files serialization and directory setups for trained estimators and outputs."""

    @staticmethod
    def save_artifacts(
        best_model: Any,
        best_model_name: str,
        leaderboard: Leaderboard,
        metrics: Dict[str, float],
        feature_importances: List[FeatureImportance],
        metadata: Dict[str, Any],
        base_dir: Path = Paths.WORKSPACE_DIR / "artifacts"
    ) -> Dict[str, str]:
        """Save modeling models, leaderboards, and statistics details.

        Args:
            best_model: Fitted sklearn best estimator.
            best_model_name: Candidate algorithm name of best estimator.
            leaderboard: Mapped leaderboard entries.
            metrics: Best model scores dict.
            feature_importances: Best model feature importances list.
            metadata: Runs context.
            base_dir: Target output root directory.

        Returns:
            Dict[str, str]: Map of saved filenames to resolved filepaths.
        """
        logger.info(f"Persistence: Serializing artifacts to: {base_dir}")
        
        # Configure output subdirectories
        models_dir = base_dir / "models"
        metrics_dir = base_dir / "metrics"
        leaderboards_dir = base_dir / "leaderboards"
        
        for d in [models_dir, metrics_dir, leaderboards_dir]:
            d.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # 1. Save Best Model (.joblib)
        model_path = models_dir / f"best_model_{best_model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best_model, str(model_path))
        paths["best_model_path"] = str(model_path)
        logger.info(f"  Best model saved to: {model_path}")

        # 2. Save Leaderboard (JSON)
        leaderboard_path = leaderboards_dir / "leaderboard.json"
        with open(leaderboard_path, "w", encoding="utf-8") as f:
            json.dump(leaderboard.model_dump(), f, indent=4)
        paths["leaderboard_path"] = str(leaderboard_path)

        # 3. Save Metrics (JSON)
        metrics_path = metrics_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        paths["metrics_path"] = str(metrics_path)

        # 4. Save Feature Importance (CSV)
        importance_path = metrics_dir / "feature_importance.csv"
        importance_df = pd.DataFrame([
            {"column": item.column, "importance": item.importance}
            for item in feature_importances
        ])
        importance_df.to_csv(importance_path, index=False)
        paths["feature_importance_path"] = str(importance_path)

        # 5. Save Training Metadata (JSON)
        metadata_path = metrics_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        paths["metadata_path"] = str(metadata_path)

        logger.info("Persistence: All modeling artifacts saved successfully.")
        return paths

    @staticmethod
    def load_model(filepath: Path) -> Any:
        """Load a serialized fitted machine learning model from disk.

        Args:
            filepath: Location of the joblib model file.

        Returns:
            Any: The loaded model instance.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(str(filepath))
        logger.info(f"Persistence: Loaded fitted model from: {filepath}")
        return model
