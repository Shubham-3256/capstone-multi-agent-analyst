"""Training manager executing candidate models fits with exceptions capturing."""

import time
from typing import Any

import pandas as pd

from src.agents.machine_learning.models import TrainingResult
from src.core.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trains machine learning estimator candidates, recording durations and exceptions."""

    @staticmethod
    def train_model(
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_score: float = 0.0
    ) -> TrainingResult:
        """Fit a single candidate model.

        Args:
            model_name: Custom name key of candidate.
            model: Un-fitted scikit-learn compatible estimator.
            X_train: Input features.
            y_train: Target label values.
            cv_score: Cross validation score calculated separately.

        Returns:
            TrainingResult: Mapped execution logs and optimal fits.
        """
        logger.info(f"Trainer: Starting fit for candidate '{model_name}'...")
        start_time = time.time()

        try:
            # Clean dataframe inputs from infs/nans (safeguard)
            X_clean = X_train.fillna(0.0)
            y_clean = y_train.fillna(0.0)

            # Perform fit
            model.fit(X_clean, y_clean)

            duration = time.time() - start_time
            logger.info(f"Trainer: Successfully trained '{model_name}' in {round(duration, 4)}s")

            # Fetch hyperparameter details from fitted model
            params = {}
            if hasattr(model, "get_params"):
                params = model.get_params()

            return TrainingResult(
                model_name=model_name,
                best_params=params,
                cv_score=cv_score,
                duration_seconds=duration,
                error_message=None
            )
        except Exception as e:
            duration = time.time() - start_time
            err_msg = str(e)
            logger.error(f"Trainer: Exception training '{model_name}': {err_msg}")

            return TrainingResult(
                model_name=model_name,
                best_params={},
                cv_score=0.0,
                duration_seconds=duration,
                error_message=err_msg
            )
class ModelTimeoutException(Exception):
    """Custom exception raised when model training exceeds allowed timeout limits."""
    pass
