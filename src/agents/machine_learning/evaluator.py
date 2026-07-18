"""Metrics evaluation engine for classification and regression models."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from src.core.logger import get_logger
from src.agents.machine_learning.models import EvaluationReport

logger = get_logger(__name__)


class Evaluator:
    """Evaluates fitted estimator model metrics on test/validation segments."""

    @staticmethod
    def evaluate_model(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        is_binary: Optional[bool] = None
    ) -> EvaluationReport:
        """Calculate performance metrics on validation data.

        Args:
            model: Fitted estimator model instance.
            X_test: Input test features.
            y_test: Target test label values.
            task_type: Mapped task type ('classification' or 'regression').
            is_binary: True if classification target contains exactly 2 unique values.

        Returns:
            EvaluationReport: Mapped metrics profile.
        """
        logger.info("Evaluator: Computing validation metrics...")
        
        # Clean inputs
        X_clean = X_test.fillna(0.0)
        y_clean = y_test.fillna(0)

        # Make predictions
        y_pred = model.predict(X_clean)

        metrics: Dict[str, float] = {}
        conf_matrix: Optional[List[List[int]]] = None
        residuals_summary: Optional[Dict[str, float]] = None

        if task_type == "classification":
            # 1. Classification Metrics
            metrics["accuracy"] = float(accuracy_score(y_clean, y_pred))
            metrics["precision"] = float(precision_score(y_clean, y_pred, average="macro", zero_division=0))
            metrics["recall"] = float(recall_score(y_clean, y_pred, average="macro", zero_division=0))
            metrics["f1"] = float(f1_score(y_clean, y_pred, average="macro", zero_division=0))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_clean, y_pred))

            # Compute confusion matrix
            matrix = confusion_matrix(y_clean, y_pred)
            conf_matrix = [list(row) for row in matrix]

            # Compute ROC AUC if model supports predict_proba
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_clean)
                    if is_binary or y_prob.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_clean, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_clean, y_prob, multi_class="ovr", average="macro"))
                except Exception as e:
                    logger.warning(f"Failed to calculate ROC AUC: {e}")
                    metrics["roc_auc"] = 0.0
            else:
                metrics["roc_auc"] = 0.0

        else:
            # 2. Regression Metrics
            metrics["mae"] = float(mean_absolute_error(y_clean, y_pred))
            mse_val = float(mean_squared_error(y_clean, y_pred))
            metrics["mse"] = mse_val
            metrics["rmse"] = float(np.sqrt(mse_val))
            metrics["r2"] = float(r2_score(y_clean, y_pred))
            try:
                metrics["mape"] = float(mean_absolute_percentage_error(y_clean, y_pred))
            except Exception:
                metrics["mape"] = 0.0

            # Residuals Summary (y_test - y_pred)
            residuals = y_clean.values - y_pred
            residuals_summary = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }

        # Round metrics for readability
        metrics = {k: round(v, 4) for k, v in metrics.items()}
        logger.info(f"Evaluator: Performance metrics calculated: {metrics}")
        
        return EvaluationReport(
            metrics=metrics,
            confusion_matrix=conf_matrix,
            residuals_summary=residuals_summary
        )
