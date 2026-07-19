"""Explainability engine compiling model coefficients and permutation feature importances."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.agents.machine_learning.models import FeatureImportance
from src.core.logger import get_logger

logger = get_logger(__name__)


class Explainer:
    """Extracts feature importances, absolute coefficients, and permutation importance metrics."""

    @staticmethod
    def explain_model(
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> list[FeatureImportance]:
        """Generate ranked feature importance metrics for the model.

        Args:
            model: Fitted estimator model instance.
            X_val: Validation features.
            y_val: Validation target labels.
            task_type: Mapped task type ('classification' or 'regression').

        Returns:
            List[FeatureImportance]: Sorted list of feature importance metrics.
        """
        logger.info("Explainer: Generating feature importances...")
        columns = list(X_val.columns)
        importances: dict[str, float] = {}

        X_clean = X_val.fillna(0.0)
        y_clean = y_val.fillna(0)

        # 1. Native feature_importances_ check
        if hasattr(model, "feature_importances_"):
            logger.info("  Using native model 'feature_importances_' attribute.")
            native_importances = model.feature_importances_
            for i, col in enumerate(columns):
                importances[col] = float(native_importances[i])

        # 2. Linear model coefficients check
        elif hasattr(model, "coef_"):
            logger.info("  Using native model 'coef_' coefficients.")
            coefs = model.coef_
            # If multi-class classification, coef_ shape is (n_classes, n_features)
            if len(coefs.shape) > 1:
                coefs_abs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs_abs = np.abs(coefs)

            # Normalize values to sum to 1.0
            sum_coefs = np.sum(coefs_abs)
            if sum_coefs > 0:
                coefs_norm = coefs_abs / sum_coefs
            else:
                coefs_norm = coefs_abs

            for i, col in enumerate(columns):
                importances[col] = float(coefs_norm[i])

        # 3. Fallback: Permutation Importance
        else:
            logger.info("  Estimator lacks native importance profiles. Running permutation importance...")
            try:
                scoring = "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"
                result = permutation_importance(
                    model, X_clean, y_clean,
                    scoring=scoring,
                    n_repeats=3,
                    random_state=42,
                    n_jobs=-1
                )

                # Use raw mean values, taking absolute to show scale of importance
                p_importances = np.abs(result.importances_mean)
                sum_p = np.sum(p_importances)
                if sum_p > 0:
                    p_norm = p_importances / sum_p
                else:
                    p_norm = p_importances

                for i, col in enumerate(columns):
                    importances[col] = float(p_norm[i])
            except Exception as e:
                logger.error(f"  Permutation importance calculation failed: {e}")
                # Fallback flat weights if even permutation fails
                for col in columns:
                    importances[col] = 1.0 / len(columns)

        # Sort and map to Pydantic FeatureImportance
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        results = [
            FeatureImportance(column=col, importance=round(val, 4))
            for col, val in sorted_items
        ]

        logger.info(f"Explainer: Mapped {len(results)} feature importance rankings.")
        return results
