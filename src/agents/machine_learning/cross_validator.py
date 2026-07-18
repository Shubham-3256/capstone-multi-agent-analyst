"""Cross-validation split validator executing KFold and StratifiedKFold runs."""

from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from src.core.logger import get_logger
from src.agents.machine_learning.models import CrossValidationResult

logger = get_logger(__name__)


class CrossValidator:
    """Evaluates estimator stability across StratifiedKFold or KFold cross-validation splits."""

    @staticmethod
    def evaluate(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int = 5,
        random_seed: int = 42
    ) -> CrossValidationResult:
        """Run cross-validation on X and y for a given model.

        Args:
            model: Estimator candidate to evaluate.
            X: Input features.
            y: Target label variable.
            task_type: Mapped task type ('classification' or 'regression').
            cv_folds: Number of folds.
            random_seed: Random state seed.

        Returns:
            CrossValidationResult: Scoring report.
        """
        logger.info(f"CrossValidator: Evaluating model using {cv_folds}-fold CV...")
        
        # Clean inputs
        X_clean = X.fillna(0.0)
        y_clean = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

        # 1. Resolve fold strategy and scorer
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            # Use f1_macro score for ranking classification
            scoring = "f1_macro"
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            # Use negative root mean squared error for regression
            scoring = "neg_root_mean_squared_error"

        try:
            scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring=scoring, n_jobs=-1)
            
            # If negative RMSE, negate values to show true positive error metrics
            if scoring == "neg_root_mean_squared_error":
                scores = -scores

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            fold_scores = [float(s) for s in scores]

            logger.info(f"  CV Complete. Mean score: {round(mean_score, 4)} (Std: {round(std_score, 4)})")
            return CrossValidationResult(
                fold_scores=fold_scores,
                mean_score=mean_score,
                std_score=std_score
            )
        except Exception as e:
            logger.error(f"  CV Execution failed: {e}")
            return CrossValidationResult(
                fold_scores=[0.0] * cv_folds,
                mean_score=0.0,
                std_score=0.0
            )
