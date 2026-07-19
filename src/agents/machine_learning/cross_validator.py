"""Cross-validation split validator executing KFold and StratifiedKFold runs."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score

from src.agents.machine_learning.models import CrossValidationResult
from src.core.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """Evaluates estimator stability across StratifiedKFold, KFold, or LeaveOneOut cross-validation splits."""

    @staticmethod
    def evaluate(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int = 5,
        random_seed: int = 42,
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

        n_samples = X_clean.shape[0]

        # 1. Resolve fold strategy and scorer
        if n_samples <= 1:
            logger.warning(
                "CrossValidator: Dataset contains 1 or 0 rows. Cross-validation is not possible."
            )
            return CrossValidationResult(
                fold_scores=[0.0], mean_score=0.0, std_score=0.0
            )

        if n_samples < cv_folds:
            logger.warning(
                f"CrossValidator: Dataset has only {n_samples} rows, which is less than cv_folds={cv_folds}. "
                "Automatically switching to Leave-One-Out (LOOCV) cross-validation."
            )
            cv = LeaveOneOut()
            # F1-macro on single-sample test fold returns NaN/0.0; use accuracy for classification under LOOCV
            scoring = (
                "accuracy"
                if task_type == "classification"
                else "neg_root_mean_squared_error"
            )
            actual_folds = n_samples
        else:
            actual_folds = cv_folds
            if task_type == "classification":
                # Check class representation to avoid StratifiedKFold errors
                class_counts = y_clean.value_counts()
                min_class_count = class_counts.min()
                if min_class_count < cv_folds:
                    logger.warning(
                        f"CrossValidator: The least populated class has only {min_class_count} members. "
                        f"Reducing classification cv splits to {min_class_count}."
                    )
                    if min_class_count <= 1:
                        logger.warning(
                            "CrossValidator: Minimum class count is 1. Switching to standard KFold."
                        )
                        cv = KFold(
                            n_splits=cv_folds, shuffle=True, random_state=random_seed
                        )
                    else:
                        cv = StratifiedKFold(
                            n_splits=min_class_count,
                            shuffle=True,
                            random_state=random_seed,
                        )
                        actual_folds = min_class_count
                else:
                    cv = StratifiedKFold(
                        n_splits=cv_folds, shuffle=True, random_state=random_seed
                    )
                scoring = "f1_macro"
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
                scoring = "neg_root_mean_squared_error"

        try:
            scores = cross_val_score(
                model, X_clean, y_clean, cv=cv, scoring=scoring, n_jobs=-1
            )

            # If negative RMSE, negate values to show true positive error metrics
            if scoring == "neg_root_mean_squared_error":
                scores = -scores

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            fold_scores = [float(s) for s in scores]

            logger.info(
                f"  CV Complete. Mean score: {round(mean_score, 4)} (Std: {round(std_score, 4)})"
            )
            return CrossValidationResult(
                fold_scores=fold_scores, mean_score=mean_score, std_score=std_score
            )
        except Exception as e:
            logger.error(f"  CV Execution failed: {e}")
            return CrossValidationResult(
                fold_scores=[0.0] * actual_folds, mean_score=0.0, std_score=0.0
            )
