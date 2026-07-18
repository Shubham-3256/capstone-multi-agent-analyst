"""Tuning manager running GridSearchCV and RandomizedSearchCV parameter sweeps."""

from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.core.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Performs parameter searches utilizing GridSearchCV or RandomizedSearchCV across candidates."""

    @staticmethod
    def get_parameter_grid(model_name: str, mode: str = "fast") -> Dict[str, list]:
        """Fetch pre-configured grids for common models.

        Args:
            model_name: Standardized algorithm candidate name.
            mode: Search grid depth ('fast' or 'full').

        Returns:
            Dict[str, list]: Grid parameters mapping.
        """
        key = model_name.lower().strip().replace(" ", "_")
        grids: Dict[str, Dict[str, list]] = {}

        if "logisticregression" in key:
            grids = {
                "fast": {"C": [0.1, 1.0, 10.0]},
                "full": {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l2"]}
            }
        elif "randomforest" in key:
            grids = {
                "fast": {"n_estimators": [50, 100], "max_depth": [5, 10]},
                "full": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 15], "min_samples_split": [2, 5]}
            }
        elif "decisiontree" in key:
            grids = {
                "fast": {"max_depth": [3, 5, 10]},
                "full": {"max_depth": [None, 3, 5, 10, 15], "min_samples_split": [2, 5, 10]}
            }
        elif "gradientboosting" in key:
            grids = {
                "fast": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "full": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 5]}
            }
        elif "ridge" in key or "lasso" in key or "elasticnet" in key:
            grids = {
                "fast": {"alpha": [0.1, 1.0]},
                "full": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            }
        elif "svc" in key or "svr" in key:
            grids = {
                "fast": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
                "full": {"C": [0.01, 0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "sigmoid"]}
            }
        elif "kneighbors" in key:
            grids = {
                "fast": {"n_neighbors": [3, 5, 7]},
                "full": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
            }

        # Fallback empty grid if name not mapped
        model_grids = grids.get(mode, {})
        if not model_grids and grids:
            # Fallback to alternative grid mode if mode unavailable
            model_grids = list(grids.values())[0]

        return model_grids

    @staticmethod
    def tune_model(
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str,
        mode: str = "fast",
        search_type: str = "random",
        n_iter: int = 5,
        cv_folds: int = 3,
        random_seed: int = 42,
        n_jobs: int = -1
    ) -> Any:
        """Perform tuning on a model. If grid is empty, fits and returns the model directly.

        Args:
            model_name: Name of the candidate.
            model: Estimator to tune.
            X_train: Input features.
            y_train: Target labels.
            task_type: Task category ('classification' or 'regression').
            mode: Search grid depth ('fast' or 'full').
            search_type: Search algorithm ('random' or 'grid').
            n_iter: Sample iterations if RandomizedSearch.
            cv_folds: Cross validation folds splits.
            random_seed: Random state seed.
            n_jobs: Parallel workers limit.

        Returns:
            Any: The fitted best model estimator.
        """
        param_grid = HyperparameterTuner.get_parameter_grid(model.__class__.__name__, mode)
        
        if not param_grid:
            logger.info(f"HyperparameterTuner: No parameter grid mapped for '{model_name}'. Fitting directly...")
            model.fit(X_train.fillna(0.0), y_train.fillna(0))
            return model

        logger.info(f"HyperparameterTuner: Tuning candidate '{model_name}' (method={search_type.upper()}, grid={param_grid})")
        scoring = "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"

        try:
            if search_type == "grid":
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv_folds,
                    n_jobs=n_jobs,
                    refit=True
                )
            else:
                # Limit n_iter to size of grid space to avoid warnings
                param_combinations = 1
                for v in param_grid.values():
                    param_combinations *= len(v)
                actual_n_iter = min(n_iter, param_combinations)
                
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=actual_n_iter,
                    scoring=scoring,
                    cv=cv_folds,
                    random_state=random_seed,
                    n_jobs=n_jobs,
                    refit=True
                )

            # Fit search
            search.fit(X_train.fillna(0.0), y_train.fillna(0))
            logger.info(f"  Tuning complete. Best params for '{model_name}': {search.best_params_}")
            return search.best_estimator_
        except Exception as e:
            logger.error(f"  Tuning failed for '{model_name}' due to error: {e}. Falling back to default fit...")
            model.fit(X_train.fillna(0.0), y_train.fillna(0))
            return model
