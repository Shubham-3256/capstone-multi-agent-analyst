"""Model instantiation factory loading candidate estimators based on task type mappings."""

from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.core.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Instantiates estimators based on target task type and candidate name mappings."""

    @staticmethod
    def get_candidate_models(
        task_type: str,
        candidate_names: list[str],
        random_seed: int = 42,
        n_samples: int | None = None,
    ) -> dict[str, Any]:
        """Load classification or regression model candidates.

        Args:
            task_type: Modeling task type ('classification' or 'regression').
            candidate_names: List of model candidate keys.
            random_seed: Random state seed.
            n_samples: Optional number of training samples to sanitize default KNN settings.

        Returns:
            Dict[str, Any]: Mapping of candidate name to un-fitted estimator instance.
        """
        logger.info(
            f"ModelFactory: Loading model instances for task={task_type.upper()}"
        )
        models: dict[str, Any] = {}

        knn_neighbors = 5
        if n_samples is not None:
            # Leave-One-Out split training sample size is n_samples - 1.
            # KNN needs n_neighbors <= n_samples_fit.
            knn_neighbors = min(5, max(1, n_samples - 1))

        if task_type == "classification":
            # Map of classifiers
            registry = {
                "logistic_regression": lambda: LogisticRegression(
                    random_state=random_seed, max_iter=1000
                ),
                "decision_tree": lambda: DecisionTreeClassifier(
                    random_state=random_seed
                ),
                "random_forest": lambda: RandomForestClassifier(
                    random_state=random_seed
                ),
                "extra_trees": lambda: ExtraTreesClassifier(random_state=random_seed),
                "gradient_boosting": lambda: GradientBoostingClassifier(
                    random_state=random_seed
                ),
                "ada_boost": lambda: AdaBoostClassifier(random_state=random_seed),
                "k_neighbors": lambda: KNeighborsClassifier(n_neighbors=knn_neighbors),
                "gaussian_nb": lambda: GaussianNB(),
                "svc": lambda: SVC(random_state=random_seed, probability=True),
            }

            # Optional XGBoost, LightGBM, CatBoost loaders
            try:
                from xgboost import XGBClassifier

                registry["xgboost"] = lambda: XGBClassifier(
                    random_state=random_seed, eval_metric="logloss"
                )
            except ImportError:
                pass

            try:
                from lightgbm import LGBMClassifier

                registry["lightgbm"] = lambda: LGBMClassifier(
                    random_state=random_seed, verbose=-1
                )
            except ImportError:
                pass

        else:
            # Map of regressors
            registry = {
                "linear_regression": lambda: LinearRegression(),
                "ridge": lambda: Ridge(random_state=random_seed),
                "lasso": lambda: Lasso(random_state=random_seed),
                "elastic_net": lambda: ElasticNet(random_state=random_seed),
                "decision_tree": lambda: DecisionTreeRegressor(
                    random_state=random_seed
                ),
                "random_forest": lambda: RandomForestRegressor(
                    random_state=random_seed
                ),
                "extra_trees": lambda: ExtraTreesRegressor(random_state=random_seed),
                "gradient_boosting": lambda: GradientBoostingRegressor(
                    random_state=random_seed
                ),
                "ada_boost": lambda: AdaBoostRegressor(random_state=random_seed),
                "k_neighbors": lambda: KNeighborsRegressor(n_neighbors=knn_neighbors),
                "svr": lambda: SVR(),
            }

            try:
                from xgboost import XGBRegressor

                registry["xgboost"] = lambda: XGBRegressor(random_state=random_seed)
            except ImportError:
                pass

            try:
                from lightgbm import LGBMRegressor

                registry["lightgbm"] = lambda: LGBMRegressor(
                    random_state=random_seed, verbose=-1
                )
            except ImportError:
                pass

        # Instantiate candidates requested
        for name in candidate_names:
            key = name.lower().strip().replace(" ", "_")
            if key in registry:
                try:
                    models[name] = registry[key]()
                except Exception as e:
                    logger.warning(
                        f"ModelFactory: Failed to instantiate '{name}': {e}. Skipping."
                    )
            else:
                # Graceful warning for unsupported candidates
                logger.warning(
                    f"ModelFactory: Model name '{name}' is unsupported or missing dependencies. Skipping."
                )

        return models
