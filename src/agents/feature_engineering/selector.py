"""Custom scikit-learn compatible feature selector transformer."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Fitable custom transformer implementing variance, correlation, and Mutual Information selectors."""

    def __init__(
        self,
        method: str = "correlation",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.85,
        k_best: int = 10,
        task_type: str = "classification"
    ) -> None:
        """Initialize FeatureSelector.

        Args:
            method: Selection method ('variance', 'correlation', 'mutual_info', 'none').
            variance_threshold: Minimum variance cutoff.
            correlation_threshold: Pearson correlation cutoff.
            k_best: Number of best features to select via Mutual Information.
            task_type: Task category ('classification' or 'regression').
        """
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.k_best = k_best
        self.task_type = task_type
        
        # Fitted states
        self.variance_selector_: Optional[VarianceThreshold] = None
        self.k_best_selector_: Optional[SelectKBest] = None
        self.columns_to_keep_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit feature selectors on X (and optionally y).

        Args:
            X: Input features DataFrame.
            y: Target label.

        Returns:
            FeatureSelector: Fitted instance.
        """
        logger.info(f"FeatureSelector: Fitting feature selector (method={self.method.upper()}) on X shape {X.shape}")
        
        # Start with all columns
        current_cols = list(X.columns)
        
        # Ensure only numeric inputs for selector methods
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            self.columns_to_keep_ = list(X.columns)
            return self

        # 1. Apply Variance Threshold
        if self.method in ["variance", "correlation", "mutual_info"]:
            self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
            self.variance_selector_.fit(numeric_df)
            support = self.variance_selector_.get_support()
            low_var_cols = [c for i, c in enumerate(numeric_df.columns) if not support[i]]
            current_cols = [c for c in current_cols if c not in low_var_cols]
            logger.info(f"  Variance Threshold removed {len(low_var_cols)} features: {low_var_cols}")

        # 2. Apply Correlation Filtering
        if self.method in ["correlation", "mutual_info"] and len(current_cols) > 1:
            numeric_subset = X[current_cols].select_dtypes(include=[np.number])
            if not numeric_subset.empty:
                corr = numeric_subset.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
                current_cols = [c for c in current_cols if c not in to_drop]
                logger.info(f"  Correlation Filter dropped {len(to_drop)} features: {to_drop}")

        # 3. Apply SelectKBest (Mutual Information)
        if self.method == "mutual_info" and y is not None:
            numeric_subset = X[current_cols].select_dtypes(include=[np.number])
            if not numeric_subset.empty:
                # Resolve target vector and handle NaNs
                y_clean = pd.Series(y).fillna(pd.Series(y).mode().iloc[0] if not pd.Series(y).mode().empty else 0)
                # Cap k_best to column counts
                k = min(self.k_best, numeric_subset.shape[1])
                
                score_func = mutual_info_classif if self.task_type == "classification" else mutual_info_regression
                
                self.k_best_selector_ = SelectKBest(score_func=score_func, k=k)
                self.k_best_selector_.fit(numeric_subset, y_clean)
                
                # Fetch importances
                scores = self.k_best_selector_.scores_
                for i, col in enumerate(numeric_subset.columns):
                    self.feature_importances_[col] = float(scores[i])
                
                # Get best features
                support = self.k_best_selector_.get_support()
                k_best_cols = [col for i, col in enumerate(numeric_subset.columns) if support[i]]
                
                # Combine non-numeric columns back
                non_numeric = [c for c in X.columns if c not in numeric_subset.columns]
                current_cols = k_best_cols + non_numeric
                logger.info(f"  Mutual Info K-Best selected {len(k_best_cols)} numeric features: {k_best_cols}")

        self.columns_to_keep_ = current_cols
        logger.info(f"FeatureSelector: Selected {len(self.columns_to_keep_)} / {X.shape[1]} features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter the input DataFrame to contain only selected features.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing only selected features.
        """
        existing_cols = [col for col in self.columns_to_keep_ if col in X.columns]
        return X[existing_cols]

    def _get_support_mask(self) -> np.ndarray:
        """Internal sklearn method supporting SelectorMixin.

        Returns:
            np.ndarray: Bool mask of selected features.
        """
        # Note: In standard sklearn pipelines, X column length during transform matches fit.
        # This mask indicates support if used in an sklearn selector sequence.
        pass
