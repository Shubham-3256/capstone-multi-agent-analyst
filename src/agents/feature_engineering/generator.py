"""Custom scikit-learn compatible feature generation transformer."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Fitable custom transformer generating interactions, polynomials, log, and datetime features."""

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        polynomial_degree: int = 2,
        interaction_only: bool = True,
        date_features: bool = True,
        log_transform: bool = True,
        skew_threshold: float = 1.5
    ) -> None:
        """Initialize FeatureGenerator.

        Args:
            numeric_columns: Numeric columns to use for numerical generation.
            datetime_columns: Datetime columns to extract components from.
            polynomial_degree: Polynomial degree to generate (e.g., 2).
            interaction_only: If True, do not generate squared values (only pairs).
            date_features: Extract year, month, day, weekday, quarter.
            log_transform: Apply log1p to highly skewed features.
            skew_threshold: Minimum skewness ratio to trigger log transform.
        """
        self.numeric_columns = numeric_columns
        self.datetime_columns = datetime_columns
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.date_features = date_features
        self.log_transform = log_transform
        self.skew_threshold = skew_threshold

        # Fitted parameters
        self.skewed_cols_: List[str] = []
        self.interaction_pairs_: List[tuple] = []
        self.fitted_numeric_cols_: List[str] = []
        self.fitted_datetime_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureGenerator":
        """Determine columns to transform, pairs to interact, and skewness metrics.

        Args:
            X: Input DataFrame.
            y: Target label.

        Returns:
            FeatureGenerator: Fitted instance.
        """
        # Determine numeric columns
        self.fitted_numeric_cols_ = self.numeric_columns if self.numeric_columns is not None else list(X.select_dtypes(include=[np.number]).columns)
        self.fitted_datetime_cols_ = self.datetime_columns if self.datetime_columns is not None else list(X.select_dtypes(include=[np.datetime64, "datetime64[ns]"]).columns)

        logger.info(f"FeatureGenerator: Fitting generation rules. Numerics={len(self.fitted_numeric_cols_)}, Datetimes={len(self.fitted_datetime_cols_)}")

        # 1. Identify highly skewed columns for log transformation
        if self.log_transform:
            for col in self.fitted_numeric_cols_:
                series = X[col].dropna()
                if not series.empty:
                    try:
                        skew_val = float(series.skew())
                        if abs(skew_val) > self.skew_threshold:
                            # Verify all values are non-negative for log1p
                            if (series >= 0.0).all():
                                self.skewed_cols_.append(col)
                                logger.info(f"  Column '{col}' skewness ({round(skew_val, 2)}) > threshold. Log transform enabled.")
                    except Exception:
                        pass

        # 2. Select columns to pair for interaction features
        # Limit interaction pairs to prevent feature explosion (max 15 pairs)
        if self.polynomial_degree >= 2 and len(self.fitted_numeric_cols_) > 1:
            num_cols = self.fitted_numeric_cols_[:6]  # Limit to top 6 columns for interactions
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    self.interaction_pairs_.append((num_cols[i], num_cols[j]))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature generation transformations to the input DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing generated features.
        """
        X_trans = X.copy()

        # 1. Apply log transformations
        for col in self.skewed_cols_:
            if col in X_trans.columns:
                log_col_name = f"log_{col}"
                X_trans[log_col_name] = np.log1p(X_trans[col].astype(float))

        # 2. Generate interaction features
        for col1, col2 in self.interaction_pairs_:
            if col1 in X_trans.columns and col2 in X_trans.columns:
                inter_col_name = f"{col1}_x_{col2}"
                X_trans[inter_col_name] = X_trans[col1].astype(float) * X_trans[col2].astype(float)

        # 3. Generate polynomial features (e.g. squared terms if interaction_only is False)
        if self.polynomial_degree >= 2 and not self.interaction_only:
            for col in self.fitted_numeric_cols_[:5]:  # Limit to top 5 columns
                if col in X_trans.columns:
                    sq_col_name = f"{col}_squared"
                    X_trans[sq_col_name] = X_trans[col].astype(float) ** 2

        # 4. Extract date components
        if self.date_features:
            for col in self.fitted_datetime_cols_:
                if col in X_trans.columns:
                    try:
                        date_series = pd.to_datetime(X_trans[col])
                        X_trans[f"{col}_year"] = date_series.dt.year.fillna(0).astype(int)
                        X_trans[f"{col}_month"] = date_series.dt.month.fillna(0).astype(int)
                        X_trans[f"{col}_day"] = date_series.dt.day.fillna(0).astype(int)
                        X_trans[f"{col}_weekday"] = date_series.dt.weekday.fillna(0).astype(int)
                        X_trans[f"{col}_quarter"] = date_series.dt.quarter.fillna(0).astype(int)
                    except Exception as e:
                        logger.warning(f"Failed to generate date features for column '{col}': {e}")

        return X_trans
