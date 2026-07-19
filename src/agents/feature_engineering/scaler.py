"""Custom scikit-learn compatible numerical features scaler transformer."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler as SKMaxAbsScaler,
)
from sklearn.preprocessing import (
    MinMaxScaler as SKMinMaxScaler,
)
from sklearn.preprocessing import (
    RobustScaler as SKRobustScaler,
)
from sklearn.preprocessing import (
    StandardScaler as SKStandardScaler,
)

from src.core.logger import get_logger

logger = get_logger(__name__)


class NumericalScaler(BaseEstimator, TransformerMixin):
    """Fitable custom transformer implementing auto-selected numerical scalers."""

    def __init__(
        self,
        columns: list[str] | None = None,
        outlier_threshold: float = 0.05,
        default_scaler: str = "auto"
    ) -> None:
        """Initialize NumericalScaler.

        Args:
            columns: List of numeric columns to scale. If None, auto-detected during fit.
            outlier_threshold: Outlier percentage threshold for RobustScaler.
            default_scaler: Scaler override method ('auto', 'standard', 'robust', 'minmax', 'maxabs', 'none').
        """
        self.columns = columns
        self.outlier_threshold = outlier_threshold
        self.default_scaler = default_scaler

        # Fitted states
        self.scalers_: dict[str, str] = {}
        self.scaler_instances_: dict[str, Any] = {}
        self.scaling_params_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericalScaler":
        """Evaluate numeric distributions and fit scalers.

        Args:
            X: Input features DataFrame.
            y: Optional target label.

        Returns:
            NumericalScaler: Fitted transformer instance.
        """
        cols_to_fit = self.columns if self.columns is not None else list(X.select_dtypes(include=[np.number]).columns)
        logger.info(f"NumericalScaler: Fitting scaling coefficients for columns: {cols_to_fit}")

        for col in cols_to_fit:
            if col not in X.columns:
                continue

            series = X[col].dropna()
            if series.empty:
                continue

            # 1. Decision logic for scaler selection
            if self.default_scaler != "auto":
                scaler_type = self.default_scaler
            else:
                # Check for outliers using IQR
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = ((series < lower) | (series > upper)).sum()
                    outlier_ratio = outliers / len(series)
                else:
                    outlier_ratio = 0.0

                # Check bounds
                min_val = float(series.min())
                max_val = float(series.max())

                # Select scaler
                if outlier_ratio > self.outlier_threshold:
                    scaler_type = "robust"
                elif min_val >= 0.0 and max_val <= 1.0:
                    scaler_type = "none"  # Already normalized
                elif min_val >= 0.0 and max_val <= 100.0:
                    scaler_type = "minmax"
                else:
                    scaler_type = "standard"

            self.scalers_[col] = scaler_type
            logger.info(f"  Column '{col}' (outlier_ratio={round(outlier_ratio, 4) if 'outlier_ratio' in locals() else 'N/A'}) -> scaler chosen: {scaler_type.upper()}")

            # 2. Instantiate and fit scaler
            arr = X[[col]].fillna(0.0)  # Safe fill for scikit-learn
            if scaler_type == "standard":
                scaler = SKStandardScaler()
                scaler.fit(arr)
                self.scaler_instances_[col] = scaler
                self.scaling_params_[col] = {
                    "mean": float(scaler.mean_[0]),
                    "scale": float(scaler.scale_[0])
                }
            elif scaler_type == "robust":
                scaler = SKRobustScaler()
                scaler.fit(arr)
                self.scaler_instances_[col] = scaler
                self.scaling_params_[col] = {
                    "center": float(scaler.center_[0]),
                    "scale": float(scaler.scale_[0])
                }
            elif scaler_type == "minmax":
                scaler = SKMinMaxScaler()
                scaler.fit(arr)
                self.scaler_instances_[col] = scaler
                self.scaling_params_[col] = {
                    "min": float(scaler.min_[0]),
                    "scale": float(scaler.scale_[0])
                }
            elif scaler_type == "maxabs":
                scaler = SKMaxAbsScaler()
                scaler.fit(arr)
                self.scaler_instances_[col] = scaler
                self.scaling_params_[col] = {
                    "max_abs": float(scaler.max_abs_[0])
                }
            else:
                self.scaling_params_[col] = {}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to scale numeric columns.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        X_trans = X.copy()

        for col, scaler_type in self.scalers_.items():
            if col not in X_trans.columns or scaler_type == "none":
                continue

            scaler = self.scaler_instances_[col]
            # Replace column values with scaled values
            X_trans[col] = scaler.transform(X_trans[[col]])

        return X_trans
