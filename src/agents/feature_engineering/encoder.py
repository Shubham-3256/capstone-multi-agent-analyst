"""Custom scikit-learn compatible categorical feature encoder transformer."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder
from sklearn.preprocessing import OrdinalEncoder as SKOrdinalEncoder

from src.core.logger import get_logger

logger = get_logger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Fitable custom transformer implementing OneHot, Ordinal, and Frequency encoding strategies."""

    def __init__(
        self,
        columns: list[str] | None = None,
        low_cardinality_threshold: int = 10,
        medium_cardinality_threshold: int = 25,
        default_strategy: str = "auto"
    ) -> None:
        """Initialize CategoricalEncoder.

        Args:
            columns: List of columns to encode. If None, auto-detected during fit.
            low_cardinality_threshold: Cardinality limit for onehot strategy.
            medium_cardinality_threshold: Cardinality limit for ordinal strategy.
            default_strategy: Encoder override method ('auto', 'onehot', 'ordinal', 'frequency').
        """
        self.columns = columns
        self.low_cardinality_threshold = low_cardinality_threshold
        self.medium_cardinality_threshold = medium_cardinality_threshold
        self.default_strategy = default_strategy

        # Fitted parameters stored for serialization
        self.strategies_: dict[str, str] = {}
        self.onehot_transformers_: dict[str, SKOneHotEncoder] = {}
        self.ordinal_transformers_: dict[str, SKOrdinalEncoder] = {}
        self.frequency_mappings_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalEncoder":
        """Fit encoders for each categorical column based on cardinality metrics.

        Args:
            X: Input features DataFrame.
            y: Optional target label.

        Returns:
            CategoricalEncoder: Fitted transformer instance.
        """
        cols_to_fit = self.columns if self.columns is not None else list(X.select_dtypes(include=["object", "category"]).columns)
        logger.info(f"CategoricalEncoder: Fitting encoding mappings for columns: {cols_to_fit}")

        for col in cols_to_fit:
            if col not in X.columns:
                continue

            series = X[col].astype(str)
            unique_count = series.nunique()

            # 1. Resolve strategy
            if self.default_strategy != "auto":
                strategy = self.default_strategy
            else:
                if unique_count <= self.low_cardinality_threshold:
                    strategy = "onehot"
                elif unique_count <= self.medium_cardinality_threshold:
                    strategy = "ordinal"
                else:
                    strategy = "frequency"

            self.strategies_[col] = strategy
            logger.info(f"  Column '{col}' (uniques={unique_count}) -> encoded using strategy: {strategy.upper()}")

            # 2. Fit selected strategy
            if strategy == "onehot":
                # Reshape for sklearn
                arr = series.to_numpy().reshape(-1, 1)
                encoder = SKOneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoder.fit(arr)
                self.onehot_transformers_[col] = encoder
            elif strategy == "ordinal":
                arr = series.to_numpy().reshape(-1, 1)
                encoder = SKOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                encoder.fit(arr)
                self.ordinal_transformers_[col] = encoder
            elif strategy == "frequency":
                # Compute ratios
                freqs = series.value_counts(normalize=True).to_dict()
                self.frequency_mappings_[col] = freqs

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encodings to transform the input DataFrame.

        Args:
            X: Input DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame containing numeric encoded columns.
        """
        X_trans = X.copy()

        for col, strategy in self.strategies_.items():
            if col not in X_trans.columns:
                continue

            series = X_trans[col].astype(str)

            if strategy == "onehot":
                encoder = self.onehot_transformers_[col]
                arr = series.to_numpy().reshape(-1, 1)
                encoded_arr = encoder.transform(arr)

                # Create column headers
                categories = encoder.categories_[0]
                new_cols = [f"{col}_{str(cat).strip().lower().replace(' ', '_')}" for cat in categories]

                # Build temporary DataFrame
                encoded_df = pd.DataFrame(encoded_arr, columns=new_cols, index=X_trans.index)
                X_trans = pd.concat([X_trans.drop(columns=[col]), encoded_df], axis=1)
            elif strategy == "ordinal":
                encoder = self.ordinal_transformers_[col]
                arr = series.to_numpy().reshape(-1, 1)
                encoded_arr = encoder.transform(arr)

                # Replace column
                X_trans[col] = encoded_arr
            elif strategy == "frequency":
                mapping = self.frequency_mappings_[col]
                # Map value ratio weights, defaulting unknown values to 0.0
                X_trans[col] = series.map(mapping).fillna(0.0)

        return X_trans
