"""Feature type detection engine for mapping structures and profiling column behaviors."""

import re
from typing import Any

import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureDetector:
    """Detects and categorizes column types and behaviors for downstream pipeline steps."""

    @staticmethod
    def detect(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
        """Perform comprehensive structure checks on the DataFrame features.

        Args:
            df: Dataset DataFrame to evaluate.
            target_column: Name of the target variable label.

        Returns:
            Dict[str, Any]: Mapping of column groups (types, constant, duplicates, etc.).
        """
        logger.info("FeatureDetector: Running column categorization audits...")
        rows, cols = df.shape

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        ordinal_cols: list[str] = []
        boolean_cols: list[str] = []
        datetime_cols: list[str] = []
        identifier_cols: list[str] = []
        text_cols: list[str] = []

        constant_cols: list[str] = []
        near_constant_cols: list[str] = []
        high_cardinality_cols: list[str] = []
        low_variance_cols: list[str] = []
        duplicate_features: dict[str, str] = {}  # col -> identical_to_col
        highly_correlated_features: list[dict[str, Any]] = []

        # Identifier search pattern
        id_pattern = re.compile(
            r"(_id|id_|key|code|uuid|index|hash|^id$)", re.IGNORECASE
        )

        # 1. Column classification loop
        for col in df.columns:
            col_name = str(col)
            if col_name == target_column:
                continue

            series = df[col]
            unique_count = int(series.nunique(dropna=True))
            null_count = int(series.isnull().sum())

            # A. Constant check
            if unique_count <= 1:
                constant_cols.append(col_name)
                continue

            # B. Near constant check (> 95% of rows have the same value)
            top_val_count = int(series.value_counts(dropna=True).max())
            if rows > 0 and (top_val_count / rows) > 0.95:
                near_constant_cols.append(col_name)

            # C. Inferred datatype categories
            if pd.api.types.is_bool_dtype(series):
                boolean_cols.append(col_name)
            elif id_pattern.search(col_name) or (
                unique_count == rows and series.dtype == "object"
            ):
                identifier_cols.append(col_name)
            elif pd.api.types.is_datetime64_any_dtype(series) or str(
                series.dtype
            ).startswith("datetime"):
                datetime_cols.append(col_name)
            elif pd.api.types.is_numeric_dtype(series):
                # Check low variance
                var_val = float(series.var(ddof=1)) if len(series.dropna()) > 1 else 0.0
                if var_val < 0.01:
                    low_variance_cols.append(col_name)

                # Check if it could be ordinal rating (integers and unique count between 2 and 10)
                if (
                    pd.api.types.is_integer_dtype(series)
                    and 2 < unique_count <= 10
                    and any(
                        keyword in col_name.lower()
                        for keyword in [
                            "rating",
                            "grade",
                            "level",
                            "scale",
                            "rank",
                            "score",
                        ]
                    )
                ):
                    ordinal_cols.append(col_name)
                else:
                    numeric_cols.append(col_name)
            else:
                # String/Object Column
                # Check if long string text (average word count > 3)
                sample_strings = series.dropna().astype(str).head(50)
                word_counts = sample_strings.str.split().str.len()
                if not word_counts.empty and float(word_counts.mean()) > 3.0:
                    text_cols.append(col_name)
                else:
                    # Check high cardinality Categorical limit
                    if unique_count > 50 and (unique_count / rows) > 0.1:
                        high_cardinality_cols.append(col_name)

                    # Heuristics for Ordinal text (e.g. low/med/high)
                    sample_set = {
                        str(val).lower().strip() for val in sample_strings.unique()
                    }
                    if sample_set.intersection(
                        {
                            "low",
                            "medium",
                            "high",
                            "poor",
                            "fair",
                            "good",
                            "excellent",
                            "small",
                            "large",
                        }
                    ):
                        ordinal_cols.append(col_name)
                    else:
                        categorical_cols.append(col_name)

        # 2. Duplicate features check
        checked_cols: list[str] = []
        for col in df.columns:
            col_name = str(col)
            if col_name == target_column:
                continue

            for other_col in checked_cols:
                # Compare Series values directly
                if df[col_name].equals(df[other_col]):
                    duplicate_features[col_name] = other_col
                    break
            checked_cols.append(col_name)

        # 3. Correlation checks (Numeric variables correlation > 0.85)
        all_numeric = [
            col
            for col in numeric_cols + ordinal_cols
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(all_numeric) > 1:
            corr = df[all_numeric].corr().fillna(0.0)
            seen_pairs = set()
            for c1 in corr.columns:
                for c2 in corr.columns:
                    if c1 != c2 and (c2, c1) not in seen_pairs:
                        val = float(corr.at[c1, c2])
                        if abs(val) > 0.85:
                            highly_correlated_features.append(
                                {"feature1": c1, "feature2": c2, "correlation": val}
                            )
                            seen_pairs.add((c1, c2))

        # Build feature types mapping dictionary
        feature_types = {}
        for col in numeric_cols:
            feature_types[col] = "numeric"
        for col in categorical_cols:
            feature_types[col] = "categorical"
        for col in ordinal_cols:
            feature_types[col] = "ordinal"
        for col in boolean_cols:
            feature_types[col] = "boolean"
        for col in datetime_cols:
            feature_types[col] = "datetime"
        for col in identifier_cols:
            feature_types[col] = "identifier"
        for col in text_cols:
            feature_types[col] = "text"

        report = {
            "feature_types": feature_types,
            "constant_columns": constant_cols,
            "near_constant_columns": near_constant_cols,
            "high_cardinality_columns": high_cardinality_cols,
            "low_variance_columns": low_variance_cols,
            "duplicate_features": duplicate_features,
            "highly_correlated_features": highly_correlated_features,
        }

        logger.info(
            f"FeatureDetector: Detection complete. Found: Numeric={len(numeric_cols)}, Categorical={len(categorical_cols)}, Ordinal={len(ordinal_cols)}, Identifiers={len(identifier_cols)}"
        )
        return report
