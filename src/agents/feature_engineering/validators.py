"""Validation engine implementing target, identifier, and temporal leakage checks."""

import numpy as np
import pandas as pd

from src.agents.feature_engineering.models import LeakageReport
from src.core.logger import get_logger

logger = get_logger(__name__)


class LeakageDetector:
    """Audits feature sets for data leakage issues, target flags, and duplicate columns."""

    @staticmethod
    def detect_leakage(
        df: pd.DataFrame, target_column: str, identifier_cols: list[str] | None = None
    ) -> LeakageReport:
        """Run leakage check sequences on the DataFrame.

        Args:
            df: Dataset DataFrame containing features and target column.
            target_column: Target prediction column.
            identifier_cols: List of identifier columns to warn about.

        Returns:
            LeakageReport: Compiled leakage audit report.
        """
        logger.info(
            f"LeakageDetector: Checking for data leaks targeting column '{target_column}'"
        )
        issues: list[dict[str, str]] = []
        has_leakage = False

        if target_column not in df.columns:
            logger.warning(
                f"Target column '{target_column}' not found in DataFrame for leakage checks."
            )
            return LeakageReport(has_leakage=False, leakage_issues=[])

        # 1. Target Leakage (features showing extremely high correlation with target)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column in numeric_df.columns:
            target_series = numeric_df[target_column].fillna(0)
            for col in numeric_df.columns:
                if col == target_column:
                    continue

                series = numeric_df[col].fillna(0)
                try:
                    corr_val = float(np.corrcoef(series, target_series)[0, 1])
                    if abs(corr_val) > 0.95:
                        has_leakage = True
                        issues.append(
                            {
                                "column": col,
                                "issue": f"Target Leakage: Extremely high Pearson correlation (R={round(corr_val, 4)}) with target '{target_column}'",
                            }
                        )
                except Exception:
                    pass

        # 2. Identifier Leakage (ID columns included as input features)
        ids = identifier_cols or []
        for col in df.columns:
            if col == target_column:
                continue

            col_name = str(col).lower()
            if col in ids or any(
                keyword in col_name
                for keyword in [
                    "customer_id",
                    "userid",
                    "user_id",
                    "session_id",
                    "transaction_id",
                    "index",
                ]
            ):
                has_leakage = True
                issues.append(
                    {
                        "column": col,
                        "issue": "Identifier Leakage: Column appears to be a unique ID or key. Including keys can lead to overfitting.",
                    }
                )

        # 3. Duplicate columns leakage
        checked = set()
        for col1 in df.columns:
            if col1 == target_column:
                continue
            for col2 in df.columns:
                if col1 != col2 and col2 != target_column and col2 not in checked:
                    try:
                        if df[col1].equals(df[col2]):
                            issues.append(
                                {
                                    "column": col1,
                                    "issue": f"Duplicate column: Column is identical to '{col2}'. Prune to avoid feature redundancy.",
                                }
                            )
                    except Exception:
                        pass
            checked.add(col1)

        logger.info(
            f"LeakageDetector: Check complete. Has Leakage: {has_leakage}. Found {len(issues)} issues."
        )
        return LeakageReport(has_leakage=has_leakage, leakage_issues=issues)
