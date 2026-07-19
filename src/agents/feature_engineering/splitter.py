"""Splitting engine for dividing datasets into train, validation and test splits."""


import pandas as pd
from sklearn.model_selection import train_test_split

from src.agents.feature_engineering.models import SplitReport
from src.core.logger import get_logger

logger = get_logger(__name__)


class TrainValidationSplitter:
    """Handles train-test splitting under random, stratified, or chronological conditions."""

    @staticmethod
    def split(
        df: pd.DataFrame,
        target_column: str | None = None,
        strategy: str = "random",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
        """Split dataset DataFrame into Train, Validation, and Test segments.

        Args:
            df: Input clean DataFrame to split.
            target_column: Name of target column. Required for stratified split.
            strategy: Split strategy ('random', 'stratified', 'time_series').
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            random_seed: Seed for random generators.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]: Train, Val, Test, and SplitReport.
        """
        logger.info(f"TrainValidationSplitter: Splitting dataset (strategy={strategy.upper()}, ratios=[{train_ratio}, {val_ratio}, {test_ratio}])")

        # Verify ratios sum to 1.0 (approximately)
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

        # Safeguard for extremely small datasets to prevent empty or single-class splits
        total_rows = len(df)
        if total_rows < 10:
            logger.warning(
                f"TrainValidationSplitter: Dataset has only {total_rows} rows. "
                "Bypassing split ratios to prevent single-class or empty splits. Using full dataset for train, validation, and test splits."
            )
            report = SplitReport(
                train_shape=list(df.shape),
                val_shape=list(df.shape),
                test_shape=list(df.shape),
                strategy="bypass_small_dataset"
            )
            return df, df, df, report

        # Calculate temp split size
        temp_ratio = val_ratio + test_ratio  # e.g., 0.30

        if strategy == "time_series":
            # Chronological split without shuffling
            total_rows = len(df)
            train_boundary = int(total_rows * train_ratio)
            val_boundary = int(total_rows * (train_ratio + val_ratio))

            train_df = df.iloc[:train_boundary]
            val_df = df.iloc[train_boundary:val_boundary]
            test_df = df.iloc[val_boundary:]
        elif strategy == "stratified" and target_column and target_column in df.columns:
            # Stratified splits preserving class proportions
            y = df[target_column].fillna("missing")

            # Split train vs temp
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=random_seed,
                stratify=y
            )

            # Split temp into val and test
            val_split_ratio = val_ratio / temp_ratio
            y_temp = temp_df[target_column].fillna("missing")

            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1.0 - val_split_ratio),
                random_state=random_seed,
                stratify=y_temp
            )
        else:
            # Default Random Split
            # Split train vs temp
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=random_seed
            )

            # Split temp into val and test
            val_split_ratio = val_ratio / temp_ratio
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1.0 - val_split_ratio),
                random_state=random_seed
            )

        report = SplitReport(
            train_shape=list(train_df.shape),
            val_shape=list(val_df.shape),
            test_shape=list(test_df.shape),
            strategy=strategy
        )

        logger.info(f"TrainValidationSplitter: Split shapes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df, report
