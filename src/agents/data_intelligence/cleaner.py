"""Cleaning engine implementing missing values, duplicates, datatypes, and outliers logic."""

import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.agents.data_intelligence.models import CleaningAction, CleaningReport

logger = get_logger(__name__)


class Cleaner:
    """Cleaner class executing sanitization, missing value fills, and outlier adjustments on DataFrames."""

    @staticmethod
    def clean(
        df: pd.DataFrame,
        imputation_strategies: Optional[Dict[str, str]] = None,
        outlier_strategies: Optional[Dict[str, str]] = None,  # col -> strategy (e.g. 'iqr_cap', 'zscore_cap', 'drop')
        datatype_conversions: Optional[Dict[str, str]] = None,
        drop_empty_cols: bool = True
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Perform unified data cleaning actions on a Pandas DataFrame.

        Args:
            df: Input target DataFrame to clean.
            imputation_strategies: Dict mapping column names to nan-handling strategy 
                ('mean', 'median', 'mode', 'constant', 'drop').
            outlier_strategies: Dict mapping column names to outlier-handling strategy
                ('iqr_cap', 'zscore_cap', 'drop').
            datatype_conversions: Dict mapping column names to target data types
                ('int', 'float', 'bool', 'str', 'category', 'datetime').
            drop_empty_cols: If True, columns containing only nulls are dropped.

        Returns:
            Tuple[pd.DataFrame, CleaningReport]: Cleaned DataFrame and transformations report.
        """
        logger.info("Cleaner: Starting dataset cleaning process...")
        start_time = time.time()
        
        # Deep copy to prevent original dataframe mutations
        working_df = df.copy()
        initial_shape = list(working_df.shape)
        transformations: List[CleaningAction] = []

        # 1. Convert/Resolve duplicate column names
        cols = list(working_df.columns)
        seen_cols = {}
        dup_renamed = False
        for idx, col in enumerate(cols):
            col_str = str(col)
            if col_str in seen_cols:
                seen_cols[col_str] += 1
                new_col_name = f"{col_str}_{seen_cols[col_str]}"
                cols[idx] = new_col_name
                dup_renamed = True
            else:
                seen_cols[col_str] = 0
        if dup_renamed:
            working_df.columns = cols
            transformations.append(CleaningAction(
                action_type="deduplicate_columns",
                details=f"Renamed duplicate columns to enforce uniqueness. New column list: {cols}"
            ))

        # 2. Normalize column names (lowercase, remove spaces & special characters)
        old_cols = list(working_df.columns)
        normalized_cols = []
        for col in old_cols:
            col_str = str(col).strip().lower().replace(" ", "_")
            col_str = re.sub(r"[^\w\-]", "", col_str)  # Keep alphanumeric, underscores, hyphens
            normalized_cols.append(col_str)
        
        if old_cols != normalized_cols:
            working_df.columns = normalized_cols
            transformations.append(CleaningAction(
                action_type="normalize_column_names",
                details=f"Normalized column names to snake_case. Mapping: {dict(zip(old_cols, normalized_cols))}"
            ))

        # 3. Replace infinite values with NaNs
        numeric_cols = working_df.select_dtypes(include=[np.number]).columns
        inf_replaced_count = 0
        for col in numeric_cols:
            inf_mask = np.isinf(working_df[col])
            count = int(inf_mask.sum())
            if count > 0:
                working_df.loc[inf_mask, col] = np.nan
                inf_replaced_count += count
                transformations.append(CleaningAction(
                    column=col,
                    action_type="replace_inf",
                    details=f"Replaced {count} infinite values with NaN in column '{col}'"
                ))

        # 4. Trim whitespace on string columns
        string_cols = working_df.select_dtypes(include=["object", "string"]).columns
        for col in string_cols:
            try:
                # Apply strip to string cells
                working_df[col] = working_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                # Convert empty or whitespace-only strings to NaN
                empty_mask = working_df[col] == ""
                empty_count = int(empty_mask.sum())
                if empty_count > 0:
                    working_df.loc[empty_mask, col] = np.nan
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="trim_whitespace",
                        details=f"Trimmed spaces and replaced {empty_count} empty strings with NaN in '{col}'"
                    ))
            except Exception as e:
                logger.warning(f"Failed to trim whitespaces in column '{col}': {e}")

        # 5. Drop empty columns (where all values are null)
        if drop_empty_cols:
            empty_cols = [col for col in working_df.columns if working_df[col].isnull().all()]
            if empty_cols:
                working_df = working_df.drop(columns=empty_cols)
                for col in empty_cols:
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="drop_empty_column",
                        details=f"Dropped completely empty column '{col}'"
                    ))

        # 6. Remove duplicate rows
        dup_rows = int(working_df.duplicated().sum())
        if dup_rows > 0:
            working_df = working_df.drop_duplicates()
            transformations.append(CleaningAction(
                action_type="drop_duplicate_rows",
                details=f"Removed {dup_rows} duplicate rows. Shape updated to {working_df.shape}"
            ))

        # 7. Handle missing values
        imp_strategies = imputation_strategies or {}
        for col in working_df.columns:
            null_count = int(working_df[col].isnull().sum())
            if null_count == 0:
                continue

            strategy = imp_strategies.get(col)
            # Auto-infer default strategy if not specified
            if not strategy:
                if pd.api.types.is_numeric_dtype(working_df[col]):
                    strategy = "median"
                else:
                    strategy = "mode"

            try:
                if strategy == "drop":
                    working_df = working_df.dropna(subset=[col])
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="impute_missing",
                        details=f"Dropped {null_count} rows containing missing values in column '{col}'"
                    ))
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(working_df[col]):
                    mean_val = float(working_df[col].mean())
                    working_df[col] = working_df[col].fillna(mean_val)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="impute_missing",
                        details=f"Imputed {null_count} NaNs in column '{col}' with column mean ({round(mean_val, 4)})"
                    ))
                elif strategy == "median" and pd.api.types.is_numeric_dtype(working_df[col]):
                    median_val = float(working_df[col].median())
                    working_df[col] = working_df[col].fillna(median_val)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="impute_missing",
                        details=f"Imputed {null_count} NaNs in column '{col}' with column median ({round(median_val, 4)})"
                    ))
                elif strategy == "mode":
                    mode_series = working_df[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series.iloc[0]
                        working_df[col] = working_df[col].fillna(mode_val)
                        transformations.append(CleaningAction(
                            column=col,
                            action_type="impute_missing",
                            details=f"Imputed {null_count} NaNs in column '{col}' with column mode ('{mode_val}')"
                        ))
                elif strategy == "constant" or (isinstance(strategy, str) and strategy.startswith("const_")):
                    const_val = strategy.replace("const_", "") if strategy.startswith("const_") else "missing"
                    # Attempt type conversion for constant if column is numeric
                    if pd.api.types.is_numeric_dtype(working_df[col]):
                        try:
                            const_val = float(const_val) if "." in str(const_val) else int(const_val)
                        except ValueError:
                            pass
                    working_df[col] = working_df[col].fillna(const_val)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="impute_missing",
                        details=f"Imputed {null_count} NaNs in column '{col}' with constant value '{const_val}'"
                    ))
            except Exception as e:
                logger.error(f"Failed to impute column '{col}' using strategy '{strategy}': {e}")

        # 8. Outliers Capping
        out_strategies = outlier_strategies or {}
        for col, strategy in out_strategies.items():
            if col not in working_df.columns or not pd.api.types.is_numeric_dtype(working_df[col]):
                continue

            series = working_df[col]
            if strategy == "iqr_cap":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                # Cap values
                outliers_low = (series < lower).sum()
                outliers_high = (series > upper).sum()
                total_outliers = int(outliers_low + outliers_high)
                
                if total_outliers > 0:
                    working_df[col] = np.clip(series, lower, upper)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="handle_outliers",
                        details=f"Capped {total_outliers} IQR outliers in '{col}' to bounds: [{round(lower, 2)}, {round(upper, 2)}]"
                    ))
            elif strategy == "zscore_cap":
                mean_val = series.mean()
                std_val = series.std()
                if std_val > 0:
                    lower = mean_val - 3 * std_val
                    upper = mean_val + 3 * std_val
                    
                    outliers_low = (series < lower).sum()
                    outliers_high = (series > upper).sum()
                    total_outliers = int(outliers_low + outliers_high)
                    
                    if total_outliers > 0:
                        working_df[col] = np.clip(series, lower, upper)
                        transformations.append(CleaningAction(
                            column=col,
                            action_type="handle_outliers",
                            details=f"Capped {total_outliers} Z-score outliers in '{col}' to bounds: [{round(lower, 2)}, {round(upper, 2)}]"
                        ))
            elif strategy == "drop":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outliers_mask = (series < lower) | (series > upper)
                outliers_count = int(outliers_mask.sum())
                if outliers_count > 0:
                    working_df = working_df[~outliers_mask]
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="handle_outliers",
                        details=f"Dropped {outliers_count} outlier rows in column '{col}' based on IQR thresholds."
                    ))

        # 9. Convert datatypes
        conv_strategies = datatype_conversions or {}
        for col, target_type in conv_strategies.items():
            if col not in working_df.columns:
                continue
            
            try:
                if target_type == "datetime":
                    working_df[col] = pd.to_datetime(working_df[col], errors="coerce")
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to datetime"
                    ))
                elif target_type == "bool":
                    working_df[col] = working_df[col].astype(bool)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to boolean"
                    ))
                elif target_type == "int":
                    # Fill nans first if casting to pure int, or use Int64 (nullable int)
                    working_df[col] = pd.to_numeric(working_df[col], errors="coerce").round().astype("Int64")
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to nullable Int64"
                    ))
                elif target_type == "float":
                    working_df[col] = pd.to_numeric(working_df[col], errors="coerce").astype(float)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to float"
                    ))
                elif target_type == "category":
                    working_df[col] = working_df[col].astype("category")
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to category"
                    ))
                elif target_type == "str":
                    working_df[col] = working_df[col].astype(str)
                    transformations.append(CleaningAction(
                        column=col,
                        action_type="convert_datatype",
                        details=f"Converted column '{col}' datatype to string"
                    ))
            except Exception as e:
                logger.error(f"Failed to convert column '{col}' datatype to '{target_type}': {e}")

        # Finish summary parameters
        duration = time.time() - start_time
        final_shape = list(working_df.shape)
        
        report = CleaningReport(
            transformations=transformations,
            initial_shape=initial_shape,
            final_shape=final_shape,
            duration_seconds=duration
        )

        logger.info(f"Cleaner: Cleaning complete. Shape: {initial_shape} -> {final_shape} in {round(duration, 4)}s")
        return working_df, report
