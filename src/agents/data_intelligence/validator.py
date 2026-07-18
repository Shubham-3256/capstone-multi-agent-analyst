"""Validation engine for executing structural and data quality checks on datasets."""

import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.core.constants import SUPPORTED_DATASET_EXTENSIONS
from src.agents.data_intelligence.models import ValidationIssue, ValidationReport

logger = get_logger(__name__)


class Validator:
    """Validator class to evaluate dataset dimensions, structures, types, and quality issues."""

    @staticmethod
    def validate(df: pd.DataFrame, file_path: Path, target_column: Optional[str] = None) -> ValidationReport:
        """Run structural and quality checks on a Pandas DataFrame dataset.

        Args:
            df: Input dataset to validate.
            file_path: Physical filepath of the source file.
            target_column: Optional name of target label column.

        Returns:
            ValidationReport: Compiled diagnostics report.
        """
        logger.info(f"Validator: Starting dataset validation for {file_path.name}")
        issues: list[ValidationIssue] = []
        is_valid = True

        rows, cols = df.shape

        # 1. Check extension compatibility
        ext = file_path.suffix.lower().strip(".")
        if ext not in SUPPORTED_DATASET_EXTENSIONS:
            is_valid = False
            issues.append(ValidationIssue(
                check_name="supported_extension",
                severity="error",
                message=f"Unsupported file format suffix: '{ext}'. Supported: {SUPPORTED_DATASET_EXTENSIONS}"
            ))

        # 2. Check empty dataset
        if rows == 0 or cols == 0:
            is_valid = False
            issues.append(ValidationIssue(
                check_name="empty_dataset",
                severity="error",
                message=f"Dataset is empty. Dimensions: {rows} rows, {cols} columns."
            ))
            # End early as further checks need data
            return ValidationReport(is_valid=is_valid, issues=issues, summary={"total_issues": len(issues), "errors": 1, "warnings": 0})

        # 3. Check duplicate column headers
        col_list = [str(c) for c in df.columns]
        if len(col_list) != len(set(col_list)):
            # This is a warning because pandas can load it by appending suffixes automatically
            issues.append(ValidationIssue(
                check_name="duplicate_columns",
                severity="warning",
                message=f"Duplicate column headers detected. Unique: {len(set(col_list))} vs Total: {len(col_list)}"
            ))

        # 4. Check duplicate rows
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            issues.append(ValidationIssue(
                check_name="duplicate_rows",
                severity="warning",
                message=f"Dataset contains {dup_count} duplicate row records."
            ))

        # 5. Target column existence
        if target_column:
            if target_column not in df.columns:
                is_valid = False
                issues.append(ValidationIssue(
                    check_name="target_existence",
                    severity="error",
                    column=target_column,
                    message=f"Specified target label column '{target_column}' is missing from dataset columns."
                ))

        # 6. Large dataset warning (e.g. rows > 100,000)
        if rows > 100000:
            issues.append(ValidationIssue(
                check_name="large_dataset",
                severity="warning",
                message=f"Large dataset warning: {rows} rows detected. Downstream modeling may require sampling."
            ))

        # Column-by-column checks
        numeric_count = 0
        categorical_count = 0
        boolean_count = 0
        datetime_count = 0

        invalid_col_pattern = re.compile(r"[^\w\-]")  # Non-alphanumeric/underscore/dash characters

        for col in df.columns:
            col_name = str(col)
            series = df[col]

            # Inferred column type classification
            if pd.api.types.is_bool_dtype(series):
                boolean_count += 1
            elif pd.api.types.is_numeric_dtype(series):
                numeric_count += 1
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_count += 1
            else:
                categorical_count += 1

            # 7. Invalid column name characters
            if invalid_col_pattern.search(col_name) or " " in col_name:
                issues.append(ValidationIssue(
                    check_name="invalid_column_name",
                    severity="warning",
                    column=col_name,
                    message=f"Column name '{col_name}' contains spaces or special characters. Normalize recommendation."
                ))

            # 8. Completely empty columns
            null_count = int(series.isnull().sum())
            if null_count == rows:
                issues.append(ValidationIssue(
                    check_name="empty_column",
                    severity="warning",
                    column=col_name,
                    message=f"Column '{col_name}' is completely empty (all null values)."
                ))
                continue

            # 9. High missing value warning (> 50% NaNs)
            null_pct = (null_count / rows) * 100.0
            if null_pct > 50.0:
                issues.append(ValidationIssue(
                    check_name="high_nan_percentage",
                    severity="warning",
                    column=col_name,
                    message=f"Column '{col_name}' contains {null_count} NaNs ({round(null_pct, 2)}%)."
                ))

            # 10. Constant columns (only 1 unique value)
            unique_count = int(series.nunique(dropna=True))
            if unique_count == 1:
                issues.append(ValidationIssue(
                    check_name="constant_column",
                    severity="warning",
                    column=col_name,
                    message=f"Column '{col_name}' contains a constant value (only 1 unique value)."
                ))

            # 11. Mixed datatypes check (in object/string columns)
            if series.dtype == "object":
                types = series.dropna().map(type).unique()
                if len(types) > 1:
                    issues.append(ValidationIssue(
                        check_name="mixed_datatypes",
                        severity="warning",
                        column=col_name,
                        message=f"Column '{col_name}' contains mixed data types: {[t.__name__ for t in types]}."
                    ))

            # 12. Whitespace-only string fields
            if series.dtype == "object" or str(series.dtype) == "string":
                try:
                    whitespaces = int(series.astype(str).str.match(r"^\s+$").sum())
                    if whitespaces > 0:
                        issues.append(ValidationIssue(
                            check_name="whitespace_only_strings",
                            severity="warning",
                            column=col_name,
                            message=f"Column '{col_name}' contains {whitespaces} whitespace-only string values."
                        ))
                except Exception:
                    pass

            # 13. Infinite values check (in numeric columns)
            if pd.api.types.is_numeric_dtype(series):
                inf_count = int(np.isinf(series).sum())
                if inf_count > 0:
                    issues.append(ValidationIssue(
                        check_name="infinite_values",
                        severity="warning",
                        column=col_name,
                        message=f"Column '{col_name}' contains {inf_count} infinite values."
                    ))

            # 14. High cardinality warning (categorical columns with unique > 50 and unique > 10% of rows)
            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
                if unique_count > 50 and (unique_count / rows) > 0.1:
                    issues.append(ValidationIssue(
                        check_name="high_cardinality",
                        severity="warning",
                        column=col_name,
                        message=f"Column '{col_name}' has high cardinality: {unique_count} unique values."
                    ))

        # Check total errors
        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")
        
        summary = {
            "total_issues": len(issues),
            "errors": errors,
            "warnings": warnings,
            "numeric_columns": numeric_count,
            "categorical_columns": categorical_count,
            "boolean_columns": boolean_count,
            "datetime_columns": datetime_count
        }

        # If any errors exist, classification is invalid
        if errors > 0:
            is_valid = False

        logger.info(f"Validator: Ingestion checked. Valid: {is_valid}. Errors: {errors}, Warnings: {warnings}")
        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            summary=summary
        )
