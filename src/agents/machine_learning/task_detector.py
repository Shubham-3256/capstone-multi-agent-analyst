"""Task detection engine to classify target labels as classification or regression."""

import pandas as pd

from src.agents.machine_learning.models import TaskReport
from src.core.logger import get_logger

logger = get_logger(__name__)


class TaskDetector:
    """Automatically detects predictive modeling tasks based on target label properties."""

    @staticmethod
    def detect_task(y: pd.Series) -> TaskReport:
        """Analyze the target series values and infer the machine learning task category.

        Args:
            y: Target label variable series.

        Returns:
            TaskReport: Inferred task report.
        """
        logger.info("TaskDetector: Analyzing target variable properties...")

        # Drop missing values for analysis
        y_clean = y.dropna()
        unique_vals = list(y_clean.unique())
        unique_count = len(unique_vals)

        if unique_count <= 1:
            logger.warning(
                "Target variable has 1 or fewer unique values. Inferences may be unstable."
            )

        # Heuristic detection
        is_numeric = pd.api.types.is_numeric_dtype(y_clean)
        is_bool = pd.api.types.is_bool_dtype(y_clean)

        if is_numeric and not is_bool and unique_count > 10:
            task_type = "regression"
            classes = None
            is_binary = None
        else:
            task_type = "classification"
            classes = list(unique_vals)
            is_binary = unique_count == 2

        logger.info(
            f"TaskDetector: Inferred Task type = {task_type.upper()}. Unique classes count: {unique_count}"
        )
        return TaskReport(task_type=task_type, classes=classes, is_binary=is_binary)
