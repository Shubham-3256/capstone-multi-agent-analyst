"""Unit tests for TaskDetector series categorization."""

import pandas as pd

from src.agents.machine_learning.task_detector import TaskDetector


def test_task_detector_classification():
    """Test classification task detection and binary flag checks."""
    y = pd.Series([0, 1, 0, 1, 0, 1])
    report = TaskDetector.detect_task(y)

    assert report.task_type == "classification"
    assert report.is_binary is True
    assert set(report.classes) == {0, 1}


def test_task_detector_regression():
    """Test regression task detection for float numerical labels."""
    y = pd.Series([1.2, 4.5, 3.1, 7.8, 9.4, 2.5, 8.1, 5.5, 6.3, 11.2, 10.4, 12.5])
    report = TaskDetector.detect_task(y)

    assert report.task_type == "regression"
    assert report.classes is None
    assert report.is_binary is None
