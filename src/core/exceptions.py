"""Exceptions module for the Multi-Agent AI Data Analyst.

Defines the custom hierarchical exception architecture used for explicit error 
propagation, handling, and debugging across all system layers.
"""

from typing import Any


class ProjectException(Exception):
    """Base exception class for all errors originating from the project.
    
    All custom errors in the application inherit from this class.
    """

    def __init__(self, message: str, code: str | None = None, details: Any | None = None) -> None:
        """Initialize ProjectException with messaging, diagnostic codes, and context.

        Args:
            message: Descriptive error message.
            code: Standard error code for downstream API identification.
            details: Additional debug payload or stack-trace bindings.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def __str__(self) -> str:
        """Represent the error details as a readable string."""
        base = f"[{self.code}] {self.message}" if self.code else self.message
        if self.details:
            base += f" | Details: {self.details}"
        return base


class ValidationException(ProjectException):
    """Raised when data validations or input structures fail checks."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ConfigurationException(ProjectException):
    """Raised when environment variables or settings parameters are invalid or missing."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class DatasetException(ProjectException):
    """Raised when loading, parsing, cleaning, or mapping datasets fails."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="DATASET_ERROR", details=details)


class DatabaseException(ProjectException):
    """Raised when relational database operations or schemas encounter errors."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="DATABASE_ERROR", details=details)


class ModelTrainingException(ProjectException):
    """Raised when training pipelines, parameters, or serialization fails."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="MODEL_TRAINING_ERROR", details=details)


class VisualizationException(ProjectException):
    """Raised when plot generation or formatting operations fail."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="VISUALIZATION_ERROR", details=details)


class ReportException(ProjectException):
    """Raised when HTML compilation or WeasyPrint PDF conversions fail."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="REPORT_ERROR", details=details)


class AgentException(ProjectException):
    """Raised when agent execution, tool invocations, or orchestration graphs fail."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, code="AGENT_ERROR", details=details)
