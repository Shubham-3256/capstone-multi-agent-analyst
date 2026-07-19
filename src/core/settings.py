"""Configuration settings module for the Multi-Agent AI Data Analyst.

This module defines the configuration system using Pydantic BaseSettings. It
loads variables from the environment and validates them, grouping configurations
into application, OpenAI, database, workspace, logging, Streamlit, FastAPI,
models, and security parameters.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables and .env file."""

    # --- Application Settings ---
    APP_NAME: str = Field(
        default="capstone-multi-agent-analyst", description="Name of the application"
    )
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    ENV: str = Field(
        default="development",
        description="Deployment environment (development, production, testing)",
    )

    # --- OpenAI / LLM Settings ---
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API authentication key")
    GEMINI_API_KEY: str = Field(default="", description="Gemini API authentication key")
    ANTHROPIC_API_KEY: str = Field(
        default="", description="Anthropic API authentication key"
    )
    MODEL_NAME: str = Field(default="gpt-4o", description="Target LLM model name")
    TEMPERATURE: float = Field(
        default=0.0, description="Temperature parameter for LLM sampling"
    )
    MAX_TOKENS: int = Field(
        default=4096, description="Maximum tokens parameter for LLM response"
    )

    # --- Database Settings ---
    DATABASE_URL: str = Field(
        default="sqlite:///./workspace/data.db", description="Database connection URL"
    )

    # --- Workspace Settings ---
    WORKSPACE_DIR: str = Field(
        default="./workspace", description="Base workspace directory"
    )
    UPLOAD_DIR: str = Field(
        default="./workspace/uploads", description="Directory for uploaded datasets"
    )
    CLEANED_DIR: str = Field(
        default="./workspace/cleaned", description="Directory for cleaned datasets"
    )
    PROCESSED_DIR: str = Field(
        default="./workspace/processed", description="Directory for processed datasets"
    )
    PLOT_DIR: str = Field(
        default="./workspace/plots", description="Directory for generated plots"
    )
    REPORT_DIR: str = Field(
        default="./workspace/reports", description="Directory for generated reports"
    )
    REPORT_OUTPUT_DIR: str = Field(
        default="./workspace/reports",
        description="Alternative directory for generated reports",
    )
    MODEL_DIR: str = Field(
        default="./workspace/models", description="Directory for trained ML models"
    )
    LOG_DIR: str = Field(
        default="./workspace/logs", description="Directory for log files"
    )
    CACHE_DIR: str = Field(
        default="./workspace/cache",
        description="Directory for caching model and agent payloads",
    )

    # --- Logging Settings ---
    LOG_LEVEL: str = Field(
        default="INFO", description="Global application logging level"
    )

    # --- Streamlit Settings ---
    STREAMLIT_PORT: int = Field(
        default=8501, description="Local port for the Streamlit dashboard"
    )

    # --- FastAPI Settings ---
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI server binding host")
    API_PORT: int = Field(default=8000, description="FastAPI server binding port")

    # --- Models Configuration Settings ---
    RANDOM_SEED: int = Field(
        default=42, description="Global random state seed for reproducibility"
    )
    TEST_SIZE: float = Field(
        default=0.2, description="Default test split fraction for ML models"
    )

    # --- Security Settings ---
    SECRET_KEY: str = Field(
        default="capstone_super_secret_cryptographic_key_change_in_production_32bytes!",
        description="Cryptographic key for signing sessions and sensitive values",
    )
    ALLOWED_EXTENSIONS: set[str] = Field(
        default={"csv", "xlsx", "json", "parquet"},
        description="Supported dataset file extensions for ingestion",
    )
    MAX_CONTENT_LENGTH: int = Field(
        default=104857600,  # 100 MB
        description="Maximum allowed content upload size in bytes",
    )

    # Configuration source parameters
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("ENV")
    @classmethod
    def validate_env(cls, val: str) -> str:
        """Validate the environment state."""
        allowed = {"development", "production", "testing"}
        env_lower = val.lower()
        if env_lower not in allowed:
            raise ValueError(f"ENV must be one of {allowed}, got: {val}")
        return env_lower

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, val: str) -> str:
        """Validate log level options."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level_upper = val.upper()
        if level_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got: {val}")
        return level_upper

    @field_validator("TEMPERATURE")
    @classmethod
    def validate_temperature(cls, val: float) -> float:
        """Ensure LLM temperature falls within the standard boundary."""
        if not (0.0 <= val <= 2.0):
            raise ValueError(f"TEMPERATURE must be between 0.0 and 2.0, got: {val}")
        return val


# Instantiate singleton settings object
settings = Settings()
