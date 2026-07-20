"""Constants definition module for the Multi-Agent AI Data Analyst.

This module stores global read-only variables, model selections, design layouts,
and execution limits to prevent magic strings and values in downstream modules.
"""

from enum import StrEnum
from typing import Final

# --- Application Meta Information ---
APP_NAME: Final[str] = "capstone-multi-agent-analyst"
APP_VERSION: Final[str] = "0.1.0"


# --- Data Management & Validation Limits ---
SUPPORTED_DATASET_EXTENSIONS: Final[set[str]] = {"csv", "xlsx", "json", "parquet"}
MAX_UPLOAD_SIZE_BYTES: Final[int] = 104857600  # 100 Megabytes (100 * 1024 * 1024)
DEFAULT_ENCODING: Final[str] = "utf-8"


# --- Machine Learning Default Controls ---
DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_TEST_SIZE_FRACTION: Final[float] = 0.2


# --- Supported Large Language Models ---
class ModelName(StrEnum):
    """Supported OpenAI and fallback LLM models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"


# --- Visualizations Styling & Canvas Settings ---
PLOT_DEFAULT_WIDTH_INCHES: Final[int] = 10
PLOT_DEFAULT_HEIGHT_INCHES: Final[int] = 6
PLOT_FIGSIZE: Final[tuple[int, int]] = (10, 6)
PLOT_DPI_RESOLUTION: Final[int] = 300


# --- Application Color Palette (Premium HSL/Hex Visual Aesthetics) ---
COLOR_PRIMARY: Final[str] = "#1A365D"  # Deep Corporate Navy Blue
COLOR_SECONDARY: Final[str] = "#2B6CB0"  # Vibrant Tech Blue
COLOR_ACCENT: Final[str] = "#D69E2E"  # Warm Gold/Amber Accent
COLOR_SUCCESS: Final[str] = "#2F855A"  # Emerald Forest Green
COLOR_WARNING: Final[str] = "#C53030"  # Premium Muted Red
COLOR_BG_DARK: Final[str] = "#1A202C"  # Charcoal Dark Mode Slate
COLOR_BG_LIGHT: Final[str] = "#F7FAFC"  # Off-White background
COLOR_MUTED: Final[str] = "#718096"  # Cool Gray


# --- Network & Execution Timeouts (Seconds) ---
TIMEOUT_SHORT_SECONDS: Final[int] = 5
TIMEOUT_DEFAULT_SECONDS: Final[int] = 30
TIMEOUT_LLM_REQUEST_SECONDS: Final[int] = 120
TIMEOUT_LONG_SECONDS: Final[int] = 300


# --- Folder Names ---
FOLDER_UPLOADS: Final[str] = "uploads"
FOLDER_CLEANED: Final[str] = "cleaned"
FOLDER_PROCESSED: Final[str] = "processed"
FOLDER_PLOTS: Final[str] = "plots"
FOLDER_REPORTS: Final[str] = "reports"
FOLDER_MODELS: Final[str] = "models"
FOLDER_LOGS: Final[str] = "logs"
FOLDER_DATA: Final[str] = "data"
FOLDER_ARTIFACTS: Final[str] = "artifacts"
