"""Path management module for the Multi-Agent AI Data Analyst.

This module provides a centralized manager for resolving, creating, and referencing
all static and dynamic directory paths required by the application.
"""

from pathlib import Path


class Paths:
    """Centralized manager for system and workspace directory paths.

    All paths are absolute and resolved using pathlib.
    """

    # --- Project Root ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # --- Application Source Folders ---
    SRC_DIR: Path = PROJECT_ROOT / "src"
    PROMPTS_DIR: Path = SRC_DIR / "prompts"
    CONFIG_DIR: Path = SRC_DIR / "config"

    # --- Dynamic Workspace Folders ---
    WORKSPACE_DIR: Path = PROJECT_ROOT / "workspace"
    UPLOAD_DIR: Path = WORKSPACE_DIR / "uploads"
    CLEANED_DIR: Path = WORKSPACE_DIR / "cleaned"
    PROCESSED_DIR: Path = WORKSPACE_DIR / "processed"
    REPORTS_DIR: Path = WORKSPACE_DIR / "reports"
    PLOTS_DIR: Path = WORKSPACE_DIR / "plots"
    MODELS_DIR: Path = WORKSPACE_DIR / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    DATA_DIR: Path = WORKSPACE_DIR / "data"
    ARTIFACTS_DIR: Path = WORKSPACE_DIR / "artifacts"
    CACHE_DIR: Path = WORKSPACE_DIR / "cache"
    REPORT_OUTPUT_DIR: Path = WORKSPACE_DIR / "reports"

    # --- Base Top-Level System Folders ---
    TESTS_DIR: Path = PROJECT_ROOT / "tests"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"
    SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

    # --- Aliases for maximum flexibility & backward/forward compatibility ---
    WORKSPACE: Path = WORKSPACE_DIR
    UPLOADS: Path = UPLOAD_DIR
    CLEANED: Path = CLEANED_DIR
    PROCESSED: Path = PROCESSED_DIR
    REPORTS: Path = REPORTS_DIR
    REPORT_DIR: Path = REPORTS_DIR
    PLOTS: Path = PLOTS_DIR
    PLOT_DIR: Path = PLOTS_DIR
    MODELS: Path = MODELS_DIR
    MODEL_DIR: Path = MODELS_DIR
    LOGS: Path = LOGS_DIR
    LOG_DIR: Path = LOGS_DIR
    PROMPTS: Path = PROMPTS_DIR
    CONFIG: Path = CONFIG_DIR
    TESTS: Path = TESTS_DIR
    DOCS: Path = DOCS_DIR
    SCRIPTS: Path = SCRIPTS_DIR
    DATA: Path = DATA_DIR
    ARTIFACTS: Path = ARTIFACTS_DIR


# --- Auto-initialization of required filesystem folders ---
_REQUIRED_DIRS = [
    Paths.WORKSPACE_DIR,
    Paths.UPLOAD_DIR,
    Paths.CLEANED_DIR,
    Paths.PROCESSED_DIR,
    Paths.REPORTS_DIR,
    Paths.REPORT_OUTPUT_DIR,
    Paths.PLOTS_DIR,
    Paths.MODELS_DIR,
    Paths.LOGS_DIR,
    Paths.DATA_DIR,
    Paths.ARTIFACTS_DIR,
    Paths.CACHE_DIR,
    Paths.PROMPTS_DIR,
    Paths.CONFIG_DIR,
    Paths.TESTS_DIR,
    Paths.DOCS_DIR,
    Paths.SCRIPTS_DIR,
]

for _directory in _REQUIRED_DIRS:
    _directory.mkdir(parents=True, exist_ok=True)
