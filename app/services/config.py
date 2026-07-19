"""Shared dashboard paths and display configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
UPLOAD_DIR = PROJECT_ROOT / "workspace" / "uploads"
SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "parquet"}
REPORTS_DIR = PROJECT_ROOT / "workspace" / "reports"
PLOTS_DIR = PROJECT_ROOT / "workspace" / "plots"
