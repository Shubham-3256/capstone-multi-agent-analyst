"""Environment validation module for the Multi-Agent AI Data Analyst.

This module provides routines to verify python versions, required directories,
environment variable loads, third-party libraries, and formats a startup diagnostic panel.
"""

import importlib
import platform
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.paths import Paths
from src.core.settings import settings


def check_python_version() -> tuple[bool, str]:
    """Verify that Python version complies with requirement >= 3.11.

    Returns:
        Tuple[bool, str]: Success flag and detailed version status.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    current_version = f"{major}.{minor}.{sys.version_info.micro}"

    # Requirement: 3.11+
    if major < 3 or (major == 3 and minor < 11):
        return False, f"Python 3.11+ required. Detected: {current_version}"
    return True, f"Python version OK: {current_version}"


def check_required_directories() -> tuple[bool, list[str]]:
    """Validate that required workspace and application directories exist and are writable.

    Returns:
        Tuple[bool, List[str]]: Success flag and list of validation logs.
    """
    logs = []
    success = True

    dirs_to_check = {
        "Workspace": Paths.WORKSPACE_DIR,
        "Uploads": Paths.UPLOAD_DIR,
        "Cleaned": Paths.CLEANED_DIR,
        "Processed": Paths.PROCESSED_DIR,
        "Reports": Paths.REPORTS_DIR,
        "Plots": Paths.PLOTS_DIR,
        "Models": Paths.MODELS_DIR,
        "Logs": Paths.LOGS_DIR,
        "Data": Paths.DATA_DIR,
        "Artifacts": Paths.ARTIFACTS_DIR,
    }

    for name, path in dirs_to_check.items():
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logs.append(f"[OK] Directory created: {name} ({path})")
            except Exception as e:
                success = False
                logs.append(
                    f"[FAIL] Directory missing and uncreatable: {name} ({path}). Error: {e}"
                )
                continue

        # Check write permissions
        test_file = path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            logs.append(f"[OK] Directory is writable: {name} ({path})")
        except Exception as e:
            success = False
            logs.append(
                f"[FAIL] Directory is not writable: {name} ({path}). Error: {e}"
            )

    return success, logs


def check_env_variables() -> tuple[bool, list[str]]:
    """Evaluate key settings parameters, warning if defaults or placeholders are active.

    Returns:
        Tuple[bool, List[str]]: Success status and list of findings.
    """
    logs = []
    success = True

    # Check OpenAI Key presence
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY == "your_openai_api_key_here"
    ):
        # Keep success True for setup but warn developer
        logs.append(
            "[WARN] OPENAI_API_KEY is not configured or uses placeholder value."
        )
    else:
        logs.append("[OK] OPENAI_API_KEY is configured.")

    # Check Secret Key sanity (Warn if it's default in production)
    if settings.ENV == "production" and "super_secret" in settings.SECRET_KEY:
        logs.append(
            "[WARN] SECRET_KEY uses a default value in production mode! Change this immediately."
        )
    else:
        logs.append("[OK] SECRET_KEY is set.")

    return success, logs


def check_dependencies() -> tuple[bool, list[str]]:
    """Scan if required core packages can be imported successfully.

    Returns:
        Tuple[bool, List[str]]: Success flag and package logs.
    """
    core_packages = [
        "pandas",
        "numpy",
        "pydantic",
        "pydantic_settings",
        "fastapi",
        "rich",
        "openai",
        "langchain",
        "langgraph",
        "crewai",
    ]

    logs = []
    success = True

    for pkg in core_packages:
        try:
            importlib.import_module(pkg)
            logs.append(f"[OK] Import successful: {pkg}")
        except ImportError as e:
            success = False
            logs.append(f"[FAIL] Import failed: {pkg}. Error: {e}")

    return success, logs


def system_summary() -> None:
    """Print a professional, structured startup diagnostic report to the terminal.

    Consolidates Python version, system details, directory availability, and package states.
    """
    console = Console()

    # 1. Gather stats
    py_ok, py_msg = check_python_version()
    dirs_ok, dir_logs = check_required_directories()
    env_ok, env_logs = check_env_variables()
    deps_ok, dep_logs = check_dependencies()

    all_passed = py_ok and dirs_ok and env_ok and deps_ok

    # 2. Build Title Panel
    title = "[bold green]Multi-Agent AI Data Analyst - Core System Startup Diagnostic[/bold green]"
    if not all_passed:
        title = (
            "[bold red]Multi-Agent AI Data Analyst - System Startup Warnings[/bold red]"
        )

    # 3. Create Diagnostic Tables
    table_sys = Table(title="System Information", show_header=False, expand=True)
    table_sys.add_column("Property", style="cyan")
    table_sys.add_column("Value", style="magenta")
    table_sys.add_row(
        "Platform OS",
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    )
    table_sys.add_row("Python Executable", sys.executable)
    table_sys.add_row("Python Version Status", py_msg)
    table_sys.add_row("Environment Mode", settings.ENV.upper())

    # Build directory checklist
    dir_summary = (
        "All directories verified and writable." if dirs_ok else "\n".join(dir_logs)
    )

    # Build package checklist
    dep_summary = "All core dependencies verified." if deps_ok else "\n".join(dep_logs)

    # Build env checklist
    env_summary = (
        "Required environment configurations validated."
        if env_ok
        else "\n".join(env_logs)
    )

    # Compile Group for rendering nested objects in Rich Panel
    from rich.console import Group

    diagnostic_group = Group(
        table_sys,
        "",
        "[bold yellow]Directory Checks:[/bold yellow]",
        dir_summary,
        "",
        "[bold yellow]Environment Variables Checklist:[/bold yellow]",
        env_summary,
        "",
        "[bold yellow]Dependency Checks:[/bold yellow]",
        dep_summary,
    )

    console.print(
        Panel(
            diagnostic_group,
            title=title,
            expand=False,
            border_style="green" if all_passed else "yellow",
        )
    )
