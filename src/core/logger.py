"""Logging management module for the Multi-Agent AI Data Analyst.

Provides high-performance console logging utilizing the Rich package for terminal 
coloring and daily rotating log files for trace retention.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from rich.logging import RichHandler

from src.core.paths import Paths
from src.core.settings import settings

# Global flag to ensure logging handlers are only initialized once
_LOGGING_INITIALIZED = False


def setup_logging() -> None:
    """Configure system-wide logging with rich console output and timed rotating files.
    
    Loads configuration settings (like LOG_LEVEL) from settings singleton.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    # Determine logging level from settings
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Establish log formatter for output log files
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers: list[logging.Handler] = []

    # 1. Rich Console Handler (for colored and readable terminal logging)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_path=True,
        level=numeric_level
    )
    handlers.append(console_handler)

    # 2. Daily Rotating File Handler
    try:
        log_file_path = Paths.LOGS_DIR / "app.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file_path),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    except Exception as e:
        # Fail-safe to console if file system write access fails
        print(f"CRITICAL WARNING: Unable to initialize file logger at {Paths.LOGS_DIR}: {e}", file=sys.stderr)

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",  # Handled by Rich formatting in console
        handlers=handlers,
        force=True  # Clear any pre-existing configuration
    )

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Retrieve or initialize a logger instance with default system handlers.

    Args:
        name: Name of the active module requesting log services.

    Returns:
        logging.Logger: Configured logger object.
    """
    setup_logging()
    return logging.getLogger(name)
