"""File utilities for hashing, compression, temp space, and extension validation."""

import gzip
import shutil
import tempfile
from pathlib import Path

from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.security import validate_extension

logger = get_logger(__name__)


def is_allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    """Validate if a filename has an allowed extension wrapper.

    Delegs to core.security.validate_extension for checks.

    Args:
        filename: String name of the target file.
        allowed_extensions: Set of supported extensions.

    Returns:
        bool: True if valid.
    """
    return validate_extension(filename, allowed_extensions)


def calculate_file_sha256(filepath: Path) -> str:
    """Read a file in byte blocks and compute its SHA-256 hash digest.

    Args:
        filepath: Target file path on disk.

    Returns:
        str: 64-character hex checksum.
    """
    logger.debug(f"Calculating SHA-256 checksum for: {filepath}")
    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(f"File not found for hashing: {filepath}")

    sha = hashlib_sha256 = hashlib_helper = None
    import hashlib

    h = hashlib.sha256()

    # Read in 64kb chunks for high performance on large files
    try:
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        raise


def compress_to_gzip(src: Path, dest: Path) -> Path:
    """Compress a file using GZIP compression.

    Args:
        src: Source uncompressed file path.
        dest: Destination target file path (should end in .gz).

    Returns:
        Path: Target compressed path.
    """
    logger.info(f"GZIP compressing: {src} -> {dest}")
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Compression completed: {dest}")
        return dest
    except Exception as e:
        logger.error(f"Failed to compress file: {e}")
        raise OSError(f"Compression error: {e}") from e


def decompress_from_gzip(src: Path, dest: Path) -> Path:
    """Decompress a GZIP file.

    Args:
        src: Source compressed path (ends in .gz).
        dest: Target decompressed path.

    Returns:
        Path: Target path.
    """
    logger.info(f"GZIP decompressing: {src} -> {dest}")
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(src, "rb") as f_in:
            with open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Decompression completed: {dest}")
        return dest
    except Exception as e:
        logger.error(f"Failed to decompress file: {e}")
        raise OSError(f"Decompression error: {e}") from e


def create_workspace_temp_dir(prefix: str = "temp_") -> Path:
    """Create a temporary directory located inside the safe workspace boundaries.

    Args:
        prefix: Directory name prefix.

    Returns:
        Path: Path to the newly created temporary directory.
    """
    temp_root = Paths.WORKSPACE_DIR / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(temp_root)))
    logger.debug(f"Temporary directory initialized at: {temp_path}")
    return temp_path
