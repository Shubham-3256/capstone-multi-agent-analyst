"""Security utility module for the Multi-Agent AI Data Analyst.

Provides services for filename validation, extension checks, path traversal
prevention, cryptographic hashing, and secure identifier generation.
"""

import hashlib
import re
import unicodedata
import uuid
from pathlib import Path

from src.core.exceptions import ValidationException


def validate_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """Check if the provided filename has an extension within the allowed set.

    Args:
        filename: Name of the file (e.g., "dataset.csv").
        allowed_extensions: Set of allowed file extensions (e.g., {"csv", "json"}).

    Returns:
        bool: True if extension is allowed, False otherwise.
    """
    if not filename or "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in {allowed.lower().strip(".") for allowed in allowed_extensions}


def secure_filename(filename: str) -> str:
    """Sanitize a user-provided filename to prevent filesystem injection.

    Converts characters to ASCII, removes relative directories, replaces whitespace,
    and strips special symbols. Follows standard secure filename patterns.

    Args:
        filename: Raw user-provided filename.

    Returns:
        str: Sanitized, filesystem-safe filename.
    """
    if not filename:
        return "unnamed_file"

    # Normalize unicode representations
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    # Get pure filename, discarding any directory component
    filename = Path(filename).name

    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Keep only letters, numbers, underscores, dots, and dashes
    filename = re.sub(r"[^\w\.\-]", "", filename)

    # Strip leading/trailing dots and dashes to prevent hidden files or root directory jumps
    filename = filename.strip("._-")

    # Fallback if the filename became empty
    if not filename:
        return "secured_file_" + str(uuid.uuid4())[:8]

    return filename


def verify_path_bounds(base_dir: Path, target_path: Path) -> Path:
    """Ensure a target path resides strictly within base_dir.

    Prevents directory traversal (e.g. using `../` to access files outside the workspace).

    Args:
        base_dir: The directory that contains the target (e.g. UPLOAD_DIR).
        target_path: Path representing the location to evaluate.

    Returns:
        Path: The fully resolved target path.

    Raises:
        ValidationException: If directory traversal is detected.
    """
    resolved_base = base_dir.resolve()
    # Resolve the full target path. If file does not exist, resolve parent and append name
    try:
        resolved_target = target_path.resolve()
    except Exception:
        # If resolving fails (e.g., directory does not exist yet), resolve parent and join
        resolved_target = Path(target_path.parent).resolve() / target_path.name

    # Check that target resides inside the base directory bounds
    if (
        resolved_base not in resolved_target.parents
        and resolved_base != resolved_target
    ):
        raise ValidationException(
            message=f"Path traversal detected: Target path '{resolved_target}' resides outside base '{resolved_base}'",
            details={"base": str(resolved_base), "target": str(resolved_target)},
        )

    return resolved_target


def generate_secure_id() -> str:
    """Generate a high-entropy cryptographically secure UUID4 string.

    Returns:
        str: 36-character UUID string.
    """
    return str(uuid.uuid4())


def generate_sha256_hash(data: str | bytes) -> str:
    """Calculate the SHA-256 hash of a string or byte stream.

    Args:
        data: Input content to hash.

    Returns:
        str: 64-character hexadecimal digest representation.
    """
    sha = hashlib.sha256()
    if isinstance(data, str):
        sha.update(data.encode("utf-8"))
    else:
        sha.update(data)
    return sha.hexdigest()
