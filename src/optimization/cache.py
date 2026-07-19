"""RAM-and-file-backed cache provider for model pipelines, plots, and prompt query responses."""

import hashlib
from pathlib import Path
from typing import Any

import joblib

from src.core.logger import get_logger
from src.optimization.config import OptimizationConfig

logger = get_logger(__name__)


class OptimizationCache:
    """Enterprise-grade multi-level caching system utilizing RAM and persistent local files."""

    _ram_cache: dict[str, Any] = {}
    _cache_dir: Path = OptimizationConfig.CACHE_DIR

    @classmethod
    def _initialize(cls) -> None:
        """Ensure cache directory exists."""
        cls._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _hash_key(cls, key: str) -> str:
        """Create a standard hex digest hash signature for a key string."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, key: str) -> Any | None:
        """Retrieve a cached object from memory or disk.

        Args:
            key: Lookup identifier key.

        Returns:
            Optional[Any]: Deserialized cached object or None if cache miss.
        """
        hashed = cls._hash_key(key)

        # 1. RAM Lookup (fastest)
        if hashed in cls._ram_cache:
            logger.info(f"Cache Hit (RAM): {key[:50]}...")
            return cls._ram_cache[hashed]

        # 2. File Lookup
        cls._initialize()
        file_path = cls._cache_dir / f"{hashed}.joblib"
        if file_path.exists():
            try:
                logger.info(f"Cache Hit (Disk): {key[:50]}...")
                val = joblib.load(file_path)
                cls._ram_cache[hashed] = val  # populate RAM cache
                return val
            except Exception as e:
                logger.warning(f"Failed to read cache file {file_path}: {e}")
                return None

        return None

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Write an object to both RAM and persistent disk cache.

        Args:
            key: Lookup identifier key.
            value: Python object representation to save.
        """
        hashed = cls._hash_key(key)
        cls._ram_cache[hashed] = value

        cls._initialize()
        file_path = cls._cache_dir / f"{hashed}.joblib"
        try:
            joblib.dump(value, file_path)
            logger.info(f"Cache Set: {key[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache file {file_path}: {e}")

    @classmethod
    def get_prompt_response(cls, prompt: str) -> str | None:
        """Fetch cached LLM responses to avoid redundant query runs.

        Args:
            prompt: Text prompt sent to LLM.

        Returns:
            Optional[str]: Cached text output or None.
        """
        if not OptimizationConfig.ENABLE_PROMPT_CACHING:
            return None

        hashed = cls._hash_key(prompt)
        cls._initialize()
        text_path = cls._cache_dir / f"prompt_{hashed}.txt"
        if text_path.exists():
            try:
                return text_path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    @classmethod
    def set_prompt_response(cls, prompt: str, response: str) -> None:
        """Cache an LLM response string on disk.

        Args:
            prompt: Text prompt sent to LLM.
            response: Response text from LLM.
        """
        if not OptimizationConfig.ENABLE_PROMPT_CACHING:
            return

        hashed = cls._hash_key(prompt)
        cls._initialize()
        text_path = cls._cache_dir / f"prompt_{hashed}.txt"
        try:
            text_path.write_text(response, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save cached prompt response: {e}")

    @classmethod
    def delete(cls, key: str) -> None:
        """Delete an object from both RAM and persistent disk cache.

        Args:
            key: Lookup identifier key.
        """
        hashed = cls._hash_key(key)
        if hashed in cls._ram_cache:
            del cls._ram_cache[hashed]

        file_path = cls._cache_dir / f"{hashed}.joblib"
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

    @classmethod
    def clear(cls) -> None:
        """Wipe both RAM cache and physical files in the cache folder."""
        cls._ram_cache.clear()
        cls._initialize()
        for f in cls._cache_dir.glob("*"):
            try:
                if f.is_file():
                    f.unlink()
            except Exception:
                pass
        logger.info("Optimization Cache: Cleared all entries successfully.")


def compress_prompt_context(text: str) -> str:
    """Compress query instructions context by stripping extra whitespaces and newlines.

    Args:
        text: Raw verbose prompt string.

    Returns:
        str: Token-reduced query prompt.
    """
    lines = [line.strip() for line in text.split("\n")]
    filtered = [line for line in lines if line]
    return "\n".join(filtered)
