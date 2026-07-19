"""Response cache client hashing prompts using SHA256 mapping keys to JSON entries."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.core.paths import Paths

logger = get_logger(__name__)


class LLMCache:
    """JSON-file backed cache client for indexing and retrieving LLM prompt responses."""

    def __init__(self, cache_file: Path | None = None, ttl_seconds: int = 86400) -> None:
        """Initialize LLMCache.

        Args:
            cache_file: Optional Path to cache file.
            ttl_seconds: Caching TTL in seconds.
        """
        self.cache_file = cache_file or Paths.WORKSPACE_DIR / "cache" / "llm_cache.json"
        self.ttl_seconds = ttl_seconds

        # Ensure parent folders exist
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        """Internal helper to load cache index dictionary from disk."""
        self.data_: dict[str, dict[str, Any]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    self.data_ = json.load(f)
                logger.info(f"LLMCache: Loaded {len(self.data_)} cached entries from {self.cache_file}")
            except Exception as e:
                logger.warning(f"LLMCache: Failed to load cache file: {e}. Starting fresh.")

    def _save_cache(self) -> None:
        """Internal helper to save cache index dictionary to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data_, f, indent=4)
        except Exception as e:
            logger.error(f"LLMCache: Failed to save cache file: {e}")

    @staticmethod
    def calculate_hash(text: str) -> str:
        """Calculate SHA256 hash representing target prompt text.

        Args:
            text: Prompt text.

        Returns:
            str: Hex digest hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> str | None:
        """Fetch cached response string if key matches and within TTL intervals.

        Args:
            prompt: Original prompt text.

        Returns:
            Optional[str]: Cached response text or None.
        """
        prompt_hash = self.calculate_hash(prompt)
        entry = self.data_.get(prompt_hash)

        if not entry:
            return None

        timestamp = entry.get("timestamp", 0.0)
        age = time.time() - timestamp

        if age > self.ttl_seconds:
            logger.info(f"LLMCache: Cache hit for key {prompt_hash[:8]} but expired (age: {round(age, 1)}s > TTL: {self.ttl_seconds}s).")
            # Prune expired entry
            del self.data_[prompt_hash]
            self._save_cache()
            return None

        logger.info(f"LLMCache: Cache HIT for key {prompt_hash[:8]} (age: {round(age, 1)}s)")
        return entry.get("response")

    def set(self, prompt: str, response: str) -> None:
        """Record response text inside cache indexing.

        Args:
            prompt: Original prompt text.
            response: LLM response text.
        """
        prompt_hash = self.calculate_hash(prompt)
        self.data_[prompt_hash] = {
            "timestamp": time.time(),
            "response": response
        }
        self._save_cache()
        logger.info(f"LLMCache: Cached response for key {prompt_hash[:8]}")
