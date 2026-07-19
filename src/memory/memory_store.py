"""Thread-safe key-value memory store supporting optional Time-To-Live (TTL) evictions."""

import threading
import time
from typing import Any

from src.core.logger import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """Thread-safe in-memory key-value cache supporting custom TTL expirations."""

    def __init__(self) -> None:
        """Initialize the memory store with dictionary buffers and thread locks."""
        self._lock = threading.Lock()
        # Structure: key -> (value, expiration_timestamp_or_none)
        self._store: dict[str, tuple[Any, float | None]] = {}

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value in memory with an optional TTL expiration limit.

        Args:
            key: Target unique key string.
            value: Object or payload data.
            ttl_seconds: Cache duration limit in seconds. None indicates persistent cache.
        """
        logger.debug(f"MemoryStore: Setting key='{key}' with TTL={ttl_seconds}")

        expiration = None
        if ttl_seconds is not None:
            expiration = time.time() + ttl_seconds

        with self._lock:
            self._store[key] = (value, expiration)

    def get(self, key: str) -> Any | None:
        """Retrieve a value by its key. Handles lazy TTL eviction checks.

        Args:
            key: Target key string.

        Returns:
            Optional[Any]: The cached value if found and not expired, else None.
        """
        logger.debug(f"MemoryStore: Fetching key='{key}'")

        with self._lock:
            if key not in self._store:
                return None

            value, expiration = self._store[key]

            # Check for TTL expiry
            if expiration is not None and time.time() > expiration:
                logger.info(f"MemoryStore: Key '{key}' expired. Evicting from cache.")
                del self._store[key]
                return None

            return value

    def delete(self, key: str) -> bool:
        """Remove a key and its value from the store.

        Args:
            key: Target key string to delete.

        Returns:
            bool: True if key existed and was deleted, False otherwise.
        """
        logger.debug(f"MemoryStore: Deleting key='{key}'")
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a key is present and has not expired.

        Args:
            key: Target key string.

        Returns:
            bool: True if key exists and is valid.
        """
        # Call get to check for TTL eviction, then evaluate existence
        return self.get(key) is not None

    def clear(self) -> None:
        """Evict all items from the memory store cache."""
        logger.info("MemoryStore: Clearing all cache entries.")
        with self._lock:
            self._store.clear()

    def get_size(self) -> int:
        """Get the total count of active items in the cache (filtering out expired ones).

        Returns:
            int: Number of valid keys.
        """
        with self._lock:
            current_time = time.time()
            valid_keys = [
                k
                for k, (_, exp) in self._store.items()
                if exp is None or current_time <= exp
            ]
            return len(valid_keys)
