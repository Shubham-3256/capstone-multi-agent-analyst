"""Lazy loader proxy module to defer importing heavy data science libraries."""

import importlib
from typing import Any


class LazyLoader:
    """Transparent proxy that defers importing a python module until it is first accessed.

    Useful for avoiding long import delays on startup for heavy libraries.
    """

    def __init__(self, module_name: str) -> None:
        """Initialize lazy loader proxy.

        Args:
            module_name: Full package path or name of the module to dynamically import.
        """
        self._module_name = module_name
        self._module: Any = None

    def _load(self) -> Any:
        """Eagerly load the module if not already loaded."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute accesses to the imported module."""
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> list[str]:
        """Expose listing of dynamic attributes of the wrapped module."""
        module = self._load()
        return list(dir(module))
