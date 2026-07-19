"""Small in-process event system for workflow lifecycle notifications."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class WorkflowEvent:
    """Immutable event published for a workflow lifecycle change."""

    event_type: str
    workflow_id: str
    node_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Publishes workflow events to locally registered subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[WorkflowEvent], None]] = []

    def subscribe(self, callback: Callable[[WorkflowEvent], None]) -> None:
        """Register a lifecycle event subscriber."""
        self._subscribers.append(callback)

    def publish(self, event: WorkflowEvent) -> None:
        """Publish an event without allowing subscriber failure to stop the workflow."""
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                continue
