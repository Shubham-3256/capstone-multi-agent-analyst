"""Retry and graceful-degradation wrapper for graph nodes."""

import time
from collections.abc import Callable
from typing import Any

from src.orchestration.events import EventBus, WorkflowEvent
from src.orchestration.state import NodeExecution, WorkflowState


class NodeExecutor:
    """Executes coordinator node callbacks with retry and audit tracking."""

    def __init__(self, max_retries: int, events: EventBus) -> None:
        self.max_retries = max_retries
        self.events = events

    def run(self, node_name: str, state: WorkflowState, callback: Callable[[WorkflowState], Any]) -> Any:
        """Execute a callback, recording recoverable failures in workflow state."""
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            self.events.publish(WorkflowEvent("node_started", state.metadata.workflow_id, node_name))
            try:
                result = callback(state)
                duration = time.perf_counter() - started
                state.execution_history.append(NodeExecution(node_name=node_name, status="completed", duration_seconds=duration, retries=attempt))
                state.timing[node_name] = duration
                self.events.publish(WorkflowEvent("node_completed", state.metadata.workflow_id, node_name))
                return result
            except Exception as exc:
                if attempt == self.max_retries:
                    duration = time.perf_counter() - started
                    message = f"{node_name}: {exc}"
                    state.errors.append(message)
                    state.execution_history.append(NodeExecution(node_name=node_name, status="failed", duration_seconds=duration, retries=attempt, error=str(exc)))
                    self.events.publish(WorkflowEvent("node_failed", state.metadata.workflow_id, node_name, {"error": str(exc)}))
                    return None
