"""Production-oriented LangGraph workflow that coordinates existing agents."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.business_insights.agent import BusinessInsightAgent
from src.agents.data_intelligence.agent import DataIntelligenceAgent
from src.agents.feature_engineering.agent import FeatureEngineeringAgent
from src.agents.machine_learning.agent import MachineLearningAgent
from src.agents.report_generation.agent import ReportGenerationAgent
from src.agents.visualization.agent import VisualizationAgent
from src.database.database import DatabaseManager, SessionLocal, init_db
from src.database.models import WorkflowExecution
from src.orchestration.checkpoint import FileCheckpointStore, memory_checkpointer
from src.orchestration.config import WorkflowConfig
from src.orchestration.error_handler import NodeExecutor
from src.orchestration.events import EventBus, WorkflowEvent
from src.orchestration.nodes import WorkflowNodes
from src.orchestration.router import WorkflowRouter
from src.orchestration.state import WorkflowMetadata, WorkflowResult, WorkflowState


def serialize_state(state: WorkflowState) -> Dict[str, Any]:
    """Safely serialize state to a JSON-compatible dictionary, converting SimpleNamespace and numpy types."""
    import numpy as np
    from types import SimpleNamespace

    def fallback(v: Any) -> Any:
        if isinstance(v, SimpleNamespace):
            return vars(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        try:
            return str(v)
        except Exception:
            return None

    return state.model_dump(mode="json", fallback=fallback)


class WorkflowGraph:
    """Coordinates the agent pipeline while retaining business logic in agents."""


    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        callbacks: Optional[Dict[str, Callable[..., Any]]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Create a graph with optional injected callbacks for tests or alternate agents."""
        self.config = config or WorkflowConfig()
        self.events = event_bus or EventBus()
        init_db()
        self.db = SessionLocal()
        self.callbacks = callbacks or self._default_callbacks()
        self.file_checkpoints = FileCheckpointStore(self.config.checkpoint_dir)
        self.nodes = WorkflowNodes(NodeExecutor(self.config.max_retries, self.events), self.callbacks)
        self.graph = self._compile()

    def _default_callbacks(self) -> Dict[str, Callable[..., Any]]:
        """Adapt existing agent methods to the graph callback interface."""
        return {
            "data_intelligence": lambda path, target: DataIntelligenceAgent(self.db).run(path, target),
            "feature_engineering": lambda dataframe, target: FeatureEngineeringAgent(self.db).run(dataframe, target),
            "machine_learning": lambda train, target, valid, valid_target: MachineLearningAgent(self.db).run(train, target, valid, valid_target),
            "visualization": lambda profile, feature, ml: VisualizationAgent(self.db).run(profile, feature, ml),
            "business_insights": lambda profile, feature, ml, viz: BusinessInsightAgent(self.db).run(profile, feature, ml, viz),
            "report_generation": lambda profile, feature, ml, viz, business, template: ReportGenerationAgent(self.db).run(profile, feature, ml, viz, business, template),
        }

    @staticmethod
    def _state(raw: Dict[str, Any]) -> WorkflowState:
        """Convert LangGraph's dictionary state to the canonical Pydantic model."""
        return WorkflowState.model_validate(raw)

    def _wrap(self, callback: Callable[[WorkflowState], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Adapt typed node callbacks to the dictionary protocol used by StateGraph."""
        def node(raw: Dict[str, Any]) -> Dict[str, Any]:
            state = self._state(raw)
            updates = callback(state)
            for key, value in updates.items():
                setattr(state, key, value)
            return serialize_state(state)
        return node

    def _route_data(self, raw: Dict[str, Any]) -> str:
        """Return the next node after the data intelligence decision."""
        return WorkflowRouter.after_data_intelligence(self._state(raw))

    def _route_feature(self, raw: Dict[str, Any]) -> str:
        """Return the next node after feature engineering."""
        return WorkflowRouter.after_feature_engineering(self._state(raw))

    def _compile(self) -> Any:
        """Create the directed LangGraph topology and conditional routes."""
        builder = StateGraph(dict)
        builder.add_node("load_dataset", self._wrap(self.nodes.load_dataset))
        builder.add_node("data_intelligence", self._wrap(self.nodes.data_intelligence))
        builder.add_node("feature_engineering", self._wrap(self.nodes.feature_engineering))
        builder.add_node("machine_learning", self._wrap(self.nodes.machine_learning))
        builder.add_node("visualization", self._wrap(self.nodes.visualization))
        builder.add_node("business_insights", self._wrap(self.nodes.business_insights))
        builder.add_node("report_generation", self._wrap(self.nodes.report_generation))
        builder.add_node("finalize", lambda state: state)
        builder.add_edge(START, "load_dataset")
        builder.add_edge("load_dataset", "data_intelligence")
        builder.add_conditional_edges("data_intelligence", self._route_data)
        builder.add_conditional_edges("feature_engineering", self._route_feature)
        builder.add_edge("machine_learning", "visualization")
        builder.add_edge("visualization", "business_insights")
        builder.add_edge("business_insights", "report_generation")
        builder.add_edge("report_generation", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=memory_checkpointer() if self.config.checkpoint_mode == "memory" else None)

    def run(self, dataset_path: str, target_column: Optional[str] = None, workflow_id: Optional[str] = None) -> WorkflowResult:
        """Execute the graph and return a typed result, including graceful skips/errors."""
        identifier = workflow_id or str(uuid.uuid4())
        if self.config.checkpoint_mode == "file" and workflow_id:
            restored = self.file_checkpoints.load(workflow_id)
            if restored:
                return WorkflowResult(is_success=not restored.errors, state=restored, output_paths=self._paths(restored))
        state = WorkflowState(
            dataset_path=str(Path(dataset_path)),
            metadata=WorkflowMetadata(workflow_id=identifier, target_column=target_column, template_type=self.config.template_type),
        )
        self.events.publish(WorkflowEvent("workflow_started", identifier))
        invoke_config = {"configurable": {"thread_id": identifier}} if self.config.checkpoint_mode == "memory" else {}
        final_raw = self.graph.invoke(serialize_state(state), config=invoke_config)
        final_state = self._state(final_raw)
        final_state.metadata.completed_at = datetime.now(timezone.utc)
        if self.config.checkpoint_mode == "file":
            self.file_checkpoints.save(final_state)
        result = WorkflowResult(is_success=not final_state.errors, state=final_state, output_paths=self._paths(final_state))
        self._persist(result)
        self.events.publish(WorkflowEvent("workflow_completed", identifier, payload={"success": result.is_success}))
        return result

    @staticmethod
    def _paths(state: WorkflowState) -> Dict[str, str]:
        """Extract report output paths without coupling orchestration to report internals."""
        res = state.report_result
        if res is None:
            return {}
        if isinstance(res, dict):
            return dict(res.get("output_paths", {}) or {})
        return dict(getattr(res, "output_paths", {}) or {})

    def _persist(self, result: WorkflowResult) -> None:
        """Persist workflow history, durations, errors, and output summary."""
        if not self.config.persist_execution:
            return
        record = WorkflowExecution(
            workflow_id=result.state.metadata.workflow_id,
            status="completed" if result.is_success else "degraded",
            history_json=json.dumps([item.model_dump(mode="json") for item in result.state.execution_history]),
            errors_json=json.dumps(result.state.errors),
            timing_json=json.dumps(result.state.timing),
        )
        self.db.add(record)
        self.db.commit()
