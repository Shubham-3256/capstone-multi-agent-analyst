"""LangGraph node adapters that call existing agents without duplicating logic."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.orchestration.error_handler import NodeExecutor
from src.orchestration.state import WorkflowState
from src.tools.data_loader import DataLoader


def _get(obj: Any, key: str) -> Any:
    """Safely get an attribute or key from a Pydantic model or dictionary."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class WorkflowNodes:
    """Builds node functions using injected agent callbacks or existing agent instances."""

    def __init__(
        self, executor: NodeExecutor, callbacks: dict[str, Callable[..., Any]]
    ) -> None:
        self.executor = executor
        self.callbacks = callbacks

    def load_dataset(self, state: WorkflowState) -> dict[str, Any]:
        # Run loader to validate that the file is readable, but do not keep the raw DataFrame in the state
        # to avoid msgpack serialization exceptions during Graph/Streamlit state transitions.
        self.executor.run(
            "load_dataset",
            state,
            lambda _: DataLoader.load_file(Path(state.dataset_path)),
        )
        return {"dataframe": None}

    def data_intelligence(self, state: WorkflowState) -> dict[str, Any]:
        result = self.executor.run(
            "data_intelligence",
            state,
            lambda item: self.callbacks["data_intelligence"](
                Path(item.dataset_path), item.metadata.target_column
            ),
        )
        if result is None or not getattr(result, "is_valid", False):
            state.warnings.append("Data intelligence did not produce a valid profile.")
            return {}
        return {"dataset_profile": result.profile, "data_intelligence_result": result}

    def feature_engineering(self, state: WorkflowState) -> dict[str, Any]:
        def invoke(item: WorkflowState) -> Any:
            source = _get(item.data_intelligence_result, "cleaned_filepath")
            return self.callbacks["feature_engineering"](
                DataLoader.load_file(Path(source)), item.metadata.target_column
            )

        result = self.executor.run("feature_engineering", state, invoke)
        return {"feature_result": result} if result is not None else {}

    def machine_learning(self, state: WorkflowState) -> dict[str, Any]:
        def invoke(item: WorkflowState) -> Any:
            feature = item.feature_result
            target = item.metadata.target_column
            train_path = _get(feature, "train_filepath")
            val_path = _get(feature, "val_filepath")
            train, valid = pd.read_csv(train_path), pd.read_csv(val_path)
            return self.callbacks["machine_learning"](
                train.drop(columns=[target]),
                train[target],
                valid.drop(columns=[target]),
                valid[target],
            )

        result = self.executor.run("machine_learning", state, invoke)
        return {"ml_result": result} if result is not None else {}

    def visualization(self, state: WorkflowState) -> dict[str, Any]:
        result = self.executor.run(
            "visualization",
            state,
            lambda item: self.callbacks["visualization"](
                _get(item.data_intelligence_result, "cleaned_filepath"),
                item.feature_result,
                item.ml_result,
                item.metadata.target_column,
            ),
        )
        return {"visualization_result": result} if result is not None else {}

    def business_insights(self, state: WorkflowState) -> dict[str, Any]:
        result = self.executor.run(
            "business_insights",
            state,
            lambda item: self.callbacks["business_insights"](
                item.dataset_profile,
                item.feature_result,
                item.ml_result,
                item.visualization_result,
            ),
        )
        return {"business_result": result} if result is not None else {}

    def report_generation(self, state: WorkflowState) -> dict[str, Any]:
        result = self.executor.run(
            "report_generation",
            state,
            lambda item: self.callbacks["report_generation"](
                item.dataset_profile,
                item.feature_result,
                item.ml_result,
                item.visualization_result,
                item.business_result,
                item.metadata.template_type,
            ),
        )
        return {"report_result": result} if result is not None else {}
