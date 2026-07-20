"""Conditional routing decisions for the LangGraph workflow."""

from src.orchestration.state import WorkflowState


class WorkflowRouter:
    """Keeps branching decisions isolated from nodes and agent implementations."""

    @staticmethod
    def after_data_intelligence(state: WorkflowState) -> str:
        """Skip feature engineering and ML when no valid target is available."""
        if state.dataset_profile is None:
            return "finalize"
        if not state.metadata.target_column:
            return "visualization"
        return "feature_engineering"

    @staticmethod
    def after_feature_engineering(state: WorkflowState) -> str:
        """Skip model training when preprocessing did not produce a result."""
        return (
            "machine_learning" if state.feature_result is not None else "visualization"
        )

    @staticmethod
    def after_machine_learning(_state: WorkflowState) -> str:
        """Continue with visualizations even when no trainable model was produced."""
        return "visualization"
