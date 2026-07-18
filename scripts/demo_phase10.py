"""Demonstrate Phase 10 orchestration with dependency-injected offline agents."""

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph


def main() -> None:
    """Run the workflow over a sample dataset and print orchestration telemetry."""
    dataset = PROJECT_ROOT / "workspace" / "uploads" / "mock_churn.csv"
    callbacks = {
        "data_intelligence": lambda path, target: SimpleNamespace(is_valid=True, profile={"dataset": path.name}, cleaned_filepath=str(path)),
        "feature_engineering": lambda *_: None,
        "machine_learning": lambda *_: None,
        "visualization": lambda *_: SimpleNamespace(is_success=True),
        "business_insights": lambda *_: SimpleNamespace(),
        "report_generation": lambda *_: SimpleNamespace(output_paths={"markdown": "mock-report.md"}),
    }
    workflow = WorkflowGraph(config=WorkflowConfig(checkpoint_mode="file", persist_execution=False), callbacks=callbacks)
    result = workflow.run(str(dataset))
    print("Execution order:", [item.node_name for item in result.state.execution_history])
    print("Timing:", result.state.timing)
    print("Skipped nodes:", ["feature_engineering", "machine_learning"])
    print("Final outputs:", result.output_paths)


if __name__ == "__main__":
    main()
