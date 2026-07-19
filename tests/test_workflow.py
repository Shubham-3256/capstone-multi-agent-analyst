"""End-to-end orchestration tests using mock agent callbacks."""

from types import SimpleNamespace

from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph


def test_workflow_skips_modeling_and_returns_report(tmp_path):
    """Workflow runs the coordinated path and skips modeling without a target."""
    dataset = tmp_path / "customers.csv"
    dataset.write_text("age\n30\n", encoding="utf-8")
    calls = []

    def data_agent(path, target):
        calls.append("data")
        return SimpleNamespace(
            is_valid=True, profile={"rows": 1}, cleaned_filepath=str(path)
        )

    def visualization(profile, feature, model, target=None):
        calls.append("visualization")
        return SimpleNamespace(is_success=True)

    def insights(profile, feature, model, visualization):
        calls.append("insights")
        return SimpleNamespace()

    def report(profile, feature, model, visualization, business, template):
        calls.append("report")
        return SimpleNamespace(output_paths={"markdown": "report.md"})

    callbacks = {
        "data_intelligence": data_agent,
        "feature_engineering": lambda *_: (_ for _ in ()).throw(
            AssertionError("should skip")
        ),
        "machine_learning": lambda *_: (_ for _ in ()).throw(
            AssertionError("should skip")
        ),
        "visualization": visualization,
        "business_insights": insights,
        "report_generation": report,
    }
    graph = WorkflowGraph(
        config=WorkflowConfig(checkpoint_mode="none", persist_execution=False),
        callbacks=callbacks,
    )
    result = graph.run(str(dataset))
    assert result.is_success
    assert result.output_paths["markdown"] == "report.md"
    assert calls == ["data", "visualization", "insights", "report"]
