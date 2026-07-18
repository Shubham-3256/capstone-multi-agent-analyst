"""Demonstrate Phase 11 dashboard integration and workflow execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph
from src.database.database import init_db


def main() -> None:
    """Run a mock pipeline to populate the DB history and print launch instructions."""
    print("==========================================================")
    print("Multi-Agent AI Data Analyst - Phase 11 Demo & Verification")
    print("==========================================================")
    
    # 1. Initialize DB schema
    print("\n[Step 1] Initializing SQLite database...")
    init_db()
    
    # 2. Run graph execution
    print("\n[Step 2] Executing workflow graph using mock agent callbacks...")
    dataset = PROJECT_ROOT / "workspace" / "uploads" / "mock_churn.csv"
    
    # Create mock dataset if missing
    if not dataset.exists():
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("customer_id,age,monthly_charges,churn\nC1,32.0,65.5,0\nC2,23.0,110.25,1\n", encoding="utf-8")

    # Define mock return objects mapping our models
    mock_profile = SimpleNamespace(
        row_count=100,
        column_count=4,
        memory_usage_bytes=450,
        columns={
            "age": SimpleNamespace(name="age", dtype="float64", unique_count=45, null_count=0, null_percentage=0.0, numeric_summary={"mean": 38.5, "min": 18.0, "max": 90.0}, categorical_summary=None, date_summary=None),
            "churn": SimpleNamespace(name="churn", dtype="int64", unique_count=2, null_count=0, null_percentage=0.0, numeric_summary=None, categorical_summary={"top": 0, "freq": 80}, date_summary=None)
        },
        correlation_matrix={"age": {"age": 1.0, "monthly_charges": 0.42}, "monthly_charges": {"age": 0.42, "monthly_charges": 1.0}},
        target_distribution={"0": 80, "1": 20},
        recommended_ml_task="classification",
        recommendations=[SimpleNamespace(title="Target Imbalance", description="Stratified splits recommended.", severity="warning", column="churn")]
    )
    
    mock_di = SimpleNamespace(
        is_valid=True,
        profile=mock_profile,
        cleaned_filepath=str(dataset),
        validation_report=SimpleNamespace(is_valid=True, issues=[], summary={"total_issues": 0}),
        cleaning_report=SimpleNamespace(transformations=[], initial_shape=[100, 4], final_shape=[100, 4], duration_seconds=0.1)
    )

    mock_fe = SimpleNamespace(
        dataset_id="mock-id",
        is_success=True,
        feature_types={"age": "numeric", "monthly_charges": "numeric", "churn": "categorical"},
        encoding_report=SimpleNamespace(strategy_used={}, mappings={}),
        scaling_report=SimpleNamespace(scaler_type={"age": "standard"}, scaling_parameters={"age": {"mean": 38.5, "scale": 12.3}}),
        selection_report=SimpleNamespace(method="mutual_info", original_count=3, selected_count=2, selected_features=["age", "monthly_charges"], feature_importances={"age": 0.42, "monthly_charges": 0.35}),
        leakage_report=SimpleNamespace(has_leakage=False, leakage_issues=[]),
        split_report=SimpleNamespace(train_shape=[70, 3], val_shape=[15, 3], test_shape=[15, 3], strategy="stratified"),
        pipeline_report=SimpleNamespace(pipeline_filepath="workspace/models/pipeline.joblib", components=["scaling", "selection"]),
        train_filepath=str(dataset),
        val_filepath=str(dataset),
        test_filepath=str(dataset),
        duration_seconds=0.2
    )

    mock_ml = SimpleNamespace(
        task_report=SimpleNamespace(task_type="classification", classes=[0, 1], is_binary=True),
        best_model_name="RandomForestClassifier",
        best_model_path="workspace/models/best_model.joblib",
        leaderboard=SimpleNamespace(entries=[
            SimpleNamespace(model_name="RandomForestClassifier", rank=1, score=0.85, metrics={"accuracy": 0.85, "f1": 0.83}),
            SimpleNamespace(model_name="LogisticRegression", rank=2, score=0.81, metrics={"accuracy": 0.81, "f1": 0.79})
        ]),
        best_metrics={"accuracy": 0.85, "f1": 0.83},
        feature_importances=[
            SimpleNamespace(column="age", importance=0.6),
            SimpleNamespace(column="monthly_charges", importance=0.4)
        ],
        duration_seconds=0.5
    )

    mock_viz = SimpleNamespace(
        is_success=True,
        report=SimpleNamespace(
            charts=[
                SimpleNamespace(chart_id="correlation_heatmap", title="Correlation Grid", chart_type="heatmap", file_path="plots/correlation_heatmap.png", html_path=None, caption=SimpleNamespace(summary="Weak correlation", details="No features show significant colinearity.")),
            ],
            metadata=SimpleNamespace(dataset_hash="mock-hash", created_at="2026-07-18", theme="corporate")
        ),
        output_directory="workspace/plots",
        duration_seconds=0.3
    )

    mock_biz = SimpleNamespace(
        executive_summary=SimpleNamespace(headline="High Churn Risk in Age Bracket", key_takeaways=["Targeted action recommended", "Ensure database scaling"], impact_statement="Reduction in churn by 5%"),
        dataset_insight=SimpleNamespace(completeness_score=1.0, anomalies_detected=[], recommendation="Clean"),
        model_insight=SimpleNamespace(algorithm_name="RandomForestClassifier", accuracy=0.85, f1=0.83, feature_weights={"age": 0.6}, conclusion="Valid model"),
        recommendations=[SimpleNamespace(title="Promotions", description="Send discount codes", actionability="high")],
        risks=[SimpleNamespace(severity="medium", probability="low", description="Overfitting potential")],
        confidence_report=SimpleNamespace(confidence_score=0.9, reliability_rating="high", justification="High validation accuracy"),
        token_usage={"input_tokens": 1200, "output_tokens": 400},
        estimated_cost_usd=0.02
    )

    mock_rep = SimpleNamespace(
        is_success=True,
        output_paths={"pdf": "workspace/reports/report.pdf", "markdown": "workspace/reports/report.md"},
        manifest=SimpleNamespace(report_id="mock-report-id", dataset_hash="mock-hash", pipeline_version="1.0.0", model_version="gpt-4o", charts_included=[], sections=["Executive Summary"], generation_timestamp="2026-07-18"),
        metadata=SimpleNamespace(title="Customer Churn Analysis", author="Multi-Agent Platform", template_type="executive", created_at="2026-07-18"),
        duration_seconds=0.4
    )

    callbacks = {
        "data_intelligence": lambda *_: mock_di,
        "feature_engineering": lambda *_: mock_fe,
        "machine_learning": lambda *_: mock_ml,
        "visualization": lambda *_: mock_viz,
        "business_insights": lambda *_: mock_biz,
        "report_generation": lambda *_: mock_rep,
    }
    
    # Executing using file checkpointer mode to write checkpoint JSON
    workflow = WorkflowGraph(config=WorkflowConfig(checkpoint_mode="file", persist_execution=True), callbacks=callbacks)
    result = workflow.run(str(dataset), target_column="churn")
    
    print("Execution history status:", [item.status for item in result.state.execution_history])
    print("Persisted timing info:", result.state.timing)
    print("Checkpoint Saved path:", workflow.file_checkpoints.directory / f"{result.state.metadata.workflow_id}.json")

    print("\n[Step 3] Verification complete.")
    print("\nTo launch the enterprise Streamlit dashboard locally, run:")
    print("  streamlit run app/Home.py")
    print("==========================================================")


if __name__ == "__main__":
    main()
