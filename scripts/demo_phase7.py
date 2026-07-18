"""Demo script for Phase 7 - Visualization Agent chart sweeps validation."""

import sys
import joblib
from pathlib import Path
import pandas as pd

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core import get_logger
from src.database import init_db, DatabaseManager
from src.agents.machine_learning.models import (
    MachineLearningResult,
    TaskReport,
    Leaderboard,
    LeaderboardEntry,
    FeatureImportance
)
from src.agents.visualization.agent import VisualizationAgent

logger = get_logger("demo_phase7")


def create_mock_ml_result() -> MachineLearningResult:
    """Instantiate a sample MachineLearningResult with best model metadata references.

    Returns:
        MachineLearningResult: AutoML results context.
    """
    # Create target directories
    models_dir = project_root / "workspace" / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model_dummy.joblib"

    # Fit and save a dummy classifier model to joblib to enable loading
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    model = RandomForestClassifier(random_state=42)
    model.fit(np.array([[1], [2], [3], [4]]), np.array([0, 1, 0, 1]))
    joblib.dump(model, str(best_model_path))

    task_report = TaskReport(
        task_type="classification",
        classes=[0, 1],
        is_binary=True
    )

    leaderboard = Leaderboard(entries=[
        LeaderboardEntry(
            model_name="Random Forest",
            rank=1,
            score=0.915,
            metrics={"accuracy": 0.92, "f1": 0.915}
        ),
        LeaderboardEntry(
            model_name="Logistic Regression",
            rank=2,
            score=0.825,
            metrics={"accuracy": 0.84, "f1": 0.825}
        )
    ])

    importances = [
        FeatureImportance(column="monthly_charges", importance=0.62),
        FeatureImportance(column="age", importance=0.28),
        FeatureImportance(column="city_london", importance=0.10)
    ]

    return MachineLearningResult(
        task_report=task_report,
        best_model_name="Random Forest",
        best_model_path=str(best_model_path),
        leaderboard=leaderboard,
        best_metrics={"accuracy": 0.92, "f1": 0.915},
        feature_importances=importances,
        duration_seconds=2.45
    )


def run_visualization_demo() -> None:
    """Run VisualizationAgent plotting sweeps, compile caption descriptions, and save HTML/PNGs."""
    logger.info("==========================================================")
    logger.info("Starting Phase 7 Visualization Agent Demo Run")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Get mock input DataFrame
    df = pd.DataFrame({
        "age": [34, 45, 23, 56, 38, 29, 62, 41, 33, 47],
        "monthly_charges": [65.5, 80.0, 35.4, 110.25, 70.1, 55.0, 95.5, 85.0, 45.0, 78.5],
        "city_london": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    })
    
    # 3. Create processed val.csv splits to enable model visualizer data reload checks
    processed_dir = project_root / "workspace" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    val_df = df.copy()
    val_df["target"] = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    val_df.to_csv(processed_dir / "val.csv", index=False)

    # 4. Load mock MachineLearningResult
    ml_result = create_mock_ml_result()
    logger.info("Mock MachineLearningResult model compiled successfully.")

    # 5. Run plotting sweeps via Agent orchestrator
    with DatabaseManager.get_session() as session:
        agent = VisualizationAgent(session)
        
        result = agent.run(
            dataset_profile=df,
            ml_result=ml_result
        )

    # 6. Display visualizers metadata reports
    print("\n" + "=" * 60)
    print("VISUALIZATION AGENT PLOTTING RUN RESULTS")
    print("=" * 60)
    print(f"Agent Execution Success: {result.is_success}")
    print(f"Output Directory Path:   {result.output_directory}")
    print(f"Sweep Duration Time:     {round(result.duration_seconds, 4)} seconds")
    print("=" * 60)

    print("\nGENERATED CHARTS & INSIGHTS SUMMARY")
    print("-" * 55)
    for chart in result.report.charts:
        print(f"  * Chart:    {chart.title} ({chart.chart_type.upper()})")
        print(f"    Slug ID:  {chart.chart_id}")
        print(f"    Summary:  {chart.caption.summary}")
        print(f"    Details:  {chart.caption.details}")
        print(f"    PNG Path: {chart.file_path}")
        if chart.html_path:
            print(f"    HTML Path: {chart.html_path}")
        print("-" * 55)

    logger.info("Phase 7 Agent Demo completed successfully!")


if __name__ == "__main__":
    run_visualization_demo()
