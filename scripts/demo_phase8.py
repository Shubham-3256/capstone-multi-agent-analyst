"""Demo script for Phase 8 - Business Insight Agent LLM translation sweeps validation."""

import sys
from pathlib import Path

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.agents.business_insights.agent import BusinessInsightAgent
from src.agents.machine_learning.models import (
    FeatureImportance,
    Leaderboard,
    LeaderboardEntry,
    MachineLearningResult,
    TaskReport,
)
from src.agents.visualization.models import (
    ChartCaption,
    ChartMetadata,
    VisualizationReport,
    VisualizationResult,
)
from src.core import get_logger
from src.database import DatabaseManager, init_db

logger = get_logger("demo_phase8")


def compile_preceding_mock_results() -> tuple:
    """Consolidate mock results representing Phases 4-7 outputs.

    Returns:
        tuple: (dataset_profile_str, ml_result, visualization_result)
    """
    dataset_profile_str = "Customer churn table. 100 rows. Columns: age, monthly_charges, city_london, target."

    # MachineLearningResult mock
    task_report = TaskReport(task_type="classification", classes=[0, 1], is_binary=True)
    leaderboard = Leaderboard(
        entries=[
            LeaderboardEntry(
                model_name="Logistic Regression",
                rank=1,
                score=1.0,
                metrics={"accuracy": 1.0, "f1": 1.0},
            ),
            LeaderboardEntry(
                model_name="Random Forest",
                rank=2,
                score=0.95,
                metrics={"accuracy": 0.95, "f1": 0.95},
            ),
        ]
    )
    importances = [
        FeatureImportance(column="monthly_charges", importance=0.4534),
        FeatureImportance(column="age", importance=0.4127),
        FeatureImportance(column="city_london", importance=0.1058),
    ]
    ml_result = MachineLearningResult(
        task_report=task_report,
        best_model_name="Logistic Regression",
        best_model_path="workspace/artifacts/models/best_model.joblib",
        leaderboard=leaderboard,
        best_metrics={"accuracy": 1.0, "f1": 1.0},
        feature_importances=importances,
        duration_seconds=2.15,
    )

    # VisualizationResult mock
    charts = [
        ChartMetadata(
            chart_id="missing_heatmap",
            title="Missing Value Heatmap",
            chart_type="heatmap",
            file_path="plots/missing_heatmap.png",
            caption=ChartCaption(
                summary="No missing values.", details="100% data density."
            ),
        ),
        ChartMetadata(
            chart_id="correlation_heatmap",
            title="Correlation Matrix",
            chart_type="heatmap",
            file_path="plots/correlation_heatmap.png",
            caption=ChartCaption(
                summary="Strong correlation.",
                details="age & monthly_charges correlates highly.",
            ),
        ),
    ]
    viz_report = VisualizationReport(charts=charts)
    visualization_result = VisualizationResult(
        is_success=True,
        report=viz_report,
        output_directory="workspace/artifacts/",
        duration_seconds=1.85,
    )

    return dataset_profile_str, ml_result, visualization_result


def run_business_insights_demo() -> None:
    """Run BusinessInsightAgent orchestrator, print findings, and token costs summaries."""
    logger.info("==========================================================")
    logger.info("Starting Phase 8 Business Insight Agent Demo Run")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Get mock preceding results
    profile_str, ml_result, viz_result = compile_preceding_mock_results()

    # 3. Execute insights sweeps via Agent
    with DatabaseManager.get_session() as session:
        agent = BusinessInsightAgent(session)

        result = agent.run(
            dataset_profile=profile_str,
            ml_result=ml_result,
            visualization_result=viz_result,
        )

    # 4. Print structured executive report
    print("\n" + "=" * 60)
    print("ENTERPRISE BUSINESS INSIGHT REPORT")
    print("=" * 60)
    print(f"HEADLINE: {result.executive_summary.headline}")
    print("\nKEY TAKEAWAYS:")
    for takeaway in result.executive_summary.key_takeaways:
        print(f"  * {takeaway}")
    print(f"\nIMPACT STATEMENT: {result.executive_summary.impact_statement}")
    print("=" * 60)

    print("\n1. DATA QUALITY ASSESSMENT")
    print("-" * 45)
    print(f"  * Completeness Score: {result.dataset_insight.completeness_score}")
    print(f"  * Anomalies:          {result.dataset_insight.anomalies_detected}")
    print(f"  * Recommendation:     {result.dataset_insight.recommendation}")

    print("\n2. MODEL INTERPRETATION & ERROR AUDITS")
    print("-" * 45)
    print(f"  * Best Estimator:     {result.model_insight.algorithm_name}")
    print(f"  * Validation macro-F1:{result.model_insight.f1}")
    print(f"  * Key Drivers Weights:{result.model_insight.feature_weights}")
    print(f"  * Conclusion:         {result.model_insight.conclusion}")

    print("\n3. TACTICAL RECOMMENDATIONS")
    print("-" * 45)
    for rec in result.recommendations:
        print(f"  * Strategy:      {rec.title} (Actionability: {rec.actionability})")
        print(f"    Description:   {rec.description}")

    print("\n4. OPERATIONAL RISKS & OVERFIT AUDITS")
    print("-" * 45)
    for risk in result.risks:
        print(
            f"  * Risk Level:    {risk.severity} Severity / {risk.probability} Likelihood"
        )
        print(f"    Vulnerability: {risk.description}")

    print("\n5. AUDIT CONFIDENCE VERIFICATION")
    print("-" * 45)
    print(f"  * Confidence Score:   {result.confidence_report.confidence_score}")
    print(f"  * Reliability Rating: {result.confidence_report.reliability_rating}")
    print(f"  * Justification:      {result.confidence_report.justification}")

    print("\n" + "=" * 60)
    print("LLM SESSION METRICS & RUN BILLING")
    print("=" * 60)
    print(f"Input prompt tokens:     {result.token_usage.get('input_tokens')}")
    print(f"Output completion tokens: {result.token_usage.get('output_tokens')}")
    print(f"Estimated session cost:  ${round(result.estimated_cost_usd, 6)}")
    print("=" * 60 + "\n")

    logger.info("Phase 8 Agent Demo completed successfully!")


if __name__ == "__main__":
    run_business_insights_demo()
