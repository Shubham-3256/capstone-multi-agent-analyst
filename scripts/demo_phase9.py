"""Run a self-contained Phase 9 report-generation demonstration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.report_generation.agent import ReportGenerationAgent
from src.agents.report_generation.config import ReportConfig
from src.database.database import DatabaseManager, init_db


def run_report_generation_demo() -> None:
    """Generate executive and technical reports from a representative profile."""
    init_db()
    profile = (
        "Customer churn dataset: 100 observations, fields include age, monthly "
        "charges, city, and churn target. Completeness: 100%."
    )
    with DatabaseManager.get_session() as session:
        agent = ReportGenerationAgent(session, ReportConfig())
        executive = agent.run(dataset_profile=profile, template_type="executive")
        technical = agent.run(dataset_profile=profile, template_type="technical")

    print("Executive report:", executive.output_paths)
    print("Technical report:", technical.output_paths)
    print("Manifest:", executive.manifest.report_id)


if __name__ == "__main__":
    run_report_generation_demo()
