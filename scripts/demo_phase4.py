"""Demo script for Phase 4 - Data Intelligence Agent validations, cleanings, and profilings."""

import sys
from pathlib import Path

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core import get_logger
from src.database import init_db, DatabaseManager
from src.agents.data_intelligence.agent import DataIntelligenceAgent

logger = get_logger("demo_phase4")


def generate_defect_dataset() -> Path:
    """Write a sample CSV dataset with structural defects and NaNs to test the agent pipeline.

    Returns:
        Path: Location of the generated file.
    """
    defect_file = project_root / "workspace" / "churn_with_defects.csv"
    defect_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset contains:
    # 1. Spaced & capitalized column headers: "Customer ID", " Age", "Monthly Charges "
    # 2. Duplicate rows: Row 2 and Row 6 are duplicates
    # 3. Infinite values: Salary in row 4 is inf
    # 4. Whitespaces in string column: New York with spaces
    # 5. Missing values (NaNs) in Age and Target
    content = """Customer ID, Age,Monthly Charges , Salary, City,Target
C001,34, 65.5,50000.0,  New York  ,0
C002,45, 80.0,75000.0,Chicago,1
C003,23, 35.4,42000.0, Boston ,0
C004,56,110.25,inf,  New York  ,1
C005,, 70.1,64000.0,Chicago,0
C001,34, 65.5,50000.0,  New York  ,0
"""
    defect_file.write_text(content, encoding="utf-8")
    logger.info(f"Generated sample defect churn dataset for demo at: {defect_file}")
    return defect_file


def run_agent_demo() -> None:
    """Execute the Data Intelligence Agent pipeline on the defective dataset."""
    logger.info("==========================================================")
    logger.info("Starting Phase 4 Data Intelligence Agent Demo Pipeline")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Setup mock data file
    defect_file = generate_defect_dataset()

    # 3. Run Data Intelligence Pipeline via Agent
    with DatabaseManager.get_session() as session:
        agent = DataIntelligenceAgent(session)
        
        # Configure cleaning strategies
        # Impute Age with median, replace Salary outliers with Z-score caps, and cast Age to Int
        result = agent.run(
            file_path=defect_file,
            target_column="Target",
            imputation_strategies={"age": "median", "target": "mode"},
            outlier_strategies={"salary": "zscore_cap"},
            datatype_conversions={"age": "int"}
        )

    # 4. Output results
    print("\n" + "=" * 60)
    print("DATA INTELLIGENCE PIPELINE RESULTS")
    print("=" * 60)
    print(f"Dataset ID: {result.dataset_id}")
    print(f"Is Valid:   {result.is_valid}")
    print(f"Clean File: {result.cleaned_filepath}")
    print(f"Duration:   {round(result.duration_seconds, 4)} seconds")
    print("=" * 60)

    # Print Validation Report Summary
    print("\n" + "-" * 40)
    print("1. VALIDATION REPORT SUMMARY")
    print("-" * 40)
    print(f"Errors found:   {result.validation_report.summary.get('errors')}")
    print(f"Warnings found: {result.validation_report.summary.get('warnings')}")
    for issue in result.validation_report.issues:
        print(f"  * [{issue.severity.upper()}] check='{issue.check_name}' col='{issue.column or 'global'}': {issue.message}")

    # Print Cleaning Report Summary
    if result.cleaning_report:
        print("\n" + "-" * 40)
        print("2. CLEANING TRANSFORMATIONS APPLIED")
        print("-" * 40)
        print(f"Initial Shape: {result.cleaning_report.initial_shape}")
        print(f"Final Shape:   {result.cleaning_report.final_shape}")
        for idx, action in enumerate(result.cleaning_report.transformations):
            print(f"  {idx + 1}. [{action.action_type.upper()}] col='{action.column or 'global'}': {action.details}")

    # Print Dataset Profiler Summary
    if result.profile:
        print("\n" + "-" * 40)
        print("3. STATISTICAL PROFILE & MODEL RECOMMENDATIONS")
        print("-" * 40)
        print(f"Recommended ML Task: {result.profile.recommended_ml_task.upper()}")
        print(f"Target Distribution: {result.profile.target_distribution}")
        print("\nRecommendations:")
        for idx, rec in enumerate(result.profile.recommendations):
            print(f"  {idx + 1}. [{rec.severity.upper()}] {rec.title}: {rec.description}")
            
    print("=" * 60 + "\n")
    logger.info("Phase 4 Agent Demo completed successfully!")


if __name__ == "__main__":
    run_agent_demo()
