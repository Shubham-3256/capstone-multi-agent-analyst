"""Demo script for Phase 5 - Feature Engineering Agent pre-processing pipelines validation."""

import sys
from pathlib import Path
import pandas as pd

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core import get_logger
from src.database import init_db, DatabaseManager
from src.agents.feature_engineering.agent import FeatureEngineeringAgent

logger = get_logger("demo_phase5")


def create_demo_clean_dataset() -> pd.DataFrame:
    """Create a sample clean DataFrame mimicking the output from Phase 4.

    Returns:
        pd.DataFrame: A clean churn dataset.
    """
    # 15 rows of customer data
    data = {
        "customer_id": [f"C{i:03d}" for i in range(1, 16)],  # Identifier
        "age": [34, 45, 23, 56, 38, 29, 62, 41, 33, 47, 51, 26, 30, 44, 37],  # Numeric (Standard scaler)
        "monthly_charges": [65.5, 80.0, 35.4, 110.25, 70.1, 55.0, 95.5, 85.0, 45.0, 78.5, 90.0, 40.0, 60.0, 105.0, 68.0],  # Numeric (Standard scaler)
        "city": ["NY", "London", "NY", "Chicago", "London", "NY", "London", "Chicago", "NY", "London", "NY", "London", "Chicago", "NY", "London"],  # Low cardinality -> One Hot
        "joined_date": pd.to_datetime(["2020-01-15", "2021-06-20", "2022-12-05", "2019-08-11", "2020-11-23", "2021-02-14", "2018-04-30", "2020-07-09", "2022-03-25", "2021-10-18", "2019-12-01", "2022-08-08", "2021-04-12", "2020-09-30", "2021-01-01"]),  # Datetime
        "churn": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]  # Target
    }
    return pd.DataFrame(data)


def run_feature_engineering_demo() -> None:
    """Run the Feature Engineering Agent pipeline demo."""
    logger.info("==========================================================")
    logger.info("Starting Phase 5 Feature Engineering Agent Demo Pipeline")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Get mock cleaned DataFrame
    df = create_demo_clean_dataset()
    logger.info(f"Loaded clean dataset. Dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 3. Run Pre-processing Pipeline via Agent
    with DatabaseManager.get_session() as session:
        agent = FeatureEngineeringAgent(session)
        
        # Executes: Detect -> Generate -> Encode -> Scale -> Select -> Split -> Save Pipeline
        result = agent.run(
            dataframe=df,
            target_column="churn"
        )

    # 4. Print structured results
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE RUN RESULTS")
    print("=" * 60)
    print(f"Engineered Dataset ID: {result.dataset_id}")
    print(f"Pipeline Success:      {result.is_success}")
    print(f"Duration:              {round(result.duration_seconds, 4)} seconds")
    print("=" * 60)

    # A. Inferred feature types
    print("\n1. DETECTED FEATURE COLUMN CATEGORIES")
    print("-" * 45)
    for col, type_label in result.feature_types.items():
        print(f"  * {col}: {type_label.upper()}")

    # B. Encoding strategies applied
    print("\n2. CATEGORICAL ENCODING STRATEGIES")
    print("-" * 45)
    for col, strategy in result.encoding_report.strategy_used.items():
        print(f"  * {col} encoded strategy: {strategy.upper()}")
        if col in result.encoding_report.mappings:
            print(f"    Mappings: {result.encoding_report.mappings[col]}")

    # C. Scaling strategies applied
    print("\n3. NUMERICAL SCALING STRATEGIES")
    print("-" * 45)
    for col, scaler in result.scaling_report.scaler_type.items():
        print(f"  * {col} scaling chosen: {scaler.upper()}")

    # D. Feature selection summary
    print("\n4. FEATURE SELECTION SUMMARY")
    print("-" * 45)
    print(f"Selector method:     {result.selection_report.method.upper()}")
    print(f"Original dimension:  {result.selection_report.original_count}")
    print(f"Selected dimensions: {result.selection_report.selected_count}")
    print(f"Selected features:   {result.selection_report.selected_features}")

    # E. Data split ratios
    print("\n5. DATASETS SPLIT SHAPES")
    print("-" * 45)
    print(f"Split Strategy:     {result.split_report.strategy.upper()}")
    print(f"Training shape:     {result.split_report.train_shape}")
    print(f"Validation shape:   {result.split_report.val_shape}")
    print(f"Test shape:         {result.split_report.test_shape}")

    # F. Saved Scikit-learn preprocessing pipeline
    print("\n6. PREPROCESSING PIPELINE PERSISTENCE")
    print("-" * 45)
    print(f"Joblib file path:   {result.pipeline_report.pipeline_filepath}")
    print(f"Pipeline steps:     {result.pipeline_report.components}")
    print("=" * 60 + "\n")

    logger.info("Phase 5 Agent Demo completed successfully!")


if __name__ == "__main__":
    run_feature_engineering_demo()
