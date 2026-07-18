"""Demo script for Phase 6 - Machine Learning Agent AutoML training sweeps validation."""

import sys
from pathlib import Path
import pandas as pd

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core import get_logger
from src.database import init_db, DatabaseManager
from src.agents.machine_learning.agent import MachineLearningAgent
from src.agents.machine_learning.persistence import Persistence

logger = get_logger("demo_phase6")


def create_demo_engineered_dataset():
    """Create sample engineered training and validation features mimicking Phase 5 outputs.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    # 20 rows of customer training data
    train_data = {
        "age": [0.25, 0.45, 0.15, 0.65, 0.35, 0.20, 0.70, 0.50, 0.30, 0.55, 0.60, 0.10, 0.40, 0.80, 0.38, 0.48, 0.28, 0.68, 0.18, 0.58],
        "monthly_charges": [0.65, 0.80, 0.35, 0.90, 0.70, 0.55, 0.95, 0.85, 0.45, 0.78, 0.90, 0.40, 0.60, 0.98, 0.68, 0.78, 0.58, 0.88, 0.38, 0.78],
        "city_london": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "city_ny": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    train_y = pd.Series([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])

    # 5 rows of customer validation data
    val_data = {
        "age": [0.30, 0.62, 0.40, 0.22, 0.72],
        "monthly_charges": [0.60, 0.90, 0.65, 0.42, 0.92],
        "city_london": [0, 1, 0, 1, 0],
        "city_ny": [1, 0, 0, 0, 1]
    }
    val_y = pd.Series([0, 1, 0, 0, 1])

    return pd.DataFrame(train_data), train_y, pd.DataFrame(val_data), val_y


def run_machine_learning_demo() -> None:
    """Run AutoML model training sweeps, rank leaderboards, reload and predict."""
    logger.info("==========================================================")
    logger.info("Starting Phase 6 Machine Learning Agent Demo AutoML Run")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Get mock engineered splits
    X_train, y_train, X_val, y_val = create_demo_engineered_dataset()
    logger.info(f"Loaded engineered splits. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 3. Run Pre-processing and Model fitting sweeps via Agent
    with DatabaseManager.get_session() as session:
        agent = MachineLearningAgent(session)
        
        # Executes: Detect Task -> Fit candidates -> Rank leaderboard -> Explain best -> Save artifacts
        result = agent.run(
            train_data=X_train,
            train_target=y_train,
            validation_data=X_val,
            validation_target=y_val
        )

    # 4. Print structured results
    print("\n" + "=" * 60)
    print("AUTOML MACHINE LEARNING PIPELINE LEADERBOARD")
    print("=" * 60)
    print(f"Task Inferred:      {result.task_report.task_type.upper()}")
    print(f"Total Sweep Time:   {round(result.duration_seconds, 4)} seconds")
    print(f"Leaderboard Entries:")
    for entry in result.leaderboard.entries:
        print(f"  Rank {entry.rank}: {entry.model_name.ljust(22)} | Score: {round(entry.score, 4)} | Metrics: {entry.metrics}")
    print("=" * 60)

    # 5. Save best model detail
    print("\n" + "=" * 60)
    print("BEST PERFORMING ESTIMATOR")
    print("=" * 60)
    print(f"Best Algorithm:     {result.best_model_name}")
    print(f"Best metrics:       {result.best_metrics}")
    print(f"Saved joblib path:  {result.best_model_path}")
    print("=" * 60)

    # 6. Feature Importances
    print("\n1. FEATURE IMPORTANCE COEFFICIENTS")
    print("-" * 45)
    for idx, item in enumerate(result.feature_importances[:5]):
        print(f"  * Rank {idx+1}: {item.column.ljust(18)} Importance: {item.importance}")

    # 7. Reload and Predict on sample row
    print("\n2. VERIFYING RELOADED MODEL FOR PREDICTIONS")
    print("-" * 45)
    loaded_best_model = Persistence.load_model(Path(result.best_model_path))
    
    sample_row = X_val.iloc[[0]]
    prediction = loaded_best_model.predict(sample_row)[0]
    probabilities = loaded_best_model.predict_proba(sample_row)[0] if hasattr(loaded_best_model, "predict_proba") else None
    
    print(f"  * Sample input row:   \n{sample_row.to_dict(orient='records')[0]}")
    print(f"  * Ground truth:       {y_val.iloc[0]}")
    print(f"  * Predicted class:    {prediction}")
    if probabilities is not None:
        print(f"  * Predicted probas:   {list(probabilities)}")
    print("=" * 60 + "\n")

    logger.info("Phase 6 Agent Demo completed successfully!")


if __name__ == "__main__":
    run_machine_learning_demo()
