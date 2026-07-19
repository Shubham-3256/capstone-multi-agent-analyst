"""Integration tests executing complete analytics workflows on classic real-world dataset schemas."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def generate_titanic_schema() -> pd.DataFrame:
    """Generate mock Titanic schema with identical data types."""
    return pd.DataFrame(
        {
            "PassengerId": range(1, 101),
            "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 10,
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 2] * 10,
            "Name": [f"Name {i}" for i in range(100)],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
            ]
            * 10,
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0, 27.0, 14.0] * 10,
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1] * 10,
            "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * 10,
            "Ticket": [f"Ticket {i}" for i in range(100)],
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07]
            * 10,
            "Cabin": [
                "C85",
                "C123",
                np.nan,
                np.nan,
                "E46",
                np.nan,
                np.nan,
                "G6",
                "C103",
                np.nan,
            ]
            * 10,
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "C", "S"] * 10,
        }
    )


def generate_california_housing_schema() -> pd.DataFrame:
    """Generate mock California Housing schema with matching numeric column types."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "MedInc": [3.87, 8.30, 7.25, 5.64, 3.84, 4.03, 3.65, 3.12, 2.08, 3.69] * 10,
            "HouseAge": [41, 21, 52, 52, 52, 52, 52, 52, 52, 52] * 10,
            "AveRooms": [6.98, 6.23, 8.28, 5.81, 6.28, 4.76, 5.11, 5.98, 6.32, 5.02]
            * 10,
            "AveBedrms": [1.02, 0.97, 1.07, 1.07, 1.08, 1.10, 0.95, 1.05, 1.12, 1.00]
            * 10,
            "Population": [322, 2401, 496, 558, 565, 413, 1094, 1157, 432, 742] * 10,
            "AveOccup": [2.55, 2.10, 2.80, 2.54, 2.18, 2.45, 1.80, 2.20, 2.40, 2.30]
            * 10,
            "Latitude": [
                37.88,
                37.86,
                37.85,
                37.85,
                37.85,
                37.85,
                37.84,
                37.84,
                37.84,
                37.84,
            ]
            * 10,
            "Longitude": [
                -122.23,
                -122.22,
                -122.24,
                -122.25,
                -122.25,
                -122.25,
                -122.25,
                -122.25,
                -122.26,
                -122.26,
            ]
            * 10,
            "MedHouseValue": [
                452600.0,
                358500.0,
                352100.0,
                341300.0,
                342200.0,
                269700.0,
                299200.0,
                241400.0,
                226700.0,
                261100.0,
            ]
            * 10
            + np.random.randn(100) * 1000.0,
        }
    )


def generate_adult_income_schema() -> pd.DataFrame:
    """Generate mock Adult Income census columns schema."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 37, 49, 52, 31, 42] * 10,
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Self-emp-not-inc",
                "Private",
                "Private",
            ]
            * 10,
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Bachelors",
                "Masters",
                "9th",
                "HS-grad",
                "Masters",
                "Bachelors",
            ]
            * 10,
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-spouse-absent",
                "Married-civ-spouse",
                "Never-married",
                "Married-civ-spouse",
            ]
            * 10,
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
                "Exec-managerial",
                "Other-service",
                "Exec-managerial",
                "Prof-specialty",
                "Exec-managerial",
            ]
            * 10,
            "race": [
                "White",
                "White",
                "White",
                "Black",
                "Black",
                "White",
                "Black",
                "White",
                "White",
                "White",
            ]
            * 10,
            "gender": [
                "Male",
                "Male",
                "Male",
                "Male",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Male",
            ]
            * 10,
            "hours-per-week": [40, 13, 40, 40, 40, 40, 16, 45, 50, 40] * 10,
            "income": [
                "<=50K",
                ">50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                ">50K",
                ">50K",
            ]
            * 10,
        }
    )


def generate_heart_disease_schema() -> pd.DataFrame:
    """Generate mock Heart Disease variables schema."""
    return pd.DataFrame(
        {
            "age": [63, 37, 41, 56, 57, 57, 56, 44, 52, 57] * 10,
            "sex": [1, 1, 0, 1, 0, 1, 0, 1, 1, 1] * 10,
            "cp": [3, 2, 1, 1, 0, 0, 1, 1, 2, 2] * 10,
            "trestbps": [145, 130, 130, 120, 120, 140, 140, 120, 172, 150] * 10,
            "chol": [233, 250, 204, 236, 354, 192, 294, 263, 199, 168] * 10,
            "fbs": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0] * 10,
            "target": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10,  # Classification targets
        }
    )


def run_pipeline_on_dataset(df, target_col, clean_db, filename):
    """Run E2E pipeline and verify assertions."""
    from src.core.paths import Paths

    temp_dir = Paths.WORKSPACE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = temp_dir / filename
    df.to_csv(dataset_file, index=False)

    graph = WorkflowGraph(
        config=WorkflowConfig(checkpoint_mode="none", persist_execution=False)
    )
    graph.db = clean_db
    graph.nodes.callbacks = graph._default_callbacks()

    result = graph.run(str(dataset_file), target_column=target_col)

    assert result.is_success is True
    assert result.state.ml_result is not None
    assert result.state.report_result is not None
    assert result.state.report_result.is_success is True


def test_real_dataset_titanic(clean_db):
    """Run validation pipeline on Titanic schema."""
    df = generate_titanic_schema()
    run_pipeline_on_dataset(df, "Survived", clean_db, "titanic_dataset.csv")


def test_real_dataset_california_housing(clean_db):
    """Run validation pipeline on California Housing schema."""
    df = generate_california_housing_schema()
    run_pipeline_on_dataset(df, "MedHouseValue", clean_db, "california_housing.csv")


def test_real_dataset_breast_cancer(clean_db):
    """Run validation pipeline on scikit-learn built-in Breast Cancer dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    # Subsample to keep test fast and mix classes
    df = df.sample(n=80, random_state=42)
    run_pipeline_on_dataset(df, "target", clean_db, "breast_cancer.csv")


def test_real_dataset_wine_quality(clean_db):
    """Run validation pipeline on scikit-learn built-in Wine Quality (multiclass) dataset."""
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    # Subsample to mix classes
    df = df.sample(n=80, random_state=42)
    run_pipeline_on_dataset(df, "target", clean_db, "wine_quality.csv")


def test_real_dataset_adult_income(clean_db):
    """Run validation pipeline on Adult Income schema."""
    df = generate_adult_income_schema()
    run_pipeline_on_dataset(df, "income", clean_db, "adult_income.csv")


def test_real_dataset_heart_disease(clean_db):
    """Run validation pipeline on Heart Disease schema."""
    df = generate_heart_disease_schema()
    # Add varying target class to support classification fit
    df.iloc[0:10, df.columns.get_loc("target")] = 0
    run_pipeline_on_dataset(df, "target", clean_db, "heart_disease.csv")
