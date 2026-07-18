"""Unit tests for Preprocessing Pipeline building and serialization."""

import pandas as pd
from pathlib import Path
from src.agents.feature_engineering.pipeline import PipelineBuilder


def test_pipeline_builder_lifecycle(tmp_path):
    """Test assembling preprocessor pipeline, fitting, saving, and loading it back."""
    df = pd.DataFrame({
        "age": [25, 45, 30, 22, 38],
        "city": ["NY", "LA", "NY", "CH", "LA"],
        "target": [0, 1, 0, 1, 0]
    })
    
    # Fit pipeline
    pipeline = PipelineBuilder.build_and_fit(
        df_train=df,
        y_train=df["target"],
        target_column="target"
    )
    
    # Save pipeline
    filepath = tmp_path / "pipeline.joblib"
    report = PipelineBuilder.save_pipeline(pipeline, filepath)
    
    assert filepath.exists()
    assert "generator" in report.components
    assert "encoder" in report.components
    assert "scaler" in report.components
    
    # Load pipeline back
    loaded_pipeline = PipelineBuilder.load_pipeline(filepath)
    features_df = df.drop(columns=["target"])
    transformed_df = loaded_pipeline.transform(features_df)
    
    assert transformed_df.shape[0] == 5
    assert transformed_df.shape[1] > 0
