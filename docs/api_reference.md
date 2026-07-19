# API Reference

This document describes the primary programmatic interfaces, agent call signatures, and orchestration schemas.

---

## 1. Orchestration Graph Runner

### `WorkflowGraph`
Coordinates the end-to-end execution of specialized analysis nodes.

```python
from src.orchestration.graph import WorkflowGraph
from src.orchestration.config import WorkflowConfig

# Instantiate runner
config = WorkflowConfig(checkpoint_mode="file", persist_execution=True)
graph = WorkflowGraph(config=config, db=session)

# Execute pipeline run
result = graph.run(
    dataset_path="workspace/uploads/my_data.csv",
    target_column="label",
    workflow_id="wf-uuid-123"
)
```

#### Run Parameters
*   `dataset_path` (str): Path to raw file in local filesystem.
*   `target_column` (str, optional): Target header to fit models against.
*   `workflow_id` (str, optional): Unique ID for logging and checkpoint loading.

#### Return Value (`WorkflowResult`)
*   `is_success` (bool): `True` if all nodes completed.
*   `state` (WorkflowState): Complete state payload (features, models, visualizations, reports).
*   `output_paths` (dict[str, str]): Output file paths (PDF, DOCX, HTML).

---

## 2. Agent Interfaces

Each agent exposes a standard programmatic entrypoint:

### `DataIntelligenceAgent.run(filepath: str, target: str | None = None) -> DataIntelligenceResult`
Analyzes raw file layout, types, duplicates, and missing values.

### `FeatureEngineeringAgent.run(df: pd.DataFrame, target_column: str) -> FeatureEngineeringResult`
Handles categorical encoding, feature scaling, and high-correlation column selections.

### `MachineLearningAgent.run(train_df: pd.DataFrame, target: str, val_df: pd.DataFrame, val_target: str) -> MachineLearningResult`
Splits datasets, executes hyperparameter grid searches, trains competitive models, and returns validation scores.

### `VisualizationAgent.run(profile: DatasetProfile, feature: FeatureEngineeringResult, ml: MachineLearningResult, target_column: str) -> VisualizationResult`
Generates Matplotlib and Plotly figures, writing image files to disk.

### `BusinessInsightAgent.run(profile: DatasetProfile, feature: FeatureEngineeringResult, ml: MachineLearningResult, viz: VisualizationResult) -> BusinessInsightResult`
Calls LLMs to interpret statistical findings and outputs high-level text summaries.

---

## 3. Database Repositories

Database actions are centralized in repository wrappers:
*   `DatasetRepository(session).create(record: DatasetRecord)`: Logs files in DB.
*   `WorkflowRepository(session).get_by_id(wf_id: str)`: Retrieves run checkpoint histories.
