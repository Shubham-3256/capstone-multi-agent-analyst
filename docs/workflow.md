# Workflow & Checkpointing

This document describes state propagation, graph routing, and durable checkpoint recovery.

---

## 1. State Propagation

The orchestration graph uses a centralized `WorkflowState` object, which is passed between nodes.
*   **State Accumulation**: Each node consumes the state, executes its agent routine, and updates relevant fields (e.g. `ml_result` or `report_result`).
*   **Immutability and Serialization**: To prevent memory corruption in parallel execution, the graph state is fully serialized/deserialized across nodes.

---

## 2. Dynamic Graph Routing

Decisions on workflow routing are defined in `WorkflowRouter`:

1.  **Ingestion Invalidation**: After `data_intelligence`, the system inspects the dataset profile. If the dataset format is invalid, missing critical columns, or has insufficient samples (< 4 rows), the router jumps to `finalize`, bypassing feature engineering and model training.
2.  **Unsupervised Fallback**: If the target column is missing or target-casing matches are not found, the router routes to visualizations and reports directly, skipping supervised ML.

---

## 3. Durable Checkpointing & Recovery

The platform supports two checkpointer modes:
1.  **Memory Checkpointer**: Stores checkpoints in-memory using LangGraph `MemorySaver` (ideal for unit testing).
2.  **File Checkpointer**: Saves serialized `WorkflowState` JSON files inside the workspace (`workspace/checkpoints/`).

### How Workflow Resume Works
1.  When `WorkflowGraph.run` is called with a specific `workflow_id`, the graph loads the state snapshot from `FileCheckpointStore.load(workflow_id)`.
2.  If a state snapshot exists, it returns the results immediately, bypassing rerun execution paths.
3.  If no file checkpoint exists on disk, it executes the nodes from the last paused node state registered in the SQLite `workflow_executions` table, allowing recovery from failures or user-initiated pauses.
