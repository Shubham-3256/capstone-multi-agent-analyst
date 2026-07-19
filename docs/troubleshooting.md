# Troubleshooting Guide

This guide helps you diagnose and resolve common issues and runtime errors.

---

## 1. Auditing System Logs

The system outputs detailed trace metrics with daily rotating file hooks.
*   **Log Location**: Look at [logs/app.log](file:///d:/Final%20Project/capstone-multi-agent-analyst/logs/app.log).
*   **Adjusting Log Level**: Edit the `LOG_LEVEL` value in your `.env` to `DEBUG` to see detailed execution logs.

---

## 2. Common Errors and Resolutions

### 1. `UNIQUE constraint failed: dataset_records.file_hash`
*   **Symptom**: Triggered during dataset upload or split registration.
*   **Cause**: You are attempting to register a dataset or data split that has identical contents to an existing DB entry, causing a hash collision.
*   **Resolution**: Clean the database records or upload a dataset with unique contents. In unit testing, generate random feature noise per concurrent worker to avoid split hash collisions.

### 2. `ValueError: n_neighbors is larger than n_samples`
*   **Symptom**: Model training crashes on KNN classification.
*   **Cause**: The dataset split has fewer samples than the default KNN parameter (e.g. 5 neighbors).
*   **Resolution**: The pipeline should automatically adjust `n_neighbors = min(5, len(train_df))`. Verify that you haven't forced a static hyperparameters config in settings.

### 3. `sqlite3.OperationalError: no such table: execution_logs`
*   **Symptom**: Database query fails.
*   **Cause**: Database schema was not initialized or tables were deleted.
*   **Resolution**: The application automatically creates all tables on startup via `Base.metadata.create_all(bind=engine)`. If you are using a shared SQLite in-memory instance, ensure you use a file-based temporary database for concurrent multi-session tests.
