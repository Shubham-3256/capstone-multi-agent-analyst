# Career Portfolio - Technical Highlights

This document details deep-dive engineering patterns and challenges solved during the release lifecycle.

---

## 1. Concurrency Isolation: `ThreadLocalPathProxy`

### The Challenge
During multi-threaded performance testing (`test_parallel_execution.py`), concurrent worker threads ran the `FeatureEngineeringAgent` in parallel. Since the destination directory `Paths.PROCESSED_DIR` was defined as a static global class attribute, workers concurrently wrote splits (e.g. `train.csv`) to the exact same file path. This caused file-write lock crashes, race conditions, and `UNIQUE` database constraint failures because identical files generated identical file hashes.

### The Solution
We implemented a dynamic thread-local path proxy that intercepts all file path evaluations:
```python
import threading
from pathlib import Path
from src.core.paths import Paths

thread_local_paths = threading.local()

class ThreadLocalPathProxy:
    """Proxy class that delegates directory operations to a thread-unique target folder."""
    
    def __getattr__(self, name):
        actual_path = getattr(thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR)
        return getattr(actual_path, name)
        
    def __truediv__(self, other):
        actual_path = getattr(thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR)
        return actual_path / other

    def __fspath__(self):
        actual_path = getattr(thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR)
        return str(actual_path)
```
During concurrent tests, we monkey-patch `Paths.PROCESSED_DIR = ThreadLocalPathProxy()`. When a thread runs, it sets `thread_local_paths.processed_dir = ... / f"processed_thread_{idx}"`. The agent code functions normally, unaware that its file output has been thread-isolated!

---

## 2. SQLite Connection Pooling & In-Memory Isolation

### The Challenge
By default, SQLAlchemy connection pools open separate connections for different threads. In SQLite, an in-memory database (`sqlite:///:memory:`) is private to the connection that created it. Consequently, worker threads query empty databases where `execution_logs` does not exist.

### The Solution
For multi-threaded test runs, we configure connection strings to point to a temporary local file database (e.g. `temp_parallel_test.db`), clean it up during test teardown, and configure the engine with:
```python
connect_args={"check_same_thread": False}
```
This enables multi-threaded connections to query the same DB schema safely.

---

## 3. Dynamic LOOCV for Resilient Edge-Case Modeling

For small datasets (fewer than 10 rows), standard 5-fold cross-validation fails. We implemented an automatic switch to **Leave-One-Out Cross-Validation (LOOCV)** in `CrossValidator`:
```python
if len(X) < cv_folds:
    # Switch to Leave-One-Out validator
    cv = LeaveOneOut()
```
This guarantees model training succeeds without crashes, even for datasets with only 4 samples.
