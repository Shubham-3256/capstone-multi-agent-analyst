# Career Portfolio - Interview Questions & Answers

This document lists **30 technical interview questions and detailed answers** based on the architectural design and implementation details of the Multi-Agent AI Data Analyst.

---

## 1. System Architecture & Orchestration (Q1–Q6)

#### Q1: What is the benefit of using LangGraph for agent orchestration over simple sequential Python scripts?
**Answer**: LangGraph provides state management and structured directed acyclic graph (DAG) routing. Unlike linear scripts, it supports conditional edges, loops, node retries, and checkpointing. For example, if a dataset is invalid, the graph dynamically routes from the profiler to finalization, bypassing training. Checkpointing allows workflow states to be serialized to a DB, supporting pausing and resuming.

#### Q2: How is the state passed between nodes in the graph, and how do you ensure immutability?
**Answer**: The state is encapsulated in a Pydantic `WorkflowState` object. It stores paths, profiles, ML model metrics, and report outputs. Before a node runs, the state is serialized into a standard dictionary. The node executes and returns updates, which are merged back. This prevents concurrent nodes from corrupting in-memory object references.

#### Q3: How did you implement workflow checkpointing and resume capabilities?
**Answer**: We built a `FileCheckpointStore` that serializes the `WorkflowState` (omitting active in-memory pandas DataFrames) to a JSON file named `{workflow_id}.json`. In parallel, we persist metadata (execution status, logs, errors, timings) in an SQLite database using SQLAlchemy. On start, the system loads the state from the JSON checkpoint file and restarts from the last uncompleted node.

#### Q4: Why did you choose SQLite for the database instead of PostgreSQL or MongoDB?
**Answer**: SQLite was chosen for zero-dependency portability and ease of reproduction, fulfilling open-source guidelines. Since all data remains localized to the user's workspace, SQLite provides a lightweight SQL store without requiring users to configure external database servers.

#### Q5: How did you handle target column validation case-insensitivity?
**Answer**: The user might input target columns in varying cases (e.g. `Survived`, `survived`). We standardized target strings using snake_case cleaning rules. The `Validator` evaluates column existence in a case-insensitive manner, preventing index mapping failures during feature engineering.

#### Q6: How does the system handle unsupervised datasets (where no target column is provided)?
**Answer**: If target is missing, the router redirects the flow directly from `data_intelligence` to `visualization` and `report_generation`, bypassing feature engineering and ML training nodes.

---

## 2. Machine Learning & Preprocessing (Q7–Q14)

#### Q7: How does your ML pipeline prevent data leakage?
**Answer**: We fit all preprocessing steps (scaling, encoding) strictly on the training partition and apply them to the validation/test splits. The preprocessing pipeline is saved as a single Scikit-Learn `Pipeline` object via `joblib`, ensuring no feature statistics leak into validation splits.

#### Q8: What algorithms are included in your competitive training sweep?
**Answer**: XGBoost, LightGBM, CatBoost, and Scikit-Learn (Random Forest, Logistic Regression, Linear Regression, KNN).

#### Q9: How do you handle dataset-driven task classification (binary vs. multiclass vs. regression)?
**Answer**: The system evaluates target column data types and unique value counts:
*   If numeric and unique counts are > 10, it infers **Regression**.
*   If unique counts are exactly 2, it infers **Binary Classification**.
*   If unique counts are between 3 and 10, it infers **Multiclass Classification**.

#### Q10: How does your system handle extremely small datasets (< 10 rows)?
**Answer**: Standard splitting fails on small counts. We bypass train-test splits (using the full set for both) and dynamically switch from standard cross-validation to **Leave-One-Out Cross-Validation (LOOCV)**. We also scale down the KNN `n_neighbors` parameter to avoid crashes.

#### Q11: How is hyperparameter tuning performed under LOOCV constraint?
**Answer**: The tuner detects the small sample size, filters out invalid hyperparameter grids (like high `n_neighbors` for KNN), and uses LOOCV folds for grid/randomized search validation.

#### Q12: Why did you choose correlation-based feature selection?
**Answer**: Correlation-based selection identifies highly correlated target features while pruning collinear predictors, reducing feature count and preventing multi-collinearity issues.

#### Q13: What happens to categorical variables in the preprocessing step?
**Answer**: They are processed using a Scikit-Learn `OneHotEncoder` with `handle_unknown="ignore"` to prevent crashes on out-of-vocabulary values in validation splits.

#### Q14: How does the system handle numeric features with zero variance?
**Answer**: The preprocessing pipeline includes a variance selector or handles zero-variance columns during standard scaling to prevent division-by-zero errors.

---

## 3. DevOps, Containerization & Proxying (Q15–Q22)

#### Q15: Why did you use multi-stage Docker builds for the production image?
**Answer**: Multi-stage builds compile packages in a build environment (`builder`) and copy only the final virtual environment and libraries into a minimal runner image (`runner`). This keeps the production image small and secure.

#### Q16: Why did you change the base image from Python 3.10 to Python 3.12?
**Answer**: Python 3.12 offers improved interpreter speed, cleaner tracebacks, and supports modern dependency features, aligning the container with local developer environments.

#### Q17: What was the purpose of creating a separate Dockerfile.dev?
**Answer**: `Dockerfile.dev` mounts local directories for hot-reloading code changes and maps ports directly to the host, facilitating rapid development iterations.

#### Q18: What role does Nginx play in your docker compose stack?
**Answer**: Nginx acts as a reverse proxy gateway. It listens on port 80 and routes incoming traffic to the internal Streamlit service on port 8501, handling websocket handshakes, enforcing security headers, and compression.

#### Q19: Why is WebSocket proxying configured in Nginx?
**Answer**: Streamlit requires a persistent WebSocket connection to stream charts and page updates. Standard HTTP reverse proxies drop inactive channels. Configuring WebSocket headers (`Upgrade` and `Connection`) keeps the Streamlit channel active.

#### Q20: Why did you set timeouts to 600 seconds in Nginx?
**Answer**: Training ML models and calling LLMs can take several minutes. Standard Nginx read timeouts (60 seconds) would terminate connections mid-analysis. Setting it to 600s supports long-running workflows.

#### Q21: Why does Nginx have `client_max_body_size 100M`?
**Answer**: To support uploading large CSV/Excel datasets without triggering Nginx client-entity-too-large errors.

#### Q22: How is security handled for the dockerized app running as non-root?
**Answer**: We configure the runner container to run under `appuser` (UID/GID 10001) instead of root, mitigating risk if the container gets compromised.

---

## 4. Testing, CI/CD & Concurrency (Q23–Q30)

#### Q23: How did you solve database unique constraint failures in parallel tests?
**Answer**: Workers generated identical file splits concurrently, leading to identical file hashes. When inserting records, this triggered a unique constraint violation on `file_hash`. We resolved this by injecting unique random noise into dataset features for each thread worker to ensure distinct file hashes.

#### Q24: What is the ThreadLocalPathProxy, and why did you need it?
**Answer**: It is a proxy class that routes path evaluations to thread-unique directories based on `threading.current_thread()`. This prevented concurrent workers from overwriting the same `train.csv` file, avoiding file locks and corrupt hashes.

#### Q25: Why did you use a temporary file database instead of sqlite in-memory in concurrent tests?
**Answer**: SQLite in-memory databases are private to the connection that created them. SQLAlchemy opens separate connections for concurrent thread workers, leading to empty databases. Using a temporary file database allowed workers to query the same database tables.

#### Q26: What linting rules are configured in your CI pipeline?
**Answer**: We use Ruff to check for syntax errors, formatting style violations (E/W), Pyflakes errors (F), unused arguments (ARG), and print statement usage (T20).

#### Q27: How does your CI workflow handle WeasyPrint dependencies?
**Answer**: WeasyPrint requires Cairo and Pango library files. The GHA workflow installs these graphic libraries (`libcairo2`, `libpango-1.0-0`) using `apt-get` before running tests.

#### Q28: How is the benchmark smoke test verified in GHA?
**Answer**: It runs a fast run of the benchmark suite using `--smoke` mode, validating that the benchmarking script executes without errors.

#### Q29: What is checked in pre-commit hooks?
**Answer**: Black formatting, Isort imports, Ruff linting, Mypy strict type checks, YAML validation, and trailing whitespace cleanup.

#### Q30: How does the LLM caching layer save costs?
**Answer**: It saves prompt-response pairs. If a duplicate prompt is executed (e.g. during testing or page refreshes), it returns the response directly from the file cache, reducing API costs and latency.
