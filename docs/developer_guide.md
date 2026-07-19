# Developer Guide

Welcome to the Multi-Agent AI Data Analyst development manual. Follow these practices and guidelines to keep our codebase correct, clean, and enterprise-ready.

---

## 1. Code Style and Quality Standards

We enforce strict quality tooling to maintain readability and prevent runtime type safety issues:

*   **Code Formatter**: [Black](https://github.com/psf/black) (line length: 88, target Python 3.12).
*   **Linter & Import Sorter**: [Ruff](https://github.com/astral-sh/ruff) (configured with Pyflakes, pycodestyle, isort, and flake8-bugbear).
*   **Static Type Checker**: [Mypy](https://github.com/python/mypy) (`strict = true` mode enabled).

---

## 2. Setting Up Pre-Commit Hooks

Before editing or committing any code, configure the pre-commit hooks to run checks on staged files:

```bash
# Install pre-commit tool
pip install pre-commit

# Register hooks with your local git config
pre-commit install
```

Once installed, committing code automatically runs style formatting, yaml validation, and static type checks. To run them manually:
```bash
pre-commit run --all-files
```

---

## 3. Running and Writing Tests

### Test Organization
All tests are located in `tests/`:
*   `tests/test_*.py`: Standard unit tests verifying class-level math, routers, and templates.
*   `tests/integration/`: End-to-end tests validating pipelines against synthetic or mock datasets.
*   `tests/performance/`: Scalability, memory footprint, cache performance, and parallel execution checks.

### Running Pytest
```bash
# Execute all tests
python -m pytest

# Run a specific test suite with stdout output
python -m pytest tests/integration/test_database.py -s

# Calculate test coverage
python -m pytest --cov=src --cov-report=html
```

---

## 4. How to Implement a New Agent

To introduce a new agent to the workflow graph:

1.  **Define Agent Schema**: Create your models in a new file `src/agents/my_agent/models.py` inheriting from `pydantic.BaseModel`.
2.  **Define Agent Class**: Implement the core agent logic inside `src/agents/my_agent/agent.py` receiving a SQLAlchemy Session in `__init__`.
3.  **Define Orchestration State**: Add your agent's result field to the `WorkflowState` class in `src/orchestration/state.py`.
4.  **Register Node**: Register the agent node and routing logic in `WorkflowNodes` (`src/orchestration/nodes.py`) and compile it inside `WorkflowGraph` (`src/orchestration/graph.py`).
5.  **Write Tests**: Add a unit test verifying node execution under `tests/`.
