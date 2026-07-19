# Changelog

All notable changes to the **Multi-Agent AI Data Analyst** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-19

This release compiles the codebase into a production-ready enterprise release candidate (Phases 1–14).

### Added
- **Multi-Stage Docker Setup**: Created optimized production `Dockerfile` (multi-stage Python 3.12 runner) and local development `Dockerfile.dev`.
- **Docker Compose Stacks**: Added `docker-compose.yml` (development with hot-reloading) and `docker-compose.prod.yml` (production).
- **Nginx Reverse Proxy**: Configured secure gateway `nginx.conf` supporting Streamlit WebSocket handshakes, 100MB file uploads, gzip compression, and security headers.
- **CI/CD Workflows**: Configured GitHub Actions for linting (Ruff), formatting (Black), type validation (Mypy), and Pytest executions with coverage reports.
- **Pre-commit Hooks**: Created `.pre-commit-config.yaml` to enforce standards locally.
- **Enterprise Documentation Package**: Created 10 technical guides in `docs/` detailing system design, deployment, API schemas, and troubleshooting.
- **Mermaid Architecture Charts**: Added visual diagrams mapping workflow steps, agents, persistence layers, and gateway configs.
- **Developer Portfolio**: Added career packaging materials (`portfolio/` suite), including 30 likely interview questions and detailed answers.

### Changed
- **Pydantic Settings**: Centralized environment variables (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `CACHE_DIR`, `REPORT_OUTPUT_DIR`) inside `Settings`.
- **Log Relocation**: Redirected application-wide logs to a consolidated top-level `logs/` directory.

### Fixed
- **Streamlit Port Mappings**: Repointed Docker production entrypoints and healthchecks to Streamlit server port `8501`.
- **Precedence Errors**: Fixed integer slicing bugs in `test_large_dataset.py`.
- **Thread Concurrency**: Solved database locks and split hash collisions using thread-local sessions and a `ThreadLocalPathProxy`.
- **Mixed Casing Targets**: Standardized casing matching for target labels across profiler and ML tuner models.
