import concurrent.futures
import threading
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.agents.feature_engineering.agent import FeatureEngineeringAgent
from src.core.paths import Paths
from src.database.base import Base

# Thread-local storage for paths
thread_local_paths = threading.local()


class ThreadLocalPathProxy:
    """Proxy delegating path operations to a thread-local path target."""

    def __getattr__(self, name):
        actual_path = getattr(
            thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR
        )
        return getattr(actual_path, name)

    def __truediv__(self, other):
        actual_path = getattr(
            thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR
        )
        return actual_path / other

    def __fspath__(self):
        actual_path = getattr(
            thread_local_paths, "processed_dir", Paths._ORIGINAL_PROCESSED_DIR
        )
        return str(actual_path)


@pytest.fixture
def clean_db():
    """Fixture providing clean SQLite database session using a temporary file and mocking dynamic thread paths."""
    import os

    db_file = Paths.WORKSPACE_DIR / "temp_parallel_test.db"
    if db_file.exists():
        try:
            os.remove(db_file)
        except Exception:
            pass

    engine = create_engine(
        f"sqlite:///{db_file.resolve()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Enable Thread-local path proxying
    Paths._ORIGINAL_PROCESSED_DIR = Paths.PROCESSED_DIR
    Paths.PROCESSED_DIR = ThreadLocalPathProxy()

    try:
        yield session
    finally:
        # Restore original paths
        Paths.PROCESSED_DIR = Paths._ORIGINAL_PROCESSED_DIR
        session.close()
        if db_file.exists():
            try:
                os.remove(db_file)
            except Exception:
                pass
        # Clean up thread subdirectories
        import shutil

        for i in range(4):
            thread_dir = Paths.WORKSPACE_DIR / f"processed_thread_{i}"
            if thread_dir.exists():
                try:
                    shutil.rmtree(thread_dir)
                except Exception:
                    pass


def test_concurrent_agent_runs(clean_db):
    """Run multiple FeatureEngineeringAgent tasks in parallel threads to check thread safety and database concurrency locks."""
    from sqlalchemy.orm import sessionmaker

    # Get the engine and bind to a thread-local sessionmaker
    engine = clean_db.get_bind()
    SessionLocal = sessionmaker(bind=engine)

    def worker_run(idx):
        # Configure thread-local path for this thread worker
        thread_local_paths.processed_dir = (
            Paths._ORIGINAL_PROCESSED_DIR.parent / f"processed_thread_{idx}"
        )
        thread_local_paths.processed_dir.mkdir(parents=True, exist_ok=True)

        session = SessionLocal()
        try:
            import numpy as np

            # Create a unique dataframe per worker with distinct distributions to avoid duplicate file hashes after scaling
            df_unique = pd.DataFrame(
                {
                    "id": range(100),
                    "feat": np.random.RandomState(idx).randn(100) * 10.0,
                    "target": [0, 1] * 50,
                }
            )
            agent = FeatureEngineeringAgent(session)
            return agent.run(df_unique, target_column="target")
        finally:
            session.close()

    # Run 4 workers in parallel threads
    n_workers = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_run, i) for i in range(n_workers)]

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Verify all concurrent executions completed successfully
    assert len(results) == n_workers
    for res in results:
        assert res.train_filepath != ""
        assert Path(res.train_filepath).exists()
