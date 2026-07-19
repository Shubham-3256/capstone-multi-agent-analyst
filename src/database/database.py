"""Database connection and initialization manager.

Coordinates engine construction, connection sessions, and tables setup.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.logger import get_logger
from src.core.settings import settings
from src.database.base import Base

logger = get_logger(__name__)

# Determine if we're using SQLite to configure specific pool behavior
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Configure database connection engine
# SQLite requires check_same_thread=False for multi-threaded/async workflows
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # SQL statements logging
)

# Config session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DatabaseManager:
    """Enterprise database manager managing connection lifecycles and tables creation."""

    @staticmethod
    def initialize_db() -> None:
        """Create all configured tables in the database schema.

        Should be called at application startup.
        """
        logger.info("Initializing database schemas...")
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database schemas initialized successfully.")
        except Exception as e:
            logger.critical(f"Database initialization failed: {e}", exc_info=True)
            raise

    @staticmethod
    @contextmanager
    def get_session() -> Generator[Session, None, None]:
        """Provide a transactional database session scope as a context manager.

        Yields:
            Session: Active database session.
        """
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction failed. Rolling back session. Error: {e}")
            raise
        finally:
            session.close()


def init_db() -> None:
    """Global helper interface to initialize database schema."""
    DatabaseManager.initialize_db()
