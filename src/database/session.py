"""Session dependency utilities for SQLAlchemy connections."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from src.database.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Dependency provider for database sessions.

    Ensures that connections are closed immediately following completion.

    Yields:
        Session: Database connection context.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
