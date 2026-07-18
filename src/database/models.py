"""SQLAlchemy database models for application logging and dataset management."""

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from src.core.security import generate_secure_id
from src.database.base import Base


class DatasetRecord(Base):
    """ORM representation tracking uploaded dataset metadata within the workspace."""

    __tablename__ = "dataset_records"

    id = Column(String(36), primary_key=True, default=generate_secure_id)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    status = Column(String(50), default="uploaded", nullable=False)  # uploaded, cleaned, analyzed, failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Convert ORM object to a flat dictionary payload."""
        return {
            "id": self.id,
            "filename": self.filename,
            "filepath": self.filepath,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ExecutionLog(Base):
    """ORM representation capturing logs for agent task execution tracking."""

    __tablename__ = "execution_logs"

    id = Column(String(36), primary_key=True, default=generate_secure_id)
    task_name = Column(String(255), nullable=False)
    agent_name = Column(String(255), nullable=True)
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    parameters = Column(Text, nullable=True)  # JSON-serialized parameter payload
    error_message = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Convert ORM object to a dictionary representation."""
        return {
            "id": self.id,
            "task_name": self.task_name,
            "agent_name": self.agent_name,
            "status": self.status,
            "parameters": self.parameters,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ReportRecord(Base):
    """ORM representation of generated report metadata and output artifacts."""

    __tablename__ = "report_records"

    id = Column(String(36), primary_key=True, default=generate_secure_id)
    report_id = Column(String(36), nullable=False, unique=True, index=True)
    title = Column(String(255), nullable=False)
    template_type = Column(String(50), nullable=False)
    dataset_hash = Column(String(64), nullable=False)
    manifest_json = Column(Text, nullable=False)
    output_paths_json = Column(Text, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class WorkflowExecution(Base):
    """ORM representation of a LangGraph workflow execution audit record."""

    __tablename__ = "workflow_executions"

    id = Column(String(36), primary_key=True, default=generate_secure_id)
    workflow_id = Column(String(36), nullable=False, unique=True, index=True)
    status = Column(String(50), nullable=False)
    history_json = Column(Text, nullable=False)
    errors_json = Column(Text, nullable=False)
    timing_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
