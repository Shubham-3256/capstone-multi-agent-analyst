"""Repository for managing ExecutionLog persistence."""


from sqlalchemy.orm import Session

from src.database.models import ExecutionLog


class ExecutionLogRepository:
    """Repository implementing CRUD operations for ExecutionLog SQL entries."""

    def __init__(self, session: Session) -> None:
        """Initialize ExecutionLogRepository with a database session.

        Args:
            session: Active SQLAlchemy Session connection context.
        """
        self.session = session

    def create(self, log_record: ExecutionLog) -> ExecutionLog:
        """Persist a new execution log entry.

        Args:
            log_record: ExecutionLog model instance.

        Returns:
            ExecutionLog: Saved log instance.
        """
        self.session.add(log_record)
        self.session.flush()
        return log_record

    def get_by_id(self, log_id: str) -> ExecutionLog | None:
        """Fetch a specific execution log by ID.

        Args:
            log_id: Target UUID string.

        Returns:
            Optional[ExecutionLog]: Found log or None.
        """
        return self.session.query(ExecutionLog).filter(ExecutionLog.id == log_id).first()

    def get_by_task(self, task_name: str) -> list[ExecutionLog]:
        """Fetch all execution logs associated with a task name.

        Args:
            task_name: Task descriptor filter.

        Returns:
            List[ExecutionLog]: List of matching logs.
        """
        return self.session.query(ExecutionLog).filter(ExecutionLog.task_name == task_name).all()

    def get_all(self) -> list[ExecutionLog]:
        """Fetch all system execution logs.

        Returns:
            List[ExecutionLog]: List of all execution logs.
        """
        return self.session.query(ExecutionLog).all()

    def update_status(
        self,
        log_id: str,
        status: str,
        duration_seconds: float | None = None,
        error_message: str | None = None
    ) -> ExecutionLog | None:
        """Update active parameters and status markers on execution logs.

        Args:
            log_id: Target log UUID string.
            status: Status classification (pending, running, completed, failed).
            duration_seconds: Execution time duration in seconds.
            error_message: Failure error traceback details if failed.

        Returns:
            Optional[ExecutionLog]: Updated execution log, or None.
        """
        log_record = self.get_by_id(log_id)
        if log_record:
            log_record.status = status
            if duration_seconds is not None:
                log_record.duration_seconds = duration_seconds
            if error_message is not None:
                log_record.error_message = error_message
            self.session.flush()
        return log_record
