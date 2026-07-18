"""Read-only history queries for dashboard presentation."""

from typing import Any, List, Optional

from src.database.database import DatabaseManager
from src.database.models import DatasetRecord, ReportRecord, WorkflowExecution


from types import SimpleNamespace


def _map_workflow(w: Any) -> Optional[SimpleNamespace]:
    if w is None:
        return None
    return SimpleNamespace(
        workflow_id=w.workflow_id,
        status=w.status,
        history_json=w.history_json,
        errors_json=w.errors_json,
        timing_json=w.timing_json,
        created_at=w.created_at
    )


def _map_report(r: Any) -> Optional[SimpleNamespace]:
    if r is None:
        return None
    return SimpleNamespace(
        report_id=r.report_id,
        title=r.title,
        template_type=r.template_type,
        dataset_hash=r.dataset_hash,
        manifest_json=r.manifest_json,
        output_paths_json=r.output_paths_json,
        duration_seconds=r.duration_seconds,
        created_at=r.created_at
    )


def _map_dataset(d: Any) -> Optional[SimpleNamespace]:
    if d is None:
        return None
    return SimpleNamespace(
        id=d.id,
        filename=d.filename,
        filepath=d.filepath,
        file_hash=d.file_hash,
        file_size_bytes=d.file_size_bytes,
        row_count=d.row_count,
        column_count=d.column_count,
        status=d.status,
        created_at=d.created_at
    )


class HistoryService:
    """Provides workflow and report history without workflow-side mutations."""

    @staticmethod
    def workflows(status_filter: Optional[str] = None, search_query: Optional[str] = None) -> List[Any]:
        """Return latest persisted workflow executions with optional filtering."""
        with DatabaseManager.get_session() as session:
            query = session.query(WorkflowExecution)
            if status_filter and status_filter != "All":
                query = query.filter(WorkflowExecution.status == status_filter.lower())
            if search_query:
                query = query.filter(
                    (WorkflowExecution.workflow_id.contains(search_query)) |
                    (WorkflowExecution.errors_json.contains(search_query))
                )
            results = query.order_by(WorkflowExecution.created_at.desc()).all()
            return [_map_workflow(w) for w in results if w is not None]

    @staticmethod
    def reports() -> List[Any]:
        """Return latest persisted report records."""
        with DatabaseManager.get_session() as session:
            results = session.query(ReportRecord).order_by(ReportRecord.created_at.desc()).all()
            return [_map_report(r) for r in results if r is not None]

    @staticmethod
    def datasets() -> List[Any]:
        """Return latest persisted dataset records."""
        with DatabaseManager.get_session() as session:
            results = session.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()
            return [_map_dataset(d) for d in results if d is not None]

    @staticmethod
    def workflow_by_id(workflow_id: str) -> Optional[Any]:
        """Fetch a specific workflow execution by ID."""
        with DatabaseManager.get_session() as session:
            w = session.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id).first()
            return _map_workflow(w)

    @staticmethod
    def report_by_id(report_id: str) -> Optional[Any]:
        """Fetch a specific report record by ID."""
        with DatabaseManager.get_session() as session:
            r = session.query(ReportRecord).filter(ReportRecord.report_id == report_id).first()
            return _map_report(r)


