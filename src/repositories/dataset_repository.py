"""Repository for managing DatasetRecord persistence."""


from sqlalchemy.orm import Session

from src.database.models import DatasetRecord


class DatasetRepository:
    """Repository implementing CRUD operations for DatasetRecord SQL entries."""

    def __init__(self, session: Session) -> None:
        """Initialize DatasetRepository with a database session.

        Args:
            session: Active SQLAlchemy Session connection context.
        """
        self.session = session

    def create(self, record: DatasetRecord) -> DatasetRecord:
        """Save a new DatasetRecord to the database.

        Args:
            record: DatasetRecord database model instance.

        Returns:
            DatasetRecord: Saved database model instance.
        """
        self.session.add(record)
        self.session.flush()  # Populates ID and generated fields
        return record

    def get_by_id(self, record_id: str) -> DatasetRecord | None:
        """Retrieve a DatasetRecord by its UUID identifier.

        Args:
            record_id: Target UUID string.

        Returns:
            Optional[DatasetRecord]: Found record or None.
        """
        return self.session.query(DatasetRecord).filter(DatasetRecord.id == record_id).first()

    def get_by_hash(self, file_hash: str) -> DatasetRecord | None:
        """Find a dataset by its SHA-256 checksum string.

        Args:
            file_hash: The target file content checksum.

        Returns:
            Optional[DatasetRecord]: Matching record or None.
        """
        return self.session.query(DatasetRecord).filter(DatasetRecord.file_hash == file_hash).first()

    def get_all(self) -> list[DatasetRecord]:
        """Fetch all tracked datasets.

        Returns:
            List[DatasetRecord]: List of all dataset records.
        """
        return self.session.query(DatasetRecord).all()

    def update_status(self, record_id: str, status: str) -> DatasetRecord | None:
        """Update the tracking status of a specific dataset record.

        Args:
            record_id: Target UUID string.
            status: Ingestion status to apply (e.g. uploaded, cleaned, analyzed, failed).

        Returns:
            Optional[DatasetRecord]: Updated record if found, else None.
        """
        record = self.get_by_id(record_id)
        if record:
            record.status = status
            self.session.flush()
        return record

    def delete(self, record_id: str) -> bool:
        """Delete a dataset entry from persistence checks.

        Args:
            record_id: Target UUID string.

        Returns:
            bool: True if deletion succeeded, False if record was not found.
        """
        record = self.get_by_id(record_id)
        if record:
            self.session.delete(record)
            self.session.flush()
            return True
        return False
