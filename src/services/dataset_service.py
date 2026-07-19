"""Service handling high-level dataset lifecycle tasks and database registrations."""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.core.exceptions import DatasetException
from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.security import generate_sha256_hash
from src.database.models import DatasetRecord
from src.repositories.dataset_repository import DatasetRepository
from src.services.file_service import FileService
from src.tools.data_loader import DataLoader

logger = get_logger(__name__)


class DatasetService:
    """Service coordinates dataset saving, registering, loading, and deletion workflows."""

    def __init__(self, db_session: Session) -> None:
        """Initialize DatasetService with active database session connections.

        Args:
            db_session: Session interface to database.
        """
        self.db = db_session
        self.repo = DatasetRepository(db_session)
        self.file_service = FileService(Paths.WORKSPACE_DIR)

    def calculate_checksum(self, filepath: Path) -> str:
        """Calculate the SHA-256 checksum string for a file.

        Args:
            filepath: Target file path.

        Returns:
            str: 64-character hash digest.
        """
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            return generate_sha256_hash(content)
        except Exception as e:
            logger.error(f"Failed to calculate hash checksum for {filepath}: {e}")
            raise DatasetException(f"Failed to hash dataset file: {e}") from e

    def register_dataset(self, file_path: Path, filename: str) -> DatasetRecord:
        """Move a file to the uploads workspace and record it in the database tracking.

        If a file with the identical hash matches existing tracking records, the 
        pre-existing tracking details are returned immediately to prevent duplicate data.

        Args:
            file_path: Current physical location of the dataset file.
            filename: Target naming alias.

        Returns:
            DatasetRecord: Saved record tracking database instance.
        """
        logger.info(f"Registering dataset from source: {file_path}")
        if not file_path.exists() or not file_path.is_file():
            raise DatasetException(f"Target registration file does not exist or is invalid: {file_path}")

        try:
            # 1. Check sum mapping
            checksum = self.calculate_checksum(file_path)
            existing_record = self.repo.get_by_hash(checksum)

            if existing_record:
                logger.info(f"Dataset already registered with checksum: {checksum}. ID: {existing_record.id}")
                return existing_record

            # 2. Determine file properties
            file_size = file_path.stat().st_size

            # 3. Relocate file to UPLOAD_DIR
            target_path = Paths.UPLOAD_DIR / filename
            final_path = self.file_service.move_file(file_path, target_path)

            # 4. Load basic dimensions (using DataLoader helper)
            row_count, col_count = None, None
            try:
                df = DataLoader.load_file(final_path)
                row_count, col_count = df.shape
            except Exception as e:
                # Log warning but proceed with file registration
                logger.warning(f"Could not extract shape dimension properties during registration: {e}")

            # 5. Build database record
            new_record = DatasetRecord(
                filename=filename,
                filepath=str(final_path),
                file_hash=checksum,
                file_size_bytes=file_size,
                row_count=row_count,
                column_count=col_count,
                status="uploaded"
            )

            saved = self.repo.create(new_record)
            logger.info(f"Dataset registered successfully. Database ID: {saved.id}")
            return saved

        except Exception as e:
            logger.error(f"Inability to complete dataset registration: {e}")
            raise DatasetException(f"Failed to register dataset: {e}") from e

    def load_dataset_df(self, record_id: str) -> pd.DataFrame:
        """Load a registered dataset record's content into a Pandas DataFrame.

        Args:
            record_id: Database tracking record UUID.

        Returns:
            pd.DataFrame: Calculated table representation.
        """
        record = self.repo.get_by_id(record_id)
        if not record:
            raise DatasetException(f"Dataset record not found in database: {record_id}")

        file_path = Path(record.filepath)
        logger.info(f"Loading dataset into DataFrame: {file_path}")
        return DataLoader.load_file(file_path)

    def list_datasets(self) -> list[DatasetRecord]:
        """Fetch all registered dataset entries.

        Returns:
            List[DatasetRecord]: List of stored records.
        """
        return self.repo.get_all()

    def get_dataset(self, record_id: str) -> DatasetRecord | None:
        """Retrieve tracking metadata details for a registered dataset.

        Args:
            record_id: Database tracking record UUID.

        Returns:
            Optional[DatasetRecord]: Database record or None.
        """
        return self.repo.get_by_id(record_id)

    def delete_dataset(self, record_id: str) -> bool:
        """Remove a dataset's tracking records and physical file from the workspace.

        Args:
            record_id: Database tracking record UUID.

        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        logger.info(f"Initiating deletion for dataset ID: {record_id}")
        record = self.repo.get_by_id(record_id)
        if not record:
            logger.warning(f"No dataset record found to delete for ID: {record_id}")
            return False

        try:
            # 1. Delete workspace physical file
            file_path = Path(record.filepath)
            self.file_service.delete_file(file_path)

            # 2. Clear SQL persistence entries
            success = self.repo.delete(record_id)
            logger.info(f"Dataset record {record_id} deletion status: {success}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete dataset {record_id}: {e}")
            raise DatasetException(f"Error occurred during dataset deletion: {e}") from e
