"""Demo script for Phase 3 - Backend Foundation components validation.

Demonstrates database setup, dataset registration, metadata compilation,
file workspace transitions, profiling metrics, and diagnostics.
"""

import json
import sys
from pathlib import Path

# Add project root directory to path to enable local package importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core import get_logger
from src.database import DatabaseManager, init_db
from src.services.dataset_service import DatasetService
from src.services.metadata_service import MetadataService
from src.tools.dataset_profiler import DatasetProfiler
from src.tools.file_manager import FileManager

logger = get_logger("demo_phase3")


def create_sample_dataset() -> Path:
    """Create a sample CSV dataset to run the demo process.

    Returns:
        Path: Location of the newly generated temporary CSV file.
    """
    sample_file = project_root / "workspace" / "temp_demo_churn.csv"
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    # 5 rows of customer data
    content = """customer_id,age,tenure,monthly_charges,churn
CUST001,34,12,65.5,0
CUST002,45,24,80.0,1
CUST003,23,2,35.4,0
CUST004,56,48,110.25,1
CUST005,38,18,70.1,0
"""
    sample_file.write_text(content, encoding="utf-8")
    logger.info(f"Generated sample churn dataset for demo at: {sample_file}")
    return sample_file


def run_demo() -> None:
    """Run the backend foundation pipeline execution demo."""
    logger.info("==========================================================")
    logger.info("Starting Phase 3 Backend Foundation Demo Pipeline")
    logger.info("==========================================================")

    # 1. Initialize relational database schemas
    init_db()

    # 2. Setup mock data file
    sample_file = create_sample_dataset()

    # 3. Ingest and register dataset through service layers
    # Use transactional sessions managed by DatabaseManager
    with DatabaseManager.get_session() as session:
        service = DatasetService(session)
        logger.info("Registering and moving dataset to workspace...")
        record = service.register_dataset(sample_file, "customer_churn.csv")
        # Capture parameters while session is active to avoid DetachedInstanceError
        filepath = Path(record.filepath)
        record_id = record.id
        record_filename = record.filename
        record_hash = record.file_hash
        record_size = record.file_size_bytes
        record_status = record.status

        # 4. Extract structural metadata
        logger.info("Extracting detailed dataset schema and column metadata...")
        df = service.load_dataset_df(record_id)
        metadata = MetadataService.extract_metadata(
            df=df,
            dataset_id=record_id,
            filename=record_filename,
            filepath=str(filepath),
            file_hash=record_hash,
            file_size_bytes=record_size,
            status=record_status,
        )
        logger.info(
            f"Metadata Profile generated. Columns found: {list(metadata.columns.keys())}"
        )

    # 5. Generate and print dataset summary via Profiler tool
    logger.info("Running dataset profiling report tool...")
    summary = DatasetProfiler.profile_file(filepath)

    # Output formatted results
    print("\n" + "=" * 60)
    print("PROFILED DATASET SUMMARY (JSON REPRESENTATION)")
    print("=" * 60)
    print(summary.model_dump_json(indent=4))
    print("=" * 60 + "\n")

    # 6. Audit workspace usage
    disk_summary = FileManager.get_workspace_disk_summary()
    logger.info(
        f"Workspace Storage Audit Summary: {json.dumps(disk_summary, indent=2)}"
    )

    # Clean up temp directories
    FileManager.clear_temp_workspace()

    logger.info("==========================================================")
    logger.info("Phase 3 Demo completed successfully!")
    logger.info("==========================================================")


if __name__ == "__main__":
    run_demo()
