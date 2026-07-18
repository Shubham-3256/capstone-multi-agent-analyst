"""Workspace operations manager tool for cleanup, archiving, and storage summaries."""

import shutil
from pathlib import Path
from typing import Dict, Any

from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.exceptions import ProjectException

logger = get_logger(__name__)


class FileManager:
    """Workspace orchestration tool to audit disk usage, execute cleanups, and manage archives."""

    @staticmethod
    def get_workspace_disk_summary() -> Dict[str, Any]:
        """Audit the size and file count of all folders in the workspace.

        Returns:
            Dict[str, Any]: Mapping of subfolder names to stats (bytes size, file counts).
        """
        logger.info("FileManager: Performing workspace storage audit...")
        
        directories = {
            "uploads": Paths.UPLOAD_DIR,
            "cleaned": Paths.CLEANED_DIR,
            "processed": Paths.PROCESSED_DIR,
            "plots": Paths.PLOTS_DIR,
            "reports": Paths.REPORTS_DIR,
            "models": Paths.MODELS_DIR,
            "logs": Paths.LOGS_DIR,
            "data": Paths.DATA_DIR,
            "artifacts": Paths.ARTIFACTS_DIR
        }

        summary: Dict[str, Any] = {}
        total_size = 0
        total_files = 0

        for key, path in directories.items():
            if not path.exists():
                summary[key] = {"bytes_size": 0, "file_count": 0}
                continue
                
            files = [f for f in path.glob("**/*") if f.is_file() and f.name != ".gitkeep"]
            size = sum(f.stat().st_size for f in files)
            summary[key] = {
                "bytes_size": size,
                "file_count": len(files)
            }
            total_size += size
            total_files += len(files)

        summary["total"] = {
            "bytes_size": total_size,
            "file_count": total_files
        }
        
        logger.info(f"Storage audit complete: {total_files} active files, {total_size} total bytes.")
        return summary

    @staticmethod
    def clear_temp_workspace() -> int:
        """Evict all temporary folders and files residing in the workspace/temp directory.

        Returns:
            int: Number of deleted temporary files/directories.
        """
        temp_dir = Paths.WORKSPACE_DIR / "temp"
        logger.info(f"FileManager: Clearing temporary files from: {temp_dir}")
        if not temp_dir.exists():
            return 0

        deleted_count = 0
        for item in temp_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to clear temp item {item.name}: {e}")
                
        logger.info(f"FileManager: Cleaned {deleted_count} items from temp directory.")
        return deleted_count

    @staticmethod
    def archive_processed_data(dataset_name: str) -> Path:
        """Move output data files to a compressed archive ZIP inside workspace/artifacts.

        Args:
            dataset_name: Name of the dataset identifier to group under.

        Returns:
            Path: Path to the generated archive file.
        """
        archive_dir = Paths.ARTIFACTS_DIR / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        archive_file = archive_dir / f"{dataset_name}_archive"
        logger.info(f"FileManager: Archiving outputs to {archive_file}.zip")
        
        try:
            # Archive uploads and cleaned dirs for this dataset
            # We construct a temporary folder, copy matched files, and zip it
            temp_archive_src = Paths.WORKSPACE_DIR / "temp" / f"arch_{dataset_name}"
            temp_archive_src.mkdir(parents=True, exist_ok=True)
            
            # Match files in uploads and cleaned
            files_to_archive = []
            for directory in [Paths.UPLOAD_DIR, Paths.CLEANED_DIR, Paths.PLOTS_DIR, Paths.REPORTS_DIR]:
                if directory.exists():
                    files_to_archive.extend(list(directory.glob(f"*{dataset_name}*")))

            for f in files_to_archive:
                if f.is_file():
                    shutil.copy2(str(f), str(temp_archive_src / f.name))

            # Create zip archive
            zip_output = shutil.make_archive(
                base_name=str(archive_file),
                format="zip",
                root_dir=str(temp_archive_src)
            )
            
            # Clean temp
            shutil.rmtree(temp_archive_src)
            
            final_zip_path = Path(zip_output)
            logger.info(f"FileManager: Archive successfully created at {final_zip_path}")
            return final_zip_path
            
        except Exception as e:
            logger.error(f"Failed to compile archive: {e}")
            raise ProjectException(f"Error archiving workspace files: {e}") from e
