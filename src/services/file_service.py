"""Service layer for safe and validated file manipulation within the workspace."""

import shutil
from pathlib import Path
from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.exceptions import DatasetException, ValidationException
from src.core.security import verify_path_bounds

logger = get_logger(__name__)


class FileService:
    """Service handling file operations (moving, copying, renaming, deleting) in the workspace."""

    def __init__(self, base_dir: Path = Paths.WORKSPACE_DIR) -> None:
        """Initialize FileService with a boundary root folder context.

        Args:
            base_dir: Top-level directory restriction for safety checks. Defaults to workspace.
        """
        self.base_dir = base_dir

    def move_file(self, src: Path, dest: Path) -> Path:
        """Safely move a file to a new workspace path location.

        Args:
            src: Source file path.
            dest: Target destination file path.

        Returns:
            Path: The resolved final target file path.

        Raises:
            DatasetException: If files cannot be moved.
        """
        logger.info(f"Moving file: {src} -> {dest}")
        try:
            resolved_src = verify_path_bounds(self.base_dir, src)
            resolved_dest = verify_path_bounds(self.base_dir, dest)

            if not resolved_src.exists():
                raise FileNotFoundError(f"Source file does not exist: {resolved_src}")

            # Ensure parent directories exist
            resolved_dest.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(resolved_src), str(resolved_dest))
            logger.info(f"File successfully moved to {resolved_dest}")
            return resolved_dest
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to move file {src} to {dest}: {e}")
            raise DatasetException(f"Error moving file: {e}") from e

    def copy_file(self, src: Path, dest: Path) -> Path:
        """Safely copy a file to a target destination.

        Args:
            src: Source file path.
            dest: Destination file path.

        Returns:
            Path: Resolving target file path.

        Raises:
            DatasetException: If copies fail.
        """
        logger.info(f"Copying file: {src} -> {dest}")
        try:
            resolved_src = verify_path_bounds(self.base_dir, src)
            resolved_dest = verify_path_bounds(self.base_dir, dest)

            if not resolved_src.exists():
                raise FileNotFoundError(f"Source file does not exist: {resolved_src}")

            resolved_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(resolved_src), str(resolved_dest))
            logger.info(f"File successfully copied to {resolved_dest}")
            return resolved_dest
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to copy file {src} to {dest}: {e}")
            raise DatasetException(f"Error copying file: {e}") from e

    def rename_file(self, src: Path, new_name: str) -> Path:
        """Rename a file in its active directory.

        Args:
            src: Target source file path.
            new_name: New file name string.

        Returns:
            Path: Path representation of renamed file.
        """
        logger.info(f"Renaming file {src} to {new_name}")
        try:
            resolved_src = verify_path_bounds(self.base_dir, src)
            resolved_dest = resolved_src.parent / new_name
            # Validate target location is bounds-compliant
            resolved_dest = verify_path_bounds(self.base_dir, resolved_dest)

            if not resolved_src.exists():
                raise FileNotFoundError(f"Source file does not exist: {resolved_src}")

            resolved_src.rename(resolved_dest)
            logger.info(f"File renamed to: {resolved_dest}")
            return resolved_dest
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to rename file {src} to {new_name}: {e}")
            raise DatasetException(f"Error renaming file: {e}") from e

    def delete_file(self, path: Path) -> bool:
        """Safely delete a file inside the base folder restriction limits.

        Args:
            path: Target file path.

        Returns:
            bool: True if successfully deleted.
        """
        logger.info(f"Deleting file: {path}")
        try:
            resolved_path = verify_path_bounds(self.base_dir, path)
            if resolved_path.exists() and resolved_path.is_file():
                resolved_path.unlink()
                logger.info(f"File successfully deleted: {resolved_path}")
                return True
            logger.warning(f"File not found for deletion: {resolved_path}")
            return False
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
            raise DatasetException(f"Error deleting file: {e}") from e
