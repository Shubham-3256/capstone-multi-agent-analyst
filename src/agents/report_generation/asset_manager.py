"""Asset manager organizing visual graphs and tables inside target reports directories."""

import shutil
from pathlib import Path

from PIL import Image

from src.core.logger import get_logger
from src.core.paths import Paths

logger = get_logger(__name__)


class AssetManager:
    """Copies, resizes, and organizes graph files and logos for clean document imports."""

    @staticmethod
    def organize_assets(
        charts_paths: list[str], target_dir: Path, resize_width: int = 600
    ) -> list[str] | None:
        """Copy visual charts from workspace to target reports asset directory.

        Args:
            charts_paths: Original file paths list.
            target_dir: Target destination parent directory.
            resize_width: Max scaling width in pixels.

        Returns:
            List[str]: Copied asset relative paths.
        """
        logger.info(
            f"AssetManager: Organizing {len(charts_paths)} charts to: {target_dir}"
        )
        copied_paths = []

        assets_dir = target_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        for path_str in charts_paths:
            src_path = Path(path_str)
            if not src_path.is_absolute():
                src_path = Paths.WORKSPACE_DIR / "workspace" / "artifacts" / src_path

            # Fallback path resolve check
            if not src_path.exists():
                src_path = (
                    Paths.WORKSPACE_DIR
                    / "workspace"
                    / "artifacts"
                    / "plots"
                    / src_path.name
                )

            if not src_path.exists():
                logger.warning(
                    f"AssetManager: Reference file {src_path} does not exist. Skipping."
                )
                continue

            dest_path = assets_dir / src_path.name
            try:
                # Copy file
                shutil.copy2(src_path, dest_path)

                # Perform Pillow image resizing for HTML/PDF uniformity
                try:
                    with Image.open(dest_path) as img:
                        # Only resize if wider than limit
                        if img.width > resize_width:
                            # Calculate proportional height
                            height = int(img.height * (resize_width / img.width))
                            resized_img = img.resize(
                                (resize_width, height), Image.Resampling.LANCZOS
                            )
                            resized_img.save(dest_path)
                            logger.info(
                                f"AssetManager: Scaled image {src_path.name} to {resize_width}x{height}"
                            )
                except Exception as e:
                    logger.warning(
                        f"AssetManager: Pillow image scaling failed for {src_path.name}: {e}. Keeping copy."
                    )

                # Keep relative path for document generation references
                copied_paths.append(f"assets/{src_path.name}")

            except Exception as e:
                logger.error(
                    f"AssetManager: Failed to copy reference asset {src_path.name}: {e}"
                )

        return copied_paths
