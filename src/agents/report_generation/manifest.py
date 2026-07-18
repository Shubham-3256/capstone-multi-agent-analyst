"""Manifest generator creating structured report_manifest.json catalog files."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import uuid

from src.core.logger import get_logger
from src.agents.report_generation.models import ReportManifest

logger = get_logger(__name__)


class ManifestGenerator:
    """Generates structured report manifest JSON mappings documenting data hashes, charts, and pipeline items."""

    @staticmethod
    def generate_manifest(
        dataset_hash: str,
        charts_included: List[str],
        sections: List[str],
        target_dir: Path
    ) -> ReportManifest:
        """Compile a ReportManifest metadata schema and write it to target_dir.

        Args:
            dataset_hash: SHA256 code representing source dataset.
            charts_included: Names of figure files.
            sections: Ordered generated chapters.
            target_dir: Destination folder.

        Returns:
            ReportManifest: Generated manifest schema object.
        """
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        manifest = ReportManifest(
            report_id=report_id,
            dataset_hash=dataset_hash,
            pipeline_version="1.0.0",
            model_version="gpt-4o",
            charts_included=charts_included,
            sections=sections,
            generation_timestamp=timestamp
        )

        manifest_file = target_dir / "report_manifest.json"
        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                # Use model dump JSON directly
                f.write(manifest.model_dump_json(indent=4))
            logger.info(f"ManifestGenerator: Manifest file cataloged successfully at: {manifest_file}")
        except Exception as e:
            logger.error(f"ManifestGenerator: Failed to write report_manifest.json: {e}")

        return manifest
