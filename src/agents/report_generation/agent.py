"""Report Generation Agent orchestrator compiling contexts, rendering layouts, validating and exporting formats."""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.agents.business_insights.models import BusinessInsightResult
from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult
from src.agents.report_generation.asset_manager import AssetManager
from src.agents.report_generation.citation_manager import CitationManager
from src.agents.report_generation.config import ReportConfig
from src.agents.report_generation.context_builder import ContextBuilder
from src.agents.report_generation.exporter import Exporter
from src.agents.report_generation.manifest import ManifestGenerator
from src.agents.report_generation.models import (
    ReportMetadata,
    ReportResult,
)
from src.agents.report_generation.section_builder import SectionBuilder
from src.agents.report_generation.template_engine import TemplateEngine
from src.agents.report_generation.validator import ReportValidator
from src.agents.visualization.models import VisualizationResult
from src.core.logger import get_logger
from src.database.database import SessionLocal, init_db
from src.database.models import ExecutionLog, ReportRecord
from src.repositories.log_repository import ExecutionLogRepository

logger = get_logger(__name__)


class ReportGenerationAgent:
    """Orchestrates report compilation, asset transfers, and exports in MD/HTML/PDF/DOCX formats."""

    def __init__(self, db_session: Session | None = None, config: ReportConfig | None = None) -> None:
        """Initialize ReportGenerationAgent.

        Args:
            db_session: Active SQLAlchemy connection session context. A managed session is
                created when omitted so ``ReportGenerationAgent().run(...)`` is supported.
            config: Optional ReportConfig settings override.
        """
        if db_session is None:
            init_db()
            db_session = SessionLocal()
        self.db = db_session
        self.log_repo = ExecutionLogRepository(self.db)
        self.config = config or ReportConfig()

    def run(
        self,
        dataset_profile: Any,
        feature_result: FeatureEngineeringResult | None = None,
        ml_result: MachineLearningResult | None = None,
        visualization_result: VisualizationResult | None = None,
        business_result: BusinessInsightResult | None = None,
        template_type: str = "executive"
    ) -> ReportResult:
        """Orchestrate document assembly and export cycles.

        Args:
            dataset_profile: Profiling report metadata or dataframe.
            feature_result: Preprocessing metadata context.
            ml_result: AutoML training metadata context.
            visualization_result: Chart files metadata.
            business_result: Corporate insights.
            template_type: Layout code key ('executive', 'technical').

        Returns:
            ReportResult: Export status payload mapping files.
        """
        logger.info("ReportGenerationAgent: Launching reporting pipeline...")
        start_time = time.time()

        # Database audit log initialization
        log_record = ExecutionLog(
            task_name="report_generation_pipeline",
            agent_name="ReportGenerationAgent",
            status="running"
        )
        self.log_repo.create(log_record)
        self.db.commit()

        # Determine target export directories
        base_dir = self.config.reports_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Consolidate Context
            context = ContextBuilder.build_context(
                dataset_profile=dataset_profile,
                feature_result=feature_result,
                ml_result=ml_result,
                visualization_result=visualization_result,
                business_result=business_result
            )

            # 2. Build sections and assign durable, sequential citations.
            sections = SectionBuilder.build_sections(context)
            citations = CitationManager()
            for section in sections.values():
                section.figures = [
                    figure.model_copy(update={
                        "label": citations.register_figure(figure.file_path, figure.caption)
                    })
                    for figure in section.figures
                ]
                section.tables = [
                    table.model_copy(update={
                        "label": citations.register_table(table.description, table.description)
                    })
                    for table in section.tables
                ]
            sections["references"].content_markdown += (
                "\n\n" + citations.register_reference(
                    "platform", "Multi-Agent AI Data Analyst reporting pipeline."
                ) + " Multi-Agent AI Data Analyst reporting pipeline."
            )

            # 3. Organize Assets (Charts / Tables)
            # Create subdirs
            assets_rel_paths = AssetManager.organize_assets(
                charts_paths=context.charts_paths,
                target_dir=base_dir,
                resize_width=600
            ) or []

            # 4. Generate Metadata catalog parameters
            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            metadata = ReportMetadata(
                title=f"Analytics Report - {context.dataset_name}",
                template_type=template_type,
                created_at=created_at
            )

            # 5. Render template
            sections_md = {sec_id: sec.content_markdown for sec_id, sec in sections.items()}
            meta_dict = {
                "title": metadata.title,
                "author": metadata.author,
                "created_at": metadata.created_at
            }
            compiled_md = TemplateEngine.render(template_type, meta_dict, sections_md)

            # 6. Generate Manifest catalog entry
            # Calculate mock dataset hash signature
            dataset_hash = hashlib.sha256(context.dataset_profile_str.encode("utf-8")).hexdigest()
            manifest = ManifestGenerator.generate_manifest(
                dataset_hash=dataset_hash,
                charts_included=assets_rel_paths,
                sections=list(sections_md.keys()),
                target_dir=base_dir
            )

            # 7. Perform validation
            # 8. Export files before validation so on-disk artifacts are checked.
            out_paths = {}
            # A. Markdown
            md_file = base_dir / "markdown" / "report.md"
            Exporter.export_markdown(compiled_md, md_file)
            out_paths["markdown"] = str(md_file.resolve())

            # B. HTML
            html_file = base_dir / "html" / "report.html"
            Exporter.export_html(compiled_md, html_file)
            out_paths["html"] = str(html_file.resolve())

            # C. PDF
            pdf_file = base_dir / "pdf" / "report.pdf"
            Exporter.export_pdf(compiled_md, pdf_file)
            out_paths["pdf"] = str(pdf_file.resolve())

            # D. DOCX
            docx_file = base_dir / "docx" / "report.docx"
            Exporter.export_docx(compiled_md, docx_file)
            out_paths["docx"] = str(docx_file.resolve())

            is_valid = ReportValidator.validate_report(
                sections, manifest, base_dir, output_paths=out_paths
            )
            if not is_valid:
                raise ValueError("Report validation failed; exports were not accepted.")

            report_record = ReportRecord(
                report_id=manifest.report_id,
                title=metadata.title,
                template_type=metadata.template_type,
                dataset_hash=manifest.dataset_hash,
                manifest_json=manifest.model_dump_json(),
                output_paths_json=json.dumps(out_paths),
                duration_seconds=time.time() - start_time,
            )
            self.db.add(report_record)

            # Log execution completion
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration,
                error_message=f"Generated report formats in {base_dir}"
            )
            self.db.commit()

            logger.info(f"ReportGenerationAgent: Execution completed in {round(duration, 4)}s. Files generated.")
            return ReportResult(
                is_success=is_valid,
                output_paths=out_paths,
                manifest=manifest,
                metadata=metadata,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"ReportGenerationAgent: Orchestration pipeline failed: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Orchestration error: {str(e)}"
            )
            self.db.commit()
            raise ValueError(f"Agent error running ReportGenerationAgent: {e}") from e
