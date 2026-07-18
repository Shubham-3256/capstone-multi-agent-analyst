"""Report Generation Agent package exports."""

from src.agents.report_generation.agent import ReportGenerationAgent
from src.agents.report_generation.config import ReportConfig
from src.agents.report_generation.context_builder import ContextBuilder
from src.agents.report_generation.template_engine import TemplateEngine
from src.agents.report_generation.section_builder import SectionBuilder
from src.agents.report_generation.asset_manager import AssetManager
from src.agents.report_generation.citation_manager import CitationManager
from src.agents.report_generation.manifest import ManifestGenerator
from src.agents.report_generation.validator import ReportValidator
from src.agents.report_generation.exporter import Exporter
from src.agents.report_generation.models import (
    FigureReference,
    TableReference,
    ReportSection,
    ReportContext,
    ReportManifest,
    ReportMetadata,
    ReportResult,
)

__all__ = [
    "ReportGenerationAgent",
    "ReportConfig",
    "ContextBuilder",
    "TemplateEngine",
    "SectionBuilder",
    "AssetManager",
    "CitationManager",
    "ManifestGenerator",
    "ReportValidator",
    "Exporter",
    "FigureReference",
    "TableReference",
    "ReportSection",
    "ReportContext",
    "ReportManifest",
    "ReportMetadata",
    "ReportResult",
]
