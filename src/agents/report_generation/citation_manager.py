"""Citation manager tracking and indexing tables, figures, and footnotes."""

from src.core.logger import get_logger

logger = get_logger(__name__)


class CitationManager:
    """Manages automatic indexing label generations for charts (e.g. Figure 1) and tables."""

    def __init__(self) -> None:
        """Initialize CitationManager."""
        self.figure_count = 0
        self.table_count = 0
        self.reference_count = 0

        self.figures_registry: dict[str, str] = {}
        self.tables_registry: dict[str, str] = {}
        self.references_registry: dict[str, str] = {}

    def register_figure(self, chart_id: str, _caption: str) -> str:
        """Register a visual chart, incrementing the counter and returning the label tag.

        Args:
            chart_id: Unique chart identifier.
            _caption: Figure description.

        Returns:
            str: Assigned label index (e.g. 'Figure 1').
        """
        if chart_id in self.figures_registry:
            return self.figures_registry[chart_id]

        self.figure_count += 1
        label = f"Figure {self.figure_count}"
        self.figures_registry[chart_id] = label
        logger.info(f"CitationManager: Registered {label} for chart key: '{chart_id}'")
        return label

    def register_table(self, table_id: str, _description: str) -> str:
        """Register a data table, incrementing the counter and returning the label tag.

        Args:
            table_id: Unique table identifier.
            _description: Table description.

        Returns:
            str: Assigned label index (e.g. 'Table 1').
        """
        if table_id in self.tables_registry:
            return self.tables_registry[table_id]

        self.table_count += 1
        label = f"Table {self.table_count}"
        self.tables_registry[table_id] = label
        logger.info(f"CitationManager: Registered {label} for table key: '{table_id}'")
        return label

    def register_reference(self, ref_key: str, _cite_text: str) -> str:
        """Register a source citation footnote, incrementing counter.

        Args:
            ref_key: Unique reference code.
            cite_text: Bibliographic source description.

        Returns:
            str: Assigned reference number (e.g. '[1]').
        """
        if ref_key in self.references_registry:
            return self.references_registry[ref_key]

        self.reference_count += 1
        label = f"[{self.reference_count}]"
        self.references_registry[ref_key] = label
        logger.info(
            f"CitationManager: Registered citation footnote {label} for: '{ref_key}'"
        )
        return label
