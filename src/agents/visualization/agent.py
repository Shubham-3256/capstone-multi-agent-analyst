"""Visualization Agent orchestrating dataset distributions, model metrics, and dashboards."""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult
from src.agents.visualization.caption_generator import CaptionGenerator
from src.agents.visualization.config import VisualizationConfig
from src.agents.visualization.dashboard_visualizer import DashboardVisualizer
from src.agents.visualization.dataset_visualizer import DatasetVisualizer
from src.agents.visualization.exporter import Exporter
from src.agents.visualization.model_visualizer import ModelVisualizer
from src.agents.visualization.models import (
    ChartCaption,
    ChartMetadata,
    VisualizationMetadata,
    VisualizationReport,
    VisualizationResult,
)
from src.core.exceptions import DatasetException
from src.core.logger import get_logger
from src.core.paths import Paths
from src.database.models import ExecutionLog
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.log_repository import ExecutionLogRepository
from src.utils.serialization import deserialize_dataframe

logger = get_logger(__name__)


class VisualizationAgent:
    """Agent orchestrator executing figure factory runs, captions compile and exporters saves."""

    def __init__(self, db_session: Session) -> None:
        """Initialize VisualizationAgent with database session.

        Args:
            db_session: Active SQLAlchemy connection session context.
        """
        self.db = db_session
        self.log_repo = ExecutionLogRepository(db_session)
        self.dataset_repo = DatasetRepository(db_session)

    def run(
        self,
        dataset_profile: pd.DataFrame | str | int,
        feature_result: FeatureEngineeringResult | None = None,
        ml_result: MachineLearningResult | None = None,
        config_path: Path | None = None,
        target_column: str | None = None,
    ) -> VisualizationResult:
        """Orchestrate and compile Matplotlib and Plotly figures, exporting static and interactive plots.

        Args:
            dataset_profile: Input DataFrame, dataset ID string, or registered record ID.
            feature_result: Results context from Feature Engineering phase.
            ml_result: Results context from Machine Learning phase.
            config_path: Optional path to YAML configuration settings.
            target_column: Optional target modeling label column override.

        Returns:
            VisualizationResult: Mapped locations, reports, and success status indicators.
        """
        logger.info("VisualizationAgent: Starting visualization pipeline run...")
        start_time = time.time()

        # Load configuration
        config = VisualizationConfig.load_from_yaml(config_path)
        theme = config.default_theme

        # Create audit execution log
        log_record = ExecutionLog(
            task_name="visualization_pipeline",
            agent_name="VisualizationAgent",
            status="running",
            parameters=str({"theme": theme, "formats": config.export_formats}),
        )
        self.log_repo.create(log_record)
        self.db.commit()

        try:
            # 1. Resolve and Load the DataFrame
            df: pd.DataFrame | None = None
            dataset_hash: str | None = None

            if isinstance(dataset_profile, pd.DataFrame):
                df = dataset_profile
            elif isinstance(dataset_profile, (str, int)):
                # Query database for filepath
                record = self.dataset_repo.get_by_id(str(dataset_profile))
                if record:
                    df = deserialize_dataframe(
                        Path(record.filepath), fmt=Path(record.filepath).suffix
                    )
                    dataset_hash = record.file_hash
                else:
                    # Fallback test if ID is not in DB but matches a path string
                    possible_path = Path(str(dataset_profile))
                    if possible_path.exists():
                        df = deserialize_dataframe(
                            possible_path, fmt=possible_path.suffix
                        )

            if df is None or df.empty:
                raise DatasetException(
                    "Could not load input dataset DataFrame for plotting visual heatmaps."
                )

            # Identify target column context

            # Setup output root directory
            base_dir = Paths.WORKSPACE_DIR / "artifacts"
            base_dir.mkdir(parents=True, exist_ok=True)

            chart_metadata_list: list[ChartMetadata] = []

            # 2. Dataset Visualizations
            # A. Missingness Heatmap
            total_missing = int(df.isna().sum().sum())
            if total_missing == 0:
                missing_meta = ChartMetadata(
                    chart_id="missing_heatmap",
                    title="Missing Value Matrix Heatmap",
                    chart_type="heatmap",
                    file_path="",
                    html_path=None,
                    caption=ChartCaption(
                        summary="No missing values detected.",
                        details="The dataset contains zero null cells; it is 100% complete and verified.",
                    ),
                )
            else:
                missing_figs = DatasetVisualizer.generate_missing_heatmap(df, theme)
                missing_caption = CaptionGenerator.generate_missing_heatmap_caption(df)
                missing_meta = Exporter.export_chart(
                    chart_id="missing_heatmap",
                    title="Missing Value Matrix Heatmap",
                    chart_type="heatmap",
                    figures=missing_figs,
                    caption=missing_caption,
                    base_dir=base_dir,
                )
            chart_metadata_list.append(missing_meta)

            # B. Correlation Heatmap
            corr_figs = DatasetVisualizer.generate_correlation_heatmap(df, theme)
            corr_caption = CaptionGenerator.generate_correlation_heatmap_caption(df)
            corr_meta = Exporter.export_chart(
                chart_id="correlation_heatmap",
                title="Pearson Correlation Heatmap",
                chart_type="heatmap",
                figures=corr_figs,
                caption=corr_caption,
                base_dir=base_dir,
            )
            chart_metadata_list.append(corr_meta)

            # C. Single Variable Distribution (select numeric if possible, skipping ID/constant/empty)
            valid_cols = DatasetVisualizer.get_valid_columns(df)
            valid_df = df[valid_cols] if valid_cols else df
            numeric_cols = list(valid_df.select_dtypes(include=["number"]).columns)
            dist_col = (
                numeric_cols[0]
                if numeric_cols
                else (valid_cols[0] if valid_cols else df.columns[0])
            )
            dist_figs = DatasetVisualizer.generate_distribution_plot(
                df, dist_col, theme
            )
            dist_caption = CaptionGenerator.generate_distribution_caption(df, dist_col)
            dist_meta = Exporter.export_chart(
                chart_id=f"distribution_{dist_col}",
                title=f"Distribution: {dist_col}",
                chart_type="histogram",
                figures=dist_figs,
                caption=dist_caption,
                base_dir=base_dir,
            )
            chart_metadata_list.append(dist_meta)

            # 3. Model & Prediction Visualizations
            # If MachineLearningResult details are present, execute model visual overlays
            if ml_result:
                best_model_name = ml_result.best_model_name
                task_type = ml_result.task_report.task_type

                # Retrieve fitted best model
                best_model_path = Path(ml_result.best_model_path)
                if best_model_path.exists():
                    import joblib

                    best_model = joblib.load(str(best_model_path))

                    # Resolve validation data splits context
                    # If feature engineering saved splits in default folder, reload them
                    val_filepath = Paths.PROCESSED_DIR / "val.csv"
                    if val_filepath.exists():
                        val_df = pd.read_csv(val_filepath)
                        # Extract target target vector
                        # Find overlapping label
                        common_target = target_column
                        if not common_target or common_target not in val_df.columns:
                            common_target = None
                            if ml_result.task_report.classes:
                                # Search target names in columns
                                for c in val_df.columns:
                                    if c in ["target", "churn", "label", "class"]:
                                        common_target = c
                                        break

                        if not common_target:
                            common_target = val_df.columns[-1]

                        X_val = val_df.drop(columns=[common_target])
                        y_val = val_df[common_target]

                        # A. Residuals Plot (for regression) or ROC Curve (for classification)
                        if task_type == "classification":
                            roc_figs = ModelVisualizer.generate_roc_curve(
                                best_model, X_val, y_val, theme
                            )
                            roc_caption = ChartCaption(
                                summary=f"ROC validation curve for model '{best_model_name}'.",
                                details=f"Calculated Area Under Curve: AUC={round(ml_result.best_metrics.get('roc_auc', 0.5), 4)}.",
                            )
                            roc_meta = Exporter.export_chart(
                                chart_id="roc_curve",
                                title="ROC Curve",
                                chart_type="line",
                                figures=roc_figs,
                                caption=roc_caption,
                                base_dir=base_dir,
                            )
                            chart_metadata_list.append(roc_meta)
                        else:
                            res_figs = ModelVisualizer.generate_residual_plot(
                                best_model, X_val, y_val, theme
                            )
                            res_caption = ChartCaption(
                                summary="Residuals errors distribution.",
                                details="Shows differences between true labels and predictions plotted against center zero.",
                            )
                            res_meta = Exporter.export_chart(
                                chart_id="residuals_plot",
                                title="Residuals Plot",
                                chart_type="scatter",
                                figures=res_figs,
                                caption=res_caption,
                                base_dir=base_dir,
                            )
                            chart_metadata_list.append(res_meta)

                # B. Feature Importance Plot
                importances = ml_result.feature_importances
                if importances:
                    imp_figs = ModelVisualizer.generate_feature_importance_plot(
                        importances, theme
                    )
                    imp_caption = CaptionGenerator.generate_importance_caption(
                        importances
                    )
                    imp_meta = Exporter.export_chart(
                        chart_id="feature_importance",
                        title="Feature Importance Profiles",
                        chart_type="bar",
                        figures=imp_figs,
                        caption=imp_caption,
                        base_dir=base_dir,
                    )
                    chart_metadata_list.append(imp_meta)

                # C. Leaderboard Chart comparisons
                leaderboard = ml_result.leaderboard
                if leaderboard and leaderboard.entries:
                    ld_figs = DashboardVisualizer.generate_leaderboard_chart(
                        leaderboard, task_type, theme
                    )
                    ld_caption = CaptionGenerator.generate_leaderboard_caption(
                        leaderboard, task_type
                    )
                    ld_meta = Exporter.export_chart(
                        chart_id="leaderboard_chart",
                        title="Candidate Model Leaderboard Comparison",
                        chart_type="bar",
                        figures=ld_figs,
                        caption=ld_caption,
                        base_dir=base_dir,
                    )
                    chart_metadata_list.append(ld_meta)

            # 4. Generate Unified Metadata Report JSON
            report_meta = VisualizationMetadata(
                dataset_hash=dataset_hash,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                theme=theme,
            )
            report = VisualizationReport(
                charts=chart_metadata_list, metadata=report_meta
            )

            metadata_filepath = base_dir / "metadata" / "visualization_report.json"
            Exporter.save_report_metadata(report.model_dump(), metadata_filepath)

            # Update SQLite ExecutionLog status
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration,
                error_message=f"Generated and exported {len(chart_metadata_list)} charts.",
            )
            self.db.commit()

            logger.info(
                f"VisualizationAgent: Sweep completed successfully. Generated {len(chart_metadata_list)} charts."
            )
            return VisualizationResult(
                is_success=True,
                report=report,
                output_directory=str(base_dir),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"VisualizationAgent: Execution failure: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Visualization error: {str(e)}",
            )
            self.db.commit()
            raise DatasetException(
                f"Pipeline error executing VisualizationAgent: {e}"
            ) from e
