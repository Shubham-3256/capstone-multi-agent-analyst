"""Context builder consolidating outputs from all pipeline agents into a single context."""

from typing import Any

import pandas as pd

from src.agents.business_insights.models import BusinessInsightResult
from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult
from src.agents.report_generation.models import ReportContext
from src.agents.visualization.models import VisualizationResult
from src.core.logger import get_logger

logger = get_logger(__name__)


def _get(obj: Any, key: str) -> Any:
    """Safely extract an attribute or key value from any object or dictionary."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class ContextBuilder:
    """Consolidates outputs from preceding phases into a structured report context."""

    @staticmethod
    def build_context(
        dataset_profile: Any,
        feature_result: FeatureEngineeringResult | None = None,
        ml_result: MachineLearningResult | None = None,
        visualization_result: VisualizationResult | None = None,
        business_result: BusinessInsightResult | None = None
    ) -> ReportContext:
        """Merge agent run objects into a ReportContext structure.

        Args:
            dataset_profile: Profiling report metadata or dataframe.
            feature_result: Feature engineering outputs metadata.
            ml_result: AutoML training performance metadata.
            visualization_result: Plotly charts locations lists.
            business_result: Technical business takeaways and checklists.

        Returns:
            ReportContext: Standardized reporting inputs.
        """
        logger.info("ContextBuilder: Consolidating phase outputs...")

        # 1. Dataset Profile
        if isinstance(dataset_profile, pd.DataFrame):
            profile_str = f"DataFrame: shape={dataset_profile.shape}, columns={list(dataset_profile.columns)}"
        else:
            profile_str = str(dataset_profile)

        # 2. Feature Engineering
        if feature_result:
            selected_features = _get(feature_result, "selected_features") or []
            leakages = _get(feature_result, "leakage_warnings") or []
            feature_str = (
                f"Feature Engineering Summary:\n"
                f"  - Selected Features: {', '.join(selected_features) if selected_features else 'None'}\n"
                f"  - Train Split path: {_get(feature_result, 'train_filepath')}\n"
                f"  - Test Split path: {_get(feature_result, 'test_filepath')}\n"
                f"  - Leakage Audits: {', '.join(leakages) if leakages else 'Clean'}"
            )
        else:
            feature_str = "Feature engineering was not executed or not provided."

        # 3. Machine Learning
        if ml_result:
            leaderboard_lines = []
            leaderboard = _get(ml_result, "leaderboard")
            entries = _get(leaderboard, "entries") or []
            for entry in entries:
                rank = _get(entry, "rank")
                model_name = _get(entry, "model_name")
                score = _get(entry, "score")
                leaderboard_lines.append(f"    * Rank {rank}: {model_name} (Score={score})")
            leaderboard_str = "\n".join(leaderboard_lines) if leaderboard_lines else "    * Empty Leaderboard"

            ml_str = (
                f"Machine Learning Results:\n"
                f"  - Best Model: {_get(ml_result, 'best_model_name')}\n"
                f"  - Metrics: {_get(ml_result, 'best_metrics')}\n"
                f"  - Leaderboard:\n{leaderboard_str}"
            )
        else:
            ml_str = "AutoML model training was not executed or not provided."

        # 4. Visualization Assets
        viz_lines = []
        charts_paths = []
        report = _get(visualization_result, "report")
        charts = _get(report, "charts") or []
        if charts:
            for chart in charts:
                title = _get(chart, "title")
                chart_type = _get(chart, "chart_type")
                file_path = _get(chart, "file_path")
                if file_path:
                    viz_lines.append(f"  - {title} ({str(chart_type).upper()}) at: {file_path}")
                    charts_paths.append(file_path)
                else:
                    viz_lines.append(f"  - {title} ({str(chart_type).upper()}): Not plotted (No missing values)")
            viz_str = "Visualization charts metadata:\n" + "\n".join(viz_lines)
        else:
            viz_str = "No charts were generated or visualization metadata is missing."

        # 5. Business Insights
        if business_result:
            exec_summary = _get(business_result, "executive_summary")
            key_takeaways = _get(exec_summary, "key_takeaways") or []
            takeaways = "\n".join([f"    * {t}" for t in key_takeaways])

            recs_list = _get(business_result, "recommendations") or []
            recs_formatted = []
            for r in recs_list:
                r_title = _get(r, "title")
                r_desc = _get(r, "description")
                recs_formatted.append(f"    * {r_title}: {r_desc}")
            recs = "\n".join(recs_formatted)

            risks_list = _get(business_result, "risks") or []
            risks_formatted = []
            for rk in risks_list:
                rk_desc = _get(rk, "description")
                rk_sev = _get(rk, "severity")
                risks_formatted.append(f"    * {rk_desc} (Severity={rk_sev})")
            risks = "\n".join(risks_formatted)

            insights_str = (
                f"Business Insights:\n"
                f"  - Headline: {_get(exec_summary, 'headline')}\n"
                f"  - Takeaways:\n{takeaways}\n"
                f"  - Impact: {_get(exec_summary, 'impact_statement')}\n"
                f"  - Recommendations:\n{recs}\n"
                f"  - Risks:\n{risks}"
            )
        else:
            insights_str = "AI business insights was not executed or not provided."

        return ReportContext(
            dataset_profile_str=profile_str,
            feature_summary_str=feature_str,
            ml_summary_str=ml_str,
            viz_summary_str=viz_str,
            insights_str=insights_str,
            charts_paths=charts_paths
        )
