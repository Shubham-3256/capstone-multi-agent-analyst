"""Business Insight Agent orchestrating prompt compilations, LLM client calls, and reports parsing."""

import time
from typing import Any

from sqlalchemy.orm import Session

from src.agents.business_insights.business_recommendations import (
    BusinessRecommendationsBuilder,
)
from src.agents.business_insights.confidence import ConfidenceBuilder
from src.agents.business_insights.data_quality import DataQualityBuilder
from src.agents.business_insights.executive_summary import ExecutiveSummaryBuilder
from src.agents.business_insights.model_analysis import ModelAnalysisBuilder
from src.agents.business_insights.models import (
    BusinessInsightResult,
    ConfidenceReport,
    DatasetInsight,
    ExecutiveSummary,
    ModelInsight,
    Recommendation,
    RiskItem,
)
from src.agents.business_insights.risk_analysis import RiskAnalysisBuilder
from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult
from src.agents.visualization.models import VisualizationResult
from src.core.llm import CostTracker, LLMClient, StructuredParser
from src.core.logger import get_logger
from src.database.models import ExecutionLog
from src.repositories.log_repository import ExecutionLogRepository

logger = get_logger(__name__)


class BusinessInsightAgent:
    """True AI reasoning orchestrator assembling contextual prompts and parsing LLM insights."""

    def __init__(self, db_session: Session) -> None:
        """Initialize BusinessInsightAgent with database session.

        Args:
            db_session: Active SQLAlchemy connection session context.
        """
        self.db = db_session
        self.log_repo = ExecutionLogRepository(db_session)
        self.llm = LLMClient()

    def run(
        self,
        dataset_profile: Any,
        feature_result: FeatureEngineeringResult | None = None,
        ml_result: MachineLearningResult | None = None,
        visualization_result: VisualizationResult | None = None
    ) -> BusinessInsightResult:
        """Execute Business Intelligence translations.

        Formats section prompts -> Calls LLM client -> Parsers model objects -> Computes token costs.

        Args:
            dataset_profile: Profiling report metadata or dataframe context.
            feature_result: Preprocessing metadata context.
            ml_result: AutoML training metadata context.
            visualization_result: Exporter path references metadata context.

        Returns:
            BusinessInsightResult: Compiled structured corporate reports.
        """
        logger.info("BusinessInsightAgent: Initiating corporate translation analysis...")
        start_time = time.time()

        # Create audit execution log
        log_record = ExecutionLog(
            task_name="business_insights_pipeline",
            agent_name="BusinessInsightAgent",
            status="running"
        )
        self.log_repo.create(log_record)
        self.db.commit()

        try:
            # 1. Format input summaries
            # Safeguard profile extraction
            profile_summary = str(dataset_profile)

            # Format leaderboard summary
            leaderboard_str = "None"
            best_model_name = "N/A"
            best_metrics_str = "None"
            importances_str = "None"

            if ml_result:
                best_model_name = ml_result.best_model_name
                best_metrics_str = str(ml_result.best_metrics)

                leaderboard_records = []
                for entry in ml_result.leaderboard.entries:
                    leaderboard_records.append(f"Rank {entry.rank}: {entry.model_name} (Score={entry.score})")
                leaderboard_str = "; ".join(leaderboard_records)

                importances_records = []
                for item in ml_result.feature_importances:
                    importances_records.append(f"{item.column}={item.importance}")
                importances_str = ", ".join(importances_records)

            # Format visualizer paths
            missing_meta = "None"
            corr_meta = "None"
            if visualization_result and visualization_result.report:
                for chart in visualization_result.report.charts:
                    if chart.chart_id == "missing_heatmap":
                        missing_meta = f"Title: {chart.title}, Captions: {chart.caption.summary} - {chart.caption.details}"
                    elif chart.chart_id == "correlation_heatmap":
                        corr_meta = f"Title: {chart.title}, Captions: {chart.caption.summary} - {chart.caption.details}"

            # 2. Section Prompt builds and LLM calls
            # A. Executive Summary
            exec_prompt = ExecutiveSummaryBuilder.build_prompt(profile_summary, leaderboard_str)
            exec_resp = self.llm.generate(exec_prompt)
            exec_summary = StructuredParser.parse_response(exec_resp, ExecutiveSummary)

            # B. Data Quality Assessment
            quality_prompt = DataQualityBuilder.build_prompt(missing_meta, corr_meta)
            quality_resp = self.llm.generate(quality_prompt)
            dataset_insight = StructuredParser.parse_response(quality_resp, DatasetInsight)

            # C. Model Performance Analysis
            model_prompt = ModelAnalysisBuilder.build_prompt(leaderboard_str, best_model_name, best_metrics_str)
            model_resp = self.llm.generate(model_prompt)
            model_insight = StructuredParser.parse_response(model_resp, ModelInsight)

            # D. Strategic Recommendations
            recs_prompt = BusinessRecommendationsBuilder.build_prompt(best_metrics_str, importances_str)
            recs_resp = self.llm.generate(recs_prompt)
            # Support parsing single recommendation array or single item mapping
            try:
                # Expecting single structured recommendation model mapping
                recommendation = StructuredParser.parse_response(recs_resp, Recommendation)
                recommendations = [recommendation]
            except Exception:
                # Fallback parsed items using dynamic feature reference
                feat_fallback = "features"
                if ml_result and getattr(ml_result, "feature_importances", None):
                    feat_fallback = ml_result.feature_importances[0].column
                recommendations = [Recommendation(
                    title=f"Optimize engineering for {feat_fallback}",
                    description=f"Further audit and transform feature variables like {feat_fallback} to boost predictive margins.",
                    actionability="High"
                )]

            # E. Risk Audits
            risks_prompt = RiskAnalysisBuilder.build_prompt(best_metrics_str, importances_str)
            risks_resp = self.llm.generate(risks_prompt)
            try:
                risk_item = StructuredParser.parse_response(risks_resp, RiskItem)
                risks = [risk_item]
            except Exception:
                feat_fallback = "features"
                if ml_result and getattr(ml_result, "feature_importances", None):
                    feat_fallback = ml_result.feature_importances[0].column
                risks = [RiskItem(
                    severity="Medium",
                    probability="Low",
                    description=f"Model sensitivity vulnerabilities targeting feature variance in {feat_fallback}."
                )]

            # F. Confidence verification
            conf_prompt = ConfidenceBuilder.build_prompt(best_metrics_str, profile_summary)
            conf_resp = self.llm.generate(conf_prompt)
            confidence_report = StructuredParser.parse_response(conf_resp, ConfidenceReport)

            # 3. Retrieve token costs from session cost tracker
            session_stats = CostTracker.get_session_summary()

            # Reset cost tracker session for cleanliness
            CostTracker.reset_session()

            # Log completion details inside database table
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration,
                error_message=f"Generated executive report. Estimated Cost: ${round(session_stats['session_cost'], 6)}"
            )
            self.db.commit()

            logger.info(f"BusinessInsightAgent: Analysis completed successfully in {round(duration, 4)}s")
            return BusinessInsightResult(
                executive_summary=exec_summary,
                dataset_insight=dataset_insight,
                model_insight=model_insight,
                recommendations=recommendations,
                risks=risks,
                confidence_report=confidence_report,
                token_usage={
                    "input_tokens": session_stats["session_input_tokens"],
                    "output_tokens": session_stats["session_output_tokens"]
                },
                estimated_cost_usd=session_stats["session_cost"]
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"BusinessInsightAgent: Orchestration failed: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Orchestration error: {str(e)}"
            )
            self.db.commit()
            raise ValueError(f"Agent error running BusinessInsightAgent: {e}") from e
