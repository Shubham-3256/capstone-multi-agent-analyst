"""Machine Learning Agent orchestrating model fits, cross validation, evaluations and rankings."""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session

from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.exceptions import DatasetException
from src.database.database import DatabaseManager
from src.database.models import ExecutionLog
from src.repositories.log_repository import ExecutionLogRepository
from src.agents.machine_learning.config import MachineLearningConfig
from src.agents.machine_learning.task_detector import TaskDetector
from src.agents.machine_learning.model_factory import ModelFactory
from src.agents.machine_learning.trainer import Trainer
from src.agents.machine_learning.cross_validator import CrossValidator
from src.agents.machine_learning.tuner import HyperparameterTuner
from src.agents.machine_learning.evaluator import Evaluator
from src.agents.machine_learning.ranking import ModelRanker
from src.agents.machine_learning.explainer import Explainer
from src.agents.machine_learning.persistence import Persistence
from src.agents.machine_learning.models import MachineLearningResult

logger = get_logger(__name__)


class MachineLearningAgent:
    """Orchestrates candidate training sweeps, tuning validations, and explainability exports."""

    def __init__(self, db_session: Session) -> None:
        """Initialize MachineLearningAgent with database session.

        Args:
            db_session: Active SQLAlchemy connection session context.
        """
        self.db = db_session
        self.log_repo = ExecutionLogRepository(db_session)

    def run(
        self,
        train_data: pd.DataFrame,
        train_target: pd.Series,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
        config_path: Optional[Path] = None
    ) -> MachineLearningResult:
        """Execute the AutoML Preprocessing and Training Pipeline.

        Detects tasks -> Fits candidates -> Evaluates scores -> Selects best model -> Persists assets.

        Args:
            train_data: Pre-processed training features DataFrame.
            train_target: Training target label Series.
            validation_data: Pre-processed validation features DataFrame.
            validation_target: Validation target label Series.
            config_path: Optional path to YAML configuration settings.

        Returns:
            MachineLearningResult: Summary mapping leaderboard runs and output file paths.
        """
        logger.info("MachineLearningAgent: Commencing model training pipeline...")
        start_time = time.time()

        # Load configurations
        config = MachineLearningConfig.load_from_yaml(config_path)

        # Create audit execution log
        log_record = ExecutionLog(
            task_name="machine_learning_pipeline",
            agent_name="MachineLearningAgent",
            status="running",
            parameters=str({
                "cv_folds": config.modeling.cv_folds,
                "tuning_mode": config.modeling.tuning_mode,
                "search_type": config.modeling.tuning_search_type
            })
        )
        self.log_repo.create(log_record)
        self.db.commit()

        try:
            # 1. Detect ML Task Type
            task_report = TaskDetector.detect_task(train_target)
            task_type = task_report.task_type
            is_binary = task_report.is_binary

            # 2. Model Factory: Load candidates based on task
            candidate_list = (
                config.selection.classification_candidates
                if task_type == "classification"
                else config.selection.regression_candidates
            )
            candidates = ModelFactory.get_candidate_models(task_type, candidate_list, config.modeling.random_seed)

            if not candidates:
                raise DatasetException("No candidate models could be instantiated by the ModelFactory.")

            # Mappings to collect scores and fitted models
            candidate_metrics: Dict[str, Dict[str, float]] = {}
            fitted_candidates: Dict[str, Any] = {}
            training_results = []

            # 3. Fit, Cross-Validate, Tune, and Evaluate each candidate
            for name, model in candidates.items():
                logger.info(f"MachineLearningAgent: Processing model candidate '{name}'...")
                
                # A. Cross-validate first (on untuned candidate to get baseline stability)
                cv_report = CrossValidator.evaluate(
                    model=model,
                    X=train_data,
                    y=train_target,
                    task_type=task_type,
                    cv_folds=config.modeling.cv_folds,
                    random_seed=config.modeling.random_seed
                )

                # B. Hyperparameter tuning (on training set)
                tuned_model = HyperparameterTuner.tune_model(
                    model_name=name,
                    model=model,
                    X_train=train_data,
                    y_train=train_target,
                    task_type=task_type,
                    mode=config.modeling.tuning_mode,
                    search_type=config.modeling.tuning_search_type,
                    n_iter=config.modeling.n_iter_search,
                    cv_folds=config.modeling.cv_folds,
                    random_seed=config.modeling.random_seed,
                    n_jobs=config.modeling.n_jobs
                )

                # C. Final Fit & Track training metadata
                fit_result = Trainer.train_model(
                    model_name=name,
                    model=tuned_model,
                    X_train=train_data,
                    y_train=train_target,
                    cv_score=cv_report.mean_score
                )
                
                training_results.append(fit_result)
                
                # Skip model if fit failed
                if fit_result.error_message:
                    logger.warning(f"Skipping model evaluation for '{name}' due to fit error.")
                    continue

                fitted_candidates[name] = tuned_model

                # D. Evaluate fitted model on validation set
                eval_report = Evaluator.evaluate_model(
                    model=tuned_model,
                    X_test=validation_data,
                    y_test=validation_target,
                    task_type=task_type,
                    is_binary=is_binary
                )
                
                candidate_metrics[name] = eval_report.metrics

            if not fitted_candidates:
                raise DatasetException("All candidate models failed fitting steps.")

            # 4. Model Ranking (sorts by macro F1 for classification and RMSE for regression)
            leaderboard, best_model_name = ModelRanker.rank_models(candidate_metrics, task_type)
            best_model = fitted_candidates[best_model_name]
            best_metrics = candidate_metrics[best_model_name]

            # 5. Explainer (extract importances from best model)
            feature_importances = Explainer.explain_model(best_model, validation_data, validation_target, task_type)

            # 6. Save Artifacts using Persistence layer
            # Create a run metadata summary dict
            metadata = {
                "best_model_name": best_model_name,
                "task_type": task_type,
                "tuning_mode": config.modeling.tuning_mode,
                "run_timestamp": time.time(),
                "metrics_summary": best_metrics
            }
            
            saved_paths = Persistence.save_artifacts(
                best_model=best_model,
                best_model_name=best_model_name,
                leaderboard=leaderboard,
                metrics=best_metrics,
                feature_importances=feature_importances,
                metadata=metadata
            )

            # Log completion details inside database table
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration,
                error_message=f"Best model chosen: {best_model_name} (Score: {best_metrics.get('f1') if task_type == 'classification' else best_metrics.get('rmse')})"
            )
            self.db.commit()

            logger.info(f"MachineLearningAgent: Modeling sweep finished successfully in {round(duration, 4)}s")
            return MachineLearningResult(
                task_report=task_report,
                best_model_name=best_model_name,
                best_model_path=saved_paths["best_model_path"],
                leaderboard=leaderboard,
                best_metrics=best_metrics,
                feature_importances=feature_importances,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"MachineLearningAgent: Execution failure: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Modeling run error: {str(e)}"
            )
            self.db.commit()
            raise DatasetException(f"Pipeline error executing MachineLearningAgent: {e}") from e
