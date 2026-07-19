"""Core prompt templates for business summary, operations quality, and modeling interpretations."""


class PromptTemplates:
    """Consolidated prompt prompts catalog for AutoML business translations."""

    EXECUTIVE_SUMMARY = """
You are a Principal Business Intelligence Architect and Enterprise Analyst.
Review the following analysis context and generate a high-level executive summary:
Dataset Metrics: {dataset_summary}
Trained Candidate Models: {leaderboard}

Provide your response in JSON format matching the schema below:
{{
    "headline": "A short, punchy business headline summary statement",
    "key_takeaways": [
        "Takeaway bullet point 1",
        "Takeaway bullet point 2"
    ],
    "impact_statement": "Analytical impact details summary statement"
}}
"""

    DATA_QUALITY = """
You are an expert Data Engineer. Audits the following dataset quality metrics:
Missingness Profile: {missing_meta}
Correlation Metrics: {corr_meta}

Identify validation anomalies and suggest cleaning improvements.
Provide your response in JSON format matching the schema below:
{{
    "completeness_score": 1.0,  # Float value between 0.0 and 1.0
    "anomalies_detected": [
        "Anomaly description 1",
        "Anomaly description 2"
    ],
    "recommendation": "Suggested operations cleaning action"
}}
"""

    MODEL_ANALYSIS = """
You are a Lead Data Scientist. Review the AutoML modeling performance leaderboard:
Trained Candidates Leaderboard: {leaderboard}
Best Performing Estimator: {best_model_name}
Best Model Evaluation Metrics: {best_metrics}

Explain model stability, fit, and error boundaries.
Provide your response in JSON format matching the schema below:
{{
    "algorithm_name": "{best_model_name}",
    "accuracy": 1.0,
    "f1": 1.0,
    "feature_weights": {{
        "feature_1": 0.45,
        "feature_2": 0.35
    }},
    "conclusion": "Summary explanation of model validity and errors metrics"
}}
"""

    RECOMMENDATIONS = """
You are a Principal Business Strategy Consultant.
Review the following model conclusions and feature importances:
Best Model metrics: {best_metrics}
Feature Importance Rankings: {importances}

Develop actionable business strategies to optimize target outcomes.
Provide your response in JSON format matching the schema below:
{{
    "title": "Strategy Title",
    "description": "Deep tactical description of the recommendation",
    "actionability": "Actionability level ('High', 'Medium', 'Low')"
}}
"""

    RISKS = """
You are an Enterprise Risk Officer.
Review the analysis runs results:
AutomL Modeling Summary: {best_metrics}
Feature Importances: {importances}

Identify operational risks, model overfit vulnerabilities, or data dependencies.
Provide your response in JSON format matching the schema below:
{{
    "severity": "Severity index ('High', 'Medium', 'Low')",
    "probability": "Likelihood index ('High', 'Medium', 'Low')",
    "description": "Vulnerability operational description details"
}}
"""

    CONFIDENCE = """
You are an Analytics Assurance Auditor.
Review the overall validation indicators:
Model cross-validation: {best_metrics}
Data completeness checks: {dataset_summary}

Assign a confidence score to the reports findings.
Provide your response in JSON format matching the schema below:
{{
    "confidence_score": 0.95,  # Float between 0.0 and 1.0
    "reliability_rating": "Rating classification ('High', 'Medium', 'Low')",
    "justification": "Structured audit reasoning justification statement"
}}
"""
