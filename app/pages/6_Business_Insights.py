"""Business Insights dashboard screen."""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.sidebar import setup_page
from app.components.cards import info_card
from app.components.tables import render_html_table
from app.services.session import initialize_session, get_workflow_result


def main() -> None:
    """Render the business insights page."""
    initialize_session()
    setup_page("Business Insights")

    st.title("💡 Strategic Business Insights")

    result = get_workflow_result()
    if not result:
        st.info("No active workflow result. Please upload a dataset and run the pipeline on the Upload page.")
        return

    biz_result = getattr(result.state, "business_result", None)
    if not biz_result:
        st.warning("No business insights reports found in the execution state.")
        return

    # 1. Headline & Executive Summary
    st.markdown(f"### 📢 Business Headline")
    st.info(f"**{biz_result.executive_summary.headline}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Takeaways")
        for idx, item in enumerate(biz_result.executive_summary.key_takeaways, 1):
            st.markdown(f"**{idx}.** {item}")
    with col2:
        st.subheader("Operational Business Impact")
        st.markdown(f"_{biz_result.executive_summary.impact_statement}_")

    st.divider()

    # 2. Recommendations checklist
    st.subheader("📋 Strategic Recommendations Checklist")
    headers = ["Recommendation", "Description", "Actionability"]
    rows = []
    for item in biz_result.recommendations:
        act = item.actionability.upper()
        color = "green" if act == "HIGH" else "orange" if act == "MEDIUM" else "blue"
        tag = f":{color}[**{act}**]"
        rows.append([
            f"**{item.title}**",
            item.description,
            tag
        ])
    render_html_table(headers, rows)

    st.divider()

    # 3. Operational Risks
    st.subheader("⚠️ Operational Risk Factors & Overfitting Audits")
    r_headers = ["Risk Description", "Severity", "Probability"]
    r_rows = []
    for item in biz_result.risks:
        sev = item.severity.upper()
        prob = item.probability.upper()
        
        s_color = "red" if sev == "HIGH" else "orange" if sev == "MEDIUM" else "blue"
        p_color = "red" if prob == "HIGH" else "orange" if prob == "MEDIUM" else "blue"
        
        r_rows.append([
            item.description,
            f":{s_color}[**{sev}**]",
            f":{p_color}[**{prob}**]"
        ])
    render_html_table(r_headers, r_rows)

    st.divider()

    # 4. Confidence Report
    st.subheader("🕵️ Analysis Confidence & Verification Audit")
    
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        conf = biz_result.confidence_report
        rating = conf.reliability_rating.upper()
        r_color = "green" if rating == "HIGH" else "orange" if rating == "MEDIUM" else "red"
        
        info_card(
            title="Reliability Rating",
            value=rating,
            subtitle=f"Confidence Score: {conf.confidence_score * 100:.1f}%",
            icon="🛡️",
            bg_gradient=f"linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
        )
    with col_c2:
        st.markdown(f"**Verification Justification:**")
        st.markdown(conf.justification)

    # 5. Technical insights & token logs
    with st.expander("🛠️ LLM Session Telemetry Details", expanded=False):
        st.markdown(
            f"""
            * **Input Prompt Tokens:** `{biz_result.token_usage.get('input_tokens', 0):,}`
            * **Output Response Tokens:** `{biz_result.token_usage.get('output_tokens', 0):,}`
            * **Estimated Cost (USD):** `${biz_result.estimated_cost_usd:.5f}`
            """
        )


if __name__ == "__main__":
    main()
