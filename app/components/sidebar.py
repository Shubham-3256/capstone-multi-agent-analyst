"""Shared dashboard sidebar."""

from pathlib import Path
import streamlit as st

from app.services.config import ASSETS_DIR


def render_sidebar() -> None:
    """Render branding and a compact workflow readiness summary."""
    from app.services.session import get_uploaded_dataset_path, get_workflow_result
    from pathlib import Path

    with st.sidebar:
        logo = ASSETS_DIR / "logo.png"
        if logo.exists():
            st.image(str(logo), width=72)
        st.title("AI Data Analyst")
        st.caption("Enterprise analytics workspace")
        st.divider()
        
        # Display Upload Status
        dataset_path = get_uploaded_dataset_path()
        if dataset_path:
            st.success(f"📂 **Active Dataset:**  \n`{Path(dataset_path).name}`")
        else:
            st.info("📂 No dataset active. Please upload one.")
            
        # Display Run Status
        result = get_workflow_result()
        if result:
            status = "completed" if result.is_success else "degraded"
            color = "green" if result.is_success else "orange"
            st.markdown(f"⚙️ **Status:** :{color}[{status.title()}]")
            
            # Show quick summary
            state = getattr(result, "state", None)
            if state:
                st.markdown(f"⏱️ **Duration:** {sum(state.timing.values()):.2f}s")
                st.markdown(f"🤖 **Model:** {getattr(state.ml_result, 'best_model_name', 'None')}")
        else:
            st.markdown("⚙️ **Status:** :red[Idle]")

        st.divider()
        st.caption("Use the sidebar navigation pages to inspect each phase of the AI analyst pipeline.")


def setup_page(title: str) -> None:
    """Configure page title, layout, inject custom styles, and render the sidebar."""
    st.set_page_config(page_title=f"{title} - AI Data Analyst", layout="wide")
    
    # Load CSS
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass
            
    # Render Sidebar
    render_sidebar()


