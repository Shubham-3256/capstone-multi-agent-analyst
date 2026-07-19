"""Upload and Workflow Execution page."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.notifications import notify_result, show_error
from app.components.sidebar import setup_page
from app.components.upload import upload_dataset
from app.components.workflow_status import LiveProgressViewer, render_workflow_status
from app.services.session import (
    get_workflow_result,
    initialize_session,
    set_uploaded_dataset_path,
    set_workflow_result,
)
from app.services.workflow_service import WorkflowService


def main() -> None:
    """Render the upload and run execution dashboard page."""
    initialize_session()
    setup_page("Upload & Run")

    st.title("📂 Ingestion & Execution Panel")
    st.markdown("""
        Upload your structured dataset (CSV, Excel, or Parquet) and select the target variable. 
        If you select a target, the workflow will run automated feature engineering and model training. 
        If left blank, the pipeline will skip modeling and jump to visualizations and insights.
        """)
    st.divider()

    uploaded_file = upload_dataset()

    if uploaded_file:
        # Save upload to upload folder
        try:
            saved_path = WorkflowService.save_upload(uploaded_file)
            set_uploaded_dataset_path(str(saved_path))

            # Read columns for target selection
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)

            columns = [""] + list(df.columns)

            st.divider()
            st.subheader("Configure Pipeline Parameters")

            selected_target = st.selectbox(
                "🎯 Select Target Variable (Optional)",
                options=columns,
                index=0,
                help="Variable to predict. Leaving this blank skips ML/AutoML modeling.",
            )

            st.session_state.target_column = (
                selected_target if selected_target else None
            )

            # Render Execute Button
            st.markdown("### Run Analytics Engine")
            run_btn = st.button("🚀 Run Workflow Pipeline", width="stretch")

            if run_btn:
                # Initialize live progress view
                progress_viewer = LiveProgressViewer()

                # Execute graph run
                with st.spinner("Processing workflow nodes..."):
                    service = WorkflowService()
                    # Run target column
                    target = st.session_state.target_column
                    result = service.run(
                        dataset_path=str(saved_path),
                        target_column=target,
                        on_event=progress_viewer,
                    )

                    set_workflow_result(result)
                    notify_result(result)

            # If workflow result exists, render audit summary
            current_result = get_workflow_result()
            if current_result:
                st.divider()
                render_workflow_status(current_result)

        except Exception as e:
            show_error(f"Error initializing dataset upload: {str(e)}")


if __name__ == "__main__":
    main()
