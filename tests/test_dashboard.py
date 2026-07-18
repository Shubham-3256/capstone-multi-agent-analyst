"""Unit tests verifying import and basic execution of dashboard pages."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def get_page_main(page_name: str):
    """Dynamically load page modules starting with digits."""
    module = importlib.import_module(f"app.pages.{page_name}")
    return getattr(module, "main")


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
def test_home_page_execution(mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that app/Home.py main executes without raising exceptions."""
    from app.Home import main
    # Stub database reads to return empty lists
    with patch("app.services.history_service.HistoryService.workflows", return_value=[]), \
         patch("app.services.history_service.HistoryService.reports", return_value=[]), \
         patch("app.services.history_service.HistoryService.datasets", return_value=[]):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.file_uploader", return_value=None)
def test_upload_page_execution(mock_uploader, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 1_Upload.py main executes without raising exceptions when no file is uploaded."""
    main = get_page_main("1_Upload")
    main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_data_profile_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 2_Data_Profile.py displays info when no result is loaded."""
    main = get_page_main("2_Data_Profile")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_feature_engineering_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 3_Feature_Engineering.py displays info when no result is loaded."""
    main = get_page_main("3_Feature_Engineering")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_model_training_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 4_Model_Training.py displays info when no result is loaded."""
    main = get_page_main("4_Model_Training")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_visualizations_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 5_Visualizations.py displays info when no result is loaded."""
    main = get_page_main("5_Visualizations")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_business_insights_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 6_Business_Insights.py displays info when no result is loaded."""
    main = get_page_main("6_Business_Insights")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_reports_page_empty_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 7_Reports.py displays info when no result is loaded."""
    main = get_page_main("7_Reports")
    with patch("app.services.session.get_workflow_result", return_value=None):
        main()


@patch("streamlit.set_page_config")
@patch("streamlit.sidebar")
@patch("streamlit.title")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.info")
def test_history_page_execution(mock_info, mock_div, mock_md, mock_title, mock_sidebar, mock_set_page_config):
    """Test that 8_History.py executes successfully and queries workflow history."""
    main = get_page_main("8_History")
    with patch("app.services.history_service.HistoryService.workflows", return_value=[]):
        main()
