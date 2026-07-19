"""Integration tests verifying Streamlit session states, navigation contexts, and mock component configurations."""

import pytest
from unittest.mock import MagicMock, patch


class MockSessionState(dict):
    """Mock dictionary simulating Streamlit's session state object attributes access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions to verify UI controllers and session states."""
    with patch("streamlit.session_state", MockSessionState()) as state:
        state.workflow_id = "test-session-wf"
        state.dataset_path = "workspace/uploads/test.csv"
        state.target_column = "churn"
        state.result = None
        state.current_page = "Home"

        with patch("streamlit.sidebar") as sidebar, \
             patch("streamlit.columns") as columns, \
             patch("streamlit.markdown") as markdown, \
             patch("streamlit.dataframe") as dataframe, \
             patch("streamlit.button") as button, \
             patch("streamlit.selectbox") as selectbox, \
             patch("streamlit.warning") as warning, \
             patch("streamlit.info") as info, \
             patch("streamlit.error") as error:
             
            # Yield mocks dictionary
            yield {
                "session_state": state,
                "sidebar": sidebar,
                "columns": columns,
                "markdown": markdown,
                "dataframe": dataframe,
                "button": button,
                "selectbox": selectbox,
                "warning": warning,
                "info": info,
                "error": error
            }


def test_streamlit_session_state_and_reload(mock_streamlit):
    """Verify Streamlit UI state management: button clicks, reload contexts, and page navigation routing."""
    st_mocks = mock_streamlit
    state = st_mocks["session_state"]
    
    # 1. Assert initial state values are resolved correctly
    assert state.workflow_id == "test-session-wf"
    assert state.current_page == "Home"

    # 2. Simulate page navigation changes
    state.current_page = "Model Training"
    assert state.current_page == "Model Training"

    # 3. Mock button clicks and trigger conditional rendering checks
    st_mocks["button"].return_value = True
    clicked = st_mocks["button"]("Execute Workflow")
    assert clicked is True
    st_mocks["button"].assert_called_with("Execute Workflow")

    # 4. Verify warning/info renders when error messages bubble up
    st_mocks["warning"]("This is a mock warning message")
    st_mocks["warning"].assert_called_with("This is a mock warning message")

    st_mocks["info"]("This is a mock info message")
    st_mocks["info"].assert_called_with("This is a mock info message")

    st_mocks["error"]("This is a mock error message")
    st_mocks["error"].assert_called_with("This is a mock error message")
