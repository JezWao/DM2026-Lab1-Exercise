"""
Unit tests for data tools in tools_data.py using agent_dev/sample_fixture.csv.
"""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool():
    # 1. Initialize session and load load_dataset_tool from premade_tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get student tools bound to the same session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # 3. Invoke check_missing_tool
    res = check_missing_tool.invoke({"text_column": "text"})

    # 4. Assert against known answer #2 in TEST_FIXTURE.md
    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
