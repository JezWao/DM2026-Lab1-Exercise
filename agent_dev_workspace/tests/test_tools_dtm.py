"""
tests/test_tools_dtm.py - Unit tests for tools_dtm.py using sample_fixture.csv
"""

import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    """
    Tests build_dtm_tool against known answer #4 in TEST_FIXTURE.md.
    Vocabulary: alpha, always, beta, delta, gamma (5 terms)
    Sparsity: 21 non-zero out of 40 total elements -> 47.5% sparsity
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({
        "text_column": "text",
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
    })

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["vocabulary_size"] == 5
    assert res["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5)


def test_term_frequency_tool():
    """
    Tests term_frequency_tool against known answer #14 in TEST_FIXTURE.md:
    alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_terms"] == 5

    freq_dict = {item["term"]: item["frequency"] for item in res["term_frequencies"]}
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert freq_dict == expected_freqs


def test_dtm_heatmap_tool_full():
    """
    Tests dtm_heatmap_tool against known answer #21 in TEST_FIXTURE.md (full table).
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})
    res = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res["status"] == "success"
    assert res["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert len(res["document_indices"]) == 8

    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert res["matrix_slice"] == expected_matrix
    assert session.pending_figure is not None


def test_dtm_heatmap_tool_sliced():
    """
    Tests dtm_heatmap_tool slicing (n_terms=2, n_documents=3).
    Expected top-left 3x2 corner: [[3, 1], [2, 1], [1, 1]]
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})
    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res["status"] == "success"
    assert res["term_labels"] == ["alpha", "always"]
    assert len(res["document_indices"]) == 3
    assert res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
