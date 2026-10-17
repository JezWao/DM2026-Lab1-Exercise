import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_tool_pca():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using build_dtm_tool logic
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # Test PCA (Known Answer #7 in TEST_FIXTURE.md)
    res_pca = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    
    assert res_pca["method"] == "pca"
    coords = np.array(res_pca["coordinates"])
    
    # Known PC1 and PC2 coordinates from TEST_FIXTURE.md
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    
    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)
    np.testing.assert_allclose(res_pca["explained_variance_ratio"], [0.7532, 0.1965], atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tool_tsne():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # Test t-SNE (Known Answer #8 in TEST_FIXTURE.md)
    res_tsne = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    
    assert res_tsne["method"] == "tsne"
    coords = np.array(res_tsne["coordinates"])
    
    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])
    
    np.testing.assert_allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_reduce_dimensions_tool_umap():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # Test UMAP (Known Answer #9 in TEST_FIXTURE.md)
    res_umap = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    
    assert res_umap["method"] == "umap"
    coords = np.array(res_umap["coordinates"])
    
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    np.testing.assert_allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    reduction_tools = make_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_labels_tool.invoke({})
    assert res["n_documents"] == 8
    assert res["classes"] == ["catA", "catB"]
    
    matrix = np.array(res["binarized_matrix"])
    assert matrix.shape == (8, 2)
    # The true order of labels in sample_fixture.csv is: [catA, catA, catA, catB, catB, catB, catB, catA]
    expected_matrix = np.array([
        [1, 0], # catA
        [1, 0], # catA
        [1, 0], # catA
        [0, 1], # catB
        [0, 1], # catB
        [0, 1], # catB
        [0, 1], # catB
        [1, 0]  # catA
    ])
    np.testing.assert_array_equal(matrix, expected_matrix)
