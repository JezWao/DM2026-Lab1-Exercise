import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> dict:
        """Computes the cosine similarity between two documents using session.feature_matrix.

        Args:
            doc_index_1: Index of the first document (0-based).
            doc_index_2: Index of the second document (0-based).
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Run build_dtm_tool first."}
        
        n_docs = session.feature_matrix.shape[0]
        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"doc_index_1 ({doc_index_1}) out of bounds (0 to {n_docs - 1})."}
        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"doc_index_2 ({doc_index_2}) out of bounds (0 to {n_docs - 1})."}
            
        # Extract document vectors
        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]
        
        if hasattr(vec1, "toarray"):
            vec1_dense = vec1.toarray()
            vec2_dense = vec2.toarray()
        else:
            vec1_dense = np.asarray(vec1)
            vec2_dense = np.asarray(vec2)
            
        # Handle empty text / zero vectors explicitly to avoid NaN
        norm1 = np.linalg.norm(vec1_dense)
        norm2 = np.linalg.norm(vec2_dense)
        
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim_val = cosine_similarity(vec1_dense, vec2_dense)[0, 0]
            sim = float(sim_val)
            
        result_id = session.next_result_id("cosine")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(sim, 4)
        }
        
        # Store artifact and result
        session.artifacts[f"cosine_{doc_index_1}_{doc_index_2}"] = sim
        session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
        
        return summary

    return [cosine_similarity_tool]
