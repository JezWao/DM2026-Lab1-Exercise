import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20
    ) -> dict:
        """Computes and visualizes the Pearson correlation matrix over the top terms by variance.

        Args:
            top_n: Number of highest-variance terms to include (default 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."}
            
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)
            
        feature_names = np.array(session.feature_names)
        
        # 1. Compute term variances across all documents
        variances = np.var(X_dense, axis=0)
        
        # Select top N features by variance
        n_features = len(feature_names)
        actual_k = min(top_n, n_features)
        
        # Indices of top variance terms in descending order
        top_indices = np.argsort(variances)[::-1][:actual_k]
        top_terms = feature_names[top_indices].tolist()
        top_data = X_dense[:, top_indices]
        
        # 2. Compute Pearson correlation matrix across top terms
        # np.corrcoef expects variables as rows, so transpose top_data
        corr_matrix = np.corrcoef(top_data, rowvar=False)
        # Handle single feature case or 0 variance
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 3. Build annotated heatmap and set session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1,
            vmax=1
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_k} Variance Terms)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        
        session.pending_figure = fig
        
        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": np.round(corr_matrix, 4).tolist()
        }
        
        # Store artifact and result
        session.artifacts["feature_correlation_matrix"] = {
            "terms": top_terms,
            "matrix": corr_matrix
        }
        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
