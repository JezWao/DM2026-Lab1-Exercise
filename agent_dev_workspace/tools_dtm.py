"""
tools_dtm.py - Document-Term Matrix (DTM) tools for data mining pipeline.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function to instantiate DTM tools bound to the given session state.
    """

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) from session.dataframe using CountVectorizer.

        Args:
            text_column: Name of the column containing document text (default 'text').
            ngram_range: List of 2 ints [min_n, max_n] for n-gram range (default [1, 1]).
            min_df: Minimum document frequency for terms to be included (default 1).
            max_df: Maximum document frequency proportion or count for terms (default 1.0).

        Returns:
            Dict containing result_id, matrix shape, non_zero count, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Convert list to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range) if isinstance(ngram_range, list) else (1, 1)

        # Handle missing/NaN text by replacing with empty string
        text_data = session.dataframe[text_column].fillna("").astype(str)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
        )

        dtm = vectorizer.fit_transform(text_data)
        feature_names = list(vectorizer.get_feature_names_out())

        # Store in session state
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        res_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": res_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names": feature_names,
        }

        session.store_result(
            "build_dtm_tool",
            {
                "text_column": text_column,
                "ngram_range": ngram_range,
                "min_df": min_df,
                "max_df": max_df,
            },
            summary,
        )
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """
        Renders a heatmap visualization of a positional slice of the document-term matrix.

        Args:
            n_terms: Number of terms (columns) to slice from the beginning of session.feature_names (default 20).
            n_documents: Number of documents (rows) to slice from session.dataframe (default 20).

        Returns:
            Dict containing result_id, status, sliced matrix data, term labels, and document indices.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix not found. Run build_dtm_tool first."}

        # Slice matrix up to n_documents and n_terms
        sliced_matrix_sparse = session.feature_matrix[0:n_documents, 0:n_terms]
        sliced_matrix_dense = sliced_matrix_sparse.toarray()

        term_labels = session.feature_names[0:n_terms]
        doc_indices = [f"Doc_{i}" for i in range(sliced_matrix_dense.shape[0])]

        # Create figure and heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            sliced_matrix_dense,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=term_labels,
            yticklabels=doc_indices,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap (Top {len(doc_indices)} Docs x {len(term_labels)} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Set figure on session.pending_figure
        session.pending_figure = fig

        res_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": res_id,
            "status": "success",
            "matrix_slice": sliced_matrix_dense.tolist(),
            "term_labels": term_labels,
            "document_indices": doc_indices,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {
                "n_terms": n_terms,
                "n_documents": n_documents,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Optional int to return only the top N most frequent terms.

        Returns:
            Dict containing result_id, status, total_terms, and list of term frequency records.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix not found. Run build_dtm_tool first."}

        # Column sums across feature_matrix
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": col_sums,
        })

        # Sort descending by frequency, then alphabetically by term
        df_freq = df_freq.sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        total_terms = len(df_freq)

        if top_n is not None and top_n > 0:
            df_report = df_freq.head(top_n)
        else:
            df_report = df_freq

        records = df_report.to_dict(orient="records")

        res_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": res_id,
            "status": "success",
            "total_terms": total_terms,
            "returned_terms": len(records),
            "term_frequencies": records,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_report,
        )
        return summary

    return [build_dtm_tool, dtm_heatmap_tool, term_frequency_tool]
