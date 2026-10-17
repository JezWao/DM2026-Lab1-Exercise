"""
tools_dtm.py - Document-Term Matrix (DTM) tools for data mining pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
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

    return [build_dtm_tool]
