"""
tools_filtering.py - Feature filtering tools for data mining pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function to instantiate feature filtering tools bound to the given session state.
    """

    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Flags near-constant terms by variance for potential removal.

        Args:
            threshold: Minimum variance threshold. Terms with variance < threshold are flagged for removal.

        Returns:
            Dict containing result_id, threshold, total_terms, kept_terms, removed_terms, kept_feature_names, removed_feature_names, and variances.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix not found. Run build_dtm_tool first."}

        dense = session.feature_matrix.toarray()
        variances = np.var(dense, axis=0, ddof=0)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances,
        })

        # Sort descending by variance, then alphabetically by term
        df_report = df_report.sort_values(by=["variance", "term"], ascending=[False, True]).reset_index(drop=True)

        kept_mask = df_report["variance"] >= threshold
        kept_terms = df_report[kept_mask]["term"].tolist()
        removed_terms = df_report[~kept_mask]["term"].tolist()

        variances_dict = {
            term: float(round(var, 4))
            for term, var in zip(session.feature_names, variances)
        }

        res_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": res_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_feature_names": kept_terms,
            "removed_feature_names": removed_terms,
            "variances": variances_dict,
        }

        # Store result with full_report formatted (term, variance) for visualize_result_tool
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool]
