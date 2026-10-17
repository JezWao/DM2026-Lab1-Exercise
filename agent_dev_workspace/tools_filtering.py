"""
tools_filtering.py - Feature filtering tools for data mining pipeline.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
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

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates Pearson correlation between each term's count and a binary target class indicator.

        Args:
            target_class: Category name to evaluate correlation against (e.g. 'catB').
            threshold: Optional float threshold to filter terms by absolute correlation |r| >= threshold.

        Returns:
            Dict containing result_id, target_class, threshold, and term correlation values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix not found. Run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "Labels not found in session state. Run build_dtm_tool first."}

        labels_arr = np.array(session.labels)
        if target_class not in labels_arr:
            return {"error": f"Target class '{target_class}' not found in dataset labels."}

        dense = session.feature_matrix.toarray()
        y_binary = (labels_arr == target_class).astype(float)

        correlations = []
        for col_idx in range(dense.shape[1]):
            x = dense[:, col_idx]
            if np.std(x) == 0 or np.std(y_binary) == 0:
                r = 0.0
            else:
                r_val = np.corrcoef(x, y_binary)[0, 1]
                r = 0.0 if np.isnan(r_val) else float(r_val)
            correlations.append(r)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations,
        })

        # Sort descending by correlation coefficient, then alphabetically
        df_report = df_report.sort_values(by=["pearson_r", "term"], ascending=[False, True]).reset_index(drop=True)

        if threshold is not None:
            kept_mask = df_report["pearson_r"].abs() >= threshold
            kept_terms = df_report[kept_mask]["term"].tolist()
            removed_terms = df_report[~kept_mask]["term"].tolist()
        else:
            kept_terms = df_report["term"].tolist()
            removed_terms = []

        corr_dict = {
            term: float(round(r, 4))
            for term, r in zip(session.feature_names, correlations)
        }

        res_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": res_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_feature_names": kept_terms,
            "removed_feature_names": removed_terms,
            "correlations": corr_dict,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates Spearman rank correlation between each term's count and a binary target class indicator.

        Args:
            target_class: Category name to evaluate correlation against (e.g. 'catB').
            threshold: Optional float threshold to filter terms by absolute rank correlation |r| >= threshold.

        Returns:
            Dict containing result_id, target_class, threshold, and term Spearman correlation values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix not found. Run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "Labels not found in session state. Run build_dtm_tool first."}

        labels_arr = np.array(session.labels)
        if target_class not in labels_arr:
            return {"error": f"Target class '{target_class}' not found in dataset labels."}

        dense = session.feature_matrix.toarray()
        y_binary = (labels_arr == target_class).astype(float)

        correlations = []
        for col_idx in range(dense.shape[1]):
            x = dense[:, col_idx]
            res = spearmanr(x, y_binary)
            rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
            correlations.append(rho)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations,
        })

        # Sort descending by Spearman correlation coefficient, then alphabetically
        df_report = df_report.sort_values(by=["spearman_r", "term"], ascending=[False, True]).reset_index(drop=True)

        if threshold is not None:
            kept_mask = df_report["spearman_r"].abs() >= threshold
            kept_terms = df_report[kept_mask]["term"].tolist()
            removed_terms = df_report[~kept_mask]["term"].tolist()
        else:
            kept_terms = df_report["term"].tolist()
            removed_terms = []

        corr_dict = {
            term: float(round(r, 4))
            for term, r in zip(session.feature_names, correlations)
        }

        res_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": res_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_feature_names": kept_terms,
            "removed_feature_names": removed_terms,
            "correlations": corr_dict,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
