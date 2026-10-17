"""
tools_data.py - Data ingestion and preprocessing tools for data mining pipeline.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    """
    Factory function to instantiate tools bound to the given session state.
    """

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """
        Tokenizes text documents into lowercase unigram tokens using NLTK word_tokenize.

        Args:

            text_column: Name of the text column in session.dataframe to tokenize.

        Returns:
            Dict containing result_id, total_documents, and status summary.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Tokenize text
        tokens_list = []
        for text in session.dataframe[text_column]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                tokens_list.append(tokens)

        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        res_id = session.next_result_id("tokenize")
        summary = {
            "result_id": res_id,
            "status": "success",
            "total_documents": len(tokens_list),
            "message": f"Successfully tokenized {len(tokens_list)} documents into 'unigrams' column.",
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and subdirectories in the workspace root or specified subdirectory.

        Args:

            subdirectory: Relative directory path inside the workspace to list.

        Returns:
            Dict containing listed files, directories, and total counts.
        """
        base_dir = os.getcwd()
        target_dir = os.path.normpath(os.path.join(base_dir, subdirectory))

        if not target_dir.startswith(base_dir):
            return {"error": "Access denied: directory path outside workspace scope."}

        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            return {"error": f"Directory path '{subdirectory}' does not exist."}

        entries = os.listdir(target_dir)
        files = []
        directories = []

        for entry in sorted(entries):
            full_path = os.path.join(target_dir, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                files.append(entry)

        res_id = session.next_result_id("list_files")
        summary = {
            "result_id": res_id,
            "status": "success",
            "subdirectory": subdirectory,
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_directories": len(directories),
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, position: str = "head") -> Dict[str, Any]:
        """
        Inspects the current working dataset by displaying a preview of rows (head or tail) along with summary details.

        Args:

            n: Number of rows to peek at (default 5).
            position: Whether to inspect 'head' or 'tail' of the dataframe (default 'head').

        Returns:
            Dict containing total_rows, total_columns, columns list, and preview records.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        if position == "tail":
            preview_df = df.tail(n)
        else:
            preview_df = df.head(n)

        # Convert preview to records (JSON-serializable)
        preview_records = preview_df.to_dict(orient="records")

        res_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": res_id,
            "status": "success",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "position": position,
            "preview_count": len(preview_records),
            "preview": preview_records,
        }

        session.store_result(
            "inspect_data_tool",
            {"n": n, "position": position},
            summary,
        )
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """
        Checks for missing, null, or empty string text values in the loaded dataset.

        Args:

            text_column: Name of the text column in session.dataframe to check.

        Returns:
            Dict containing total_rows, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        df = session.dataframe
        # Row is missing if value is NaN or empty/whitespace string
        missing_mask = df[text_column].isna() | df[text_column].astype(str).str.strip().eq("")
        missing_indices = df[missing_mask].index.tolist()
        missing_count = len(missing_indices)

        res_id = session.next_result_id("check_missing")
        summary = {
            "result_id": res_id,
            "status": "success",
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "message": f"Found {missing_count} missing or empty text values out of {len(df)} total rows.",
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        text_column: str = "text", drop: bool = False
    ) -> Dict[str, Any]:
        """
        Checks for duplicate document texts in the loaded dataset and optionally drops all copies of duplicate rows.

        Args:

            text_column: Name of the text column in session.dataframe to check for duplicates.
            drop: If True, drops all copies of duplicated rows using keep=False.

        Returns:
            Dict containing duplicate_count, duplicate_indices, initial_rows, remaining_rows, and status summary.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        df = session.dataframe
        # Identify duplicate rows (extra occurrences)
        duplicate_mask = df.duplicated(subset=[text_column], keep="first")
        duplicate_indices = df[duplicate_mask].index.tolist()
        duplicate_count = len(duplicate_indices)

        initial_rows = len(df)
        remaining_rows = initial_rows

        if drop and duplicate_count > 0:
            # Drop ALL copies of duplicate rows using keep=False matching Master pipeline specification
            session.dataframe.drop_duplicates(
                subset=[text_column], keep=False, inplace=True
            )
            session.dataframe.reset_index(drop=True, inplace=True)
            remaining_rows = len(session.dataframe)

        res_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": res_id,
            "status": "success",
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "remaining_rows": remaining_rows,
            "message": f"Found {duplicate_count} duplicate row(s)."
            + (
                f" Dropped all copies, remaining rows: {remaining_rows}."
                if drop
                else ""
            ),
        }

        session.store_result(
            "check_duplicates_tool",
            {"text_column": text_column, "drop": drop},
            summary,
        )
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Subsamples n random rows from the working DataFrame session.dataframe.

        Args:

            n: Number of rows to sample from session.dataframe.
            random_state: Optional seed for random number generator to ensure reproducibility.

        Returns:
            Dict containing result_id, initial_rows, sampled_rows, sampled_indices, and status summary.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        initial_rows = len(df)

        if n <= 0:
            return {"error": f"Sample size n={n} must be greater than 0."}

        if n > initial_rows:
            return {
                "error": f"Sample size n={n} cannot exceed available rows ({initial_rows})."
            }

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session dataframe to sampled subset and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        res_id = session.next_result_id("sample_data")
        summary = {
            "result_id": res_id,
            "status": "success",
            "initial_rows": initial_rows,
            "sampled_rows": n,
            "sampled_indices": sampled_indices,
            "random_state": random_state,
            "message": f"Successfully sampled {n} row(s) from {initial_rows} initial rows.",
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text", label_column: str = "category_name"
    ) -> Dict[str, Any]:
        """
        Computes summary statistics on text character lengths and generates a box plot grouped by category.

        Args:

            text_column: Name of the text column in session.dataframe.
            label_column: Name of the category label column for grouping stats and plot.

        Returns:
            Dict containing overall stats, per-category stats, and result summary.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Calculate character length for each document text
        df["text_length"] = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Overall describe stats
        overall_stats = df["text_length"].describe().to_dict()

        # Grouped describe stats if label column exists
        category_stats = {}
        effective_label_col = label_column
        if effective_label_col not in df.columns and "category" in df.columns:
            effective_label_col = "category"

        if effective_label_col in df.columns:
            grouped = df.groupby(effective_label_col)["text_length"]
            for cat_name, cat_series in grouped:
                category_stats[str(cat_name)] = cat_series.describe().to_dict()

        # Generate box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if effective_label_col in df.columns:
            sns.boxplot(data=df, x=effective_label_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        # Store pending plot on session object
        session.pending_figure = fig

        res_id = session.next_result_id("describe_data")
        summary = {
            "result_id": res_id,
            "status": "success",
            "total_documents": len(df),
            "overall_stats": overall_stats,
            "category_stats": category_stats,
            "message": f"Computed describe stats over {len(df)} documents and generated box plot.",
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "label_column": label_column},
            summary,
        )
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
