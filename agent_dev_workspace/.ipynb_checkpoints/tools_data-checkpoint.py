import nltk
from typing import Dict, Any
from langchain_core.tools import tool

# Ensure necessary NLTK tokenizer models are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text documents in session.dataframe into lowercased unigrams using NLTK.

        Stores the resulting token lists in session.dataframe['unigrams'] and session.artifacts['tokens'].
        Returns a summary including document count, total token count, and average tokens per document.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please run load_dataset_tool first.",
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Column 'text' not found in session.dataframe.",
            }

        def tokenize_text(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return nltk.word_tokenize(text.lower())

        tokens_list = session.dataframe["text"].apply(tokenize_text).tolist()
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        total_tokens = sum(len(doc_tokens) for doc_tokens in tokens_list)
        n_docs = len(tokens_list)
        avg_tokens = float(total_tokens / n_docs) if n_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4),
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    return [tokenize_tool]
