import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> dict:
        """Reduces the dimension of session.feature_matrix using PCA, t-SNE, or UMAP, and plots the results.

        Args:
            method: Reduction technique ("pca", "tsne", or "umap").
            n_components: Number of target dimensions (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).
            random_state: Random state seed for reproducibility (default 42).
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Run build_dtm_tool first."}
        
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)
            
        method_lower = method.lower()
        explained_variance_ratio = None
        
        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            # Using init="random", learning_rate=200.0 for exact compatibility with legacy sklearn TSNE defaults
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="random",
                learning_rate=200.0
            )
            coords = reducer.fit_transform(X_dense)
        elif method_lower == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)
        else:
            return {"error": f"Unknown dimensionality reduction method: {method}"}
        
        # Plot coordinates colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif labels is None and session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values
            
        df_plot = pd.DataFrame({
            "Dim1": coords[:, 0],
            "Dim2": coords[:, 1]
        })
        
        if labels is not None and len(labels) == len(coords):
            df_plot["Category"] = labels
            sns.scatterplot(data=df_plot, x="Dim1", y="Dim2", hue="Category", style="Category", ax=ax, s=100)
        else:
            sns.scatterplot(data=df_plot, x="Dim1", y="Dim2", ax=ax, s=100)
            
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()
        
        # Set pending figure on session
        session.pending_figure = fig
        
        result_id = session.next_result_id("reduction")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "coordinates": coords.tolist()
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio
            
        # Store artifact and result
        session.artifacts[f"reduced_coords_{method_lower}"] = coords
        session.store_result("reduce_dimensions_tool", {"method": method_lower}, summary)
        
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes / binarizes document category labels in session.dataframe.

        Returns a dictionary summary with the unique classes and the binarized binary matrix.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session."}
        
        df = session.dataframe
        if "category_name" in df.columns:
            raw_labels = df["category_name"].astype(str).tolist()
        elif "category" in df.columns:
            raw_labels = df["category"].astype(str).tolist()
        else:
            return {"error": "No 'category_name' or 'category' column found in session.dataframe."}
            
        lb = LabelBinarizer()
        binarized_matrix = lb.fit_transform(raw_labels)
        classes = lb.classes_.tolist()
        
        # If binary classification (single column produced by LabelBinarizer), expand to two indicator columns
        if len(classes) == 2 and binarized_matrix.shape[1] == 1:
            binarized_matrix = np.hstack([1 - binarized_matrix, binarized_matrix])
            
        result_id = session.next_result_id("binarized_labels")
        summary = {
            "result_id": result_id,
            "n_documents": len(raw_labels),
            "classes": classes,
            "binarized_matrix": binarized_matrix.tolist()
        }
        
        # Store artifact and result
        session.artifacts["binarized_labels"] = {
            "classes": classes,
            "matrix": binarized_matrix
        }
        session.store_result("binarize_labels_tool", {}, summary)
        
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
