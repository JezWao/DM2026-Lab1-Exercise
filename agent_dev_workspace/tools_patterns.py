import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI pattern mining modules
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth
import PAMI.extras.convert.DF2DB as DF2DB


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> dict:
        """Mines frequent term patterns for a single category using PAMI.

        Args:
            category_name: Name of the category to mine patterns for.
            filtering_method: Vocab filtering method ("variance", "tfidf", or "term_frequency").
            algorithm: Mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support count (for "fpgrowth" and "maxfpgrowth").
            k: Top-K count (for "topk").
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session."}
        
        # 1. Filter to specific category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No data found for category '{category_name}'."}
        
        # Drop empty texts if any
        texts = cat_df["text"].dropna().tolist()
        texts = [t for t in texts if t.strip()]
        if not texts:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category": category_name,
                "patterns": [],
                "note": "no non-empty texts in category"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary
        
        # 1. Build category-local DTM using CountVectorizer with English stopwords removed
        cv = CountVectorizer(stop_words="english")
        dtm = cv.fit_transform(texts).toarray()
        feature_names = cv.get_feature_names_out()
        
        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category": category_name,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary
        
        # 2. Apply chosen filtering_method
        # variance/term_frequency: middle 90% (5th - 95th percentile)
        # tfidf: top 80% (>= 20th percentile)
        kept_indices = []
        if filtering_method == "variance":
            variances = np.var(dtm, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            # Middle 90%: strictly between 5th and 95th percentile
            kept_indices = [i for i, v in enumerate(variances) if p5 < v < p95]
        elif filtering_method == "term_frequency":
            counts = np.sum(dtm, axis=0)
            p5 = np.percentile(counts, 5)
            p95 = np.percentile(counts, 95)
            kept_indices = [i for i, c in enumerate(counts) if p5 < c < p95]
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_indices = [i for i, score in enumerate(mean_tfidf) if score >= p20]
        else:
            return {"error": f"Unknown filtering method: {filtering_method}"}
        
        # 3. Check if any terms survived
        if not kept_indices:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category": category_name,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method}, summary)
            return summary
        
        kept_features = [feature_names[i] for i in kept_indices]
        filtered_dtm = dtm[:, kept_indices]
        
        # 4. Convert to transactional DB and mine patterns using PAMI
        binary_df = pd.DataFrame((filtered_dtm >= 1).astype(int), columns=kept_features)
        
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp_csv, \
             tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp_db:
            
            csv_path = tmp_csv.name
            db_path = tmp_db.name
            
        binary_df.to_csv(csv_path, index=False)
        
        # Convert DF to Transactional Database using PAMI DF2DB helper
        converter = DF2DB.DF2DB(csv_path, threshold=1, condition=">=")
        converter.convert2TransactionalDatabase(db_path)
        
        # Mine with chosen PAMI algorithm
        patterns = []
        if algorithm == "fpgrowth":
            miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            miner.startMine()
            df_res = miner.getPatternsAsDataFrame()
            if df_res is not None and not df_res.empty:
                for _, row in df_res.iterrows():
                    patterns.append({"pattern": row["Patterns"], "support": int(row["Support"])})
        elif algorithm == "topk":
            miner = FAE.FAE(db_path, k=k, sep="\t")
            miner.startMine()
            df_res = miner.getPatternsAsDataFrame()
            if df_res is not None and not df_res.empty:
                for _, row in df_res.iterrows():
                    patterns.append({"pattern": row["Patterns"], "support": int(row["Support"])})
        elif algorithm == "maxfpgrowth":
            miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
            miner.startMine()
            df_res = miner.getPatternsAsDataFrame()
            if df_res is not None and not df_res.empty:
                for _, row in df_res.iterrows():
                    patterns.append({"pattern": row["Patterns"], "support": int(row["Support"])})
        else:
            return {"error": f"Unknown algorithm: {algorithm}"}
        
        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns
        }
        
        # Store artifact and result
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns
        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
        
        return summary

    return [mine_patterns_tool]
