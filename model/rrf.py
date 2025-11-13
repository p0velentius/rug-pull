import pandas as pd
import numpy as np

def rrf(bge: pd.DataFrame,
        bm25: pd.DataFrame,
        k: int = 60,
        alpha: float = 0.6,
        weight_bge_rrf: float = 1.0,
        weight_bm25_rrf: float = 1.0,
        weight_bge_score: float = 1.0,
        weight_bm25_score: float = 1.0,
        n: int = 52) -> pd.DataFrame:
    """
    bge/bm: columns ['question_id','website_id','rank','score']
    alpha: вес RRF в итоговом скоре; (1-alpha) — вес линейного комбинирования нормализованных скорoв
    return top-n websites for all requests
    """

    # rrf
    merged = pd.merge(bge, bm25, on=["question_id", "website_id"], how="outer", suffixes=("_bge", "_bm25"))
    merged["score_rrf_part"] = weight_bge_rrf / (k + merged["rank_bge"]) + weight_bm25_rrf / (k + merged["rank_bm25"])

    # score
    def norm_group(g, col):
        """
        normalize raw scores per question to [0,1] (min-max) — avoid division by zero.
        """
        vals = g[col].to_numpy(dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            return (vals - vmin) / (vmax - vmin)
        else:
            return np.zeros_like(vals)

    merged["score_bge_norm"] = merged.groupby("question_id", group_keys=False).apply(lambda g: pd.Series(norm_group(g, "score_bge"), index=g.index))
    merged["score_bm25_norm"]  = merged.groupby("question_id", group_keys=False).apply(lambda g: pd.Series(norm_group(g, "score_bm25"),  index=g.index))
    merged["score_raw_combined"] = weight_bge_score * merged["score_bge_norm"] + weight_bm25_score * merged["score_bm25_norm"]  
    
    # combination
    merged["score_rrf"] = alpha * merged["score_rrf_part"] + (1 - alpha) * merged["score_raw_combined"]
    merged["rank_rrf"] = merged.groupby("question_id")["score_rrf"].rank(method="first", ascending=False).astype(int)
    merged = merged.sort_values(["question_id", "score_rrf"], ascending=[True, False])

    return merged.groupby("question_id").head(n).reset_index(drop=True)
